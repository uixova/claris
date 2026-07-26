# -*- coding: utf-8 -*-
"""
Claris ile sohbet/çıkarım tarafı... eğittiğimiz Transformer'la akıcı Türkçe üretiyor.

train_claris.py ile eğittiğimiz modeli (models/claris_model.pt) ve BPE'yi yüklüyor,
sonra top-k/top-p (nucleus) örneklemeyle cevap yazıyor. Sohbet yapısını rol tokenlarıyla
(<user>/<bot>) kuruyor, yani kim ne dedi belli oluyor.

  python claris_chat.py                      # oturup sohbet et
  python claris_chat.py "merhaba nasılsın"   # tek soru sor, cevabını al
"""

import os
import sys
import torch
from torch.nn import functional as F

import bpe as bpemod
import train_claris as T

import time

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
MODELS = os.path.join(ROOT, "models")
LOG_DIR = os.path.join(ROOT, "logs")
# Sohbet logu KAPALI gelir (gizlilik). Açmak: CLARIS_LOG=1 python claris_chat.py
LOG_CHAT = os.environ.get("CLARIS_LOG", "0") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"


def log_chat(user, bot):
    """Soru/cevabı logs/claris_chat.jsonl'e ekler (CLARIS_LOG=1 ise). Çıktı kalitesini
    sonradan incelemek için. Eğitime/çıkarıma etkisi yok."""
    if not LOG_CHAT:
        return
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        import json
        with open(os.path.join(LOG_DIR, "claris_chat.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "user": user, "bot": bot}, ensure_ascii=False) + "\n")
    except Exception:
        pass                                  # log hatası sohbeti bozmasın

# HAFİF ÇALIŞ: tüm CPU'yu sömürme, sistemi yorma. Model yalnızca çağrılınca
# yüklenir, iş bitince çıkar — daemon yok, sisteme bağlanma yok ("hazıra beklet").
try:
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    torch.set_grad_enabled(False)
except Exception:
    pass


def load(bpe_path=None, ckpt_path=None):
    bpe_path = bpe_path or os.path.join(MODELS, "bpe.json")
    # SFT modeli varsa onu tercih ediyoruz (sohbet için daha iyi), yoksa pretrain modeline düşüyor
    if ckpt_path is None:
        order = ("claris_sft.pt", "claris_model.pt")
        ckpt_path = next((os.path.join(MODELS, n) for n in order
                          if os.path.exists(os.path.join(MODELS, n))),
                         os.path.join(MODELS, "claris_model.pt"))
    if "sft" in os.path.basename(ckpt_path):
        print("[model] SFT modeli yüklendi (sohbet/cevap modu).")
    if not (os.path.exists(bpe_path) and os.path.exists(ckpt_path)):
        print("Model yok. Önce train_claris.py (pretrain) sonra train_sft.py "
              "ile eğit, bpe.json + claris_model.pt / claris_sft.pt'yi models/ koy.")
        sys.exit(0)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    # model mimarisini checkpoint'e göre ayarla (train_transformer global'lerini geçersiz kıl)
    T.N_EMBD, T.N_HEAD, T.N_LAYER, T.CONTEXT_LEN = (
        cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["context"])
    T.DROPOUT = 0.0
    tok = bpemod.BPE().load(bpe_path)
    model = T.Transformer(cfg["vocab"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return tok, model, cfg


@torch.no_grad()
def generate(tok, model, cfg, prompt, max_new=120, temperature=0.8, top_k=40,
             top_p=0.9, rep_penalty=1.3, no_repeat_ngram=3):
    ctx = cfg["context"]
    u, b, eos, bos = tok.tok("<user>"), tok.tok("<bot>"), tok.tok("<eos>"), tok.tok("<bos>")
    if None in (u, b, eos, bos):
        # Bozuk/eski bpe.json: özel token yoksa aşağıda anlaşılmaz TypeError patlar -> net dur.
        raise RuntimeError("bpe.json özel tokenları eksik (<user>/<bot>/<eos>/<bos>) — "
                           "eğitimde kullanılan bpe.json'u models/ içine koy.")
    ids = [bos, u] + tok.encode(prompt) + [b]
    start = len(ids)
    prompt_toks = set(ids[2:start - 1])            # kullanıcı mesajı tokenları (echo'yu engelle)
    # KV CACHE: prompt bir kez prefill edilir, sonra adım başına TEK token forward
    # -> tam-bağlam forward'a göre kat kat hızlı. Bağlam dolarsa (pos>=ctx) eski
    # kayan-pencere yoluna düşer (ids[-ctx:] semantiği cache ile temsil edilemez).
    # Kapatmak: CLARIS_NO_KVCACHE=1 (eski yol birebir durur).
    use_cache = os.environ.get("CLARIS_NO_KVCACHE", "0") != "1" and len(ids) < ctx
    caches, pos = None, 0
    for _ in range(max_new):
        if use_cache and caches is not None and pos >= ctx:
            use_cache, caches = False, None        # cache doldu -> kayan pencere fallback
        if use_cache:
            if caches is None:                     # prefill: tüm prompt tek seferde
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits_row, caches = model.forward_infer(x, None, 0)
                pos = len(ids)
            else:                                  # decode: sadece son token
                x = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                logits_row, caches = model.forward_infer(x, caches, pos)
                pos += 1
            logits = logits_row[0]
        else:
            x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]               # targets yok -> forward logits döner
        # temperature<=0 = GREEDY (argmax). 1e-5'e bölüp dev logit üretmek yerine
        # örnekleme komple atlanır (ceza/yasaklar yine uygulanır).
        greedy = temperature <= 0
        if not greedy:
            logits = logits / temperature

        gen = ids[start:]                          # üretilen yanıt (loop önleme bunun üstünde)
        # Tekrar cezası: son üretilen tokenlar + PROMPT tokenları kıs ("Merhaba Claris" echo'su gider)
        if rep_penalty and rep_penalty != 1.0:
            for t in set(gen[-60:]) | prompt_toks:
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        # no-repeat-ngram: aynı n-gram'ı YASAKLA — TÜM geçmişi (prompt+yanıt) tara
        # ("Hava nasıl Claris?" gibi prompt'u aynen kopyalamayı kırar)
        hist = ids[1:]                             # bos hariç (prompt + üretilen)
        if no_repeat_ngram and len(hist) >= no_repeat_ngram - 1:
            prefix = tuple(hist[-(no_repeat_ngram - 1):])
            for i in range(len(hist) - no_repeat_ngram + 1):
                if tuple(hist[i:i + no_repeat_ngram - 1]) == prefix:
                    logits[hist[i + no_repeat_ngram - 1]] = float("-inf")
        if greedy:
            nxt = int(torch.argmax(logits).item())
        else:
            # top-k
            if top_k:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            # top-p (nucleus)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                mask = cum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sp[mask] = 0.0
                probs = torch.zeros_like(probs).scatter_(0, si, sp)
                probs = probs / probs.sum()
            nxt = torch.multinomial(probs, 1).item()
        if nxt in (eos, u):
            break
        ids.append(nxt)
    return _capitalize_tr(tok.decode(ids[start:]).strip())


# Özel isim sözlüğü — FALLBACK çekirdek (truecase.json yokken de temel isimler doğru
# çıksın). ASIL kaynak: `python truecase.py build` -> models/truecase.json (korpustan
# on binlerce özel ismi otomatik öğrenir; bu elle listeyi genişletmeye gerek YOK).
# Türkçe-doğru biçimleriyle: İstanbul/İzmir/İran (İ noktalı), Irak (I noktasız).
_PROPER = {
    "claris": "Claris", "türkiye": "Türkiye", "türk": "Türk", "türkçe": "Türkçe",
    "türkler": "Türkler", "atatürk": "Atatürk", "istanbul": "İstanbul",
    "ankara": "Ankara", "izmir": "İzmir", "bursa": "Bursa", "antalya": "Antalya",
    "adana": "Adana", "konya": "Konya", "gaziantep": "Gaziantep", "mersin": "Mersin",
    "kayseri": "Kayseri", "eskişehir": "Eskişehir", "diyarbakır": "Diyarbakır",
    "samsun": "Samsun", "trabzon": "Trabzon", "erzurum": "Erzurum", "van": "Van",
    "avrupa": "Avrupa", "asya": "Asya", "afrika": "Afrika", "amerika": "Amerika",
    "almanya": "Almanya", "alman": "Alman", "fransa": "Fransa", "fransız": "Fransız",
    "ingiltere": "İngiltere", "ingiliz": "İngiliz", "rusya": "Rusya", "rus": "Rus",
    "çin": "Çin", "japonya": "Japonya", "japon": "Japon", "italya": "İtalya",
    "ispanya": "İspanya", "yunanistan": "Yunanistan", "iran": "İran", "irak": "Irak",
    "suriye": "Suriye", "mısır": "Mısır", "hindistan": "Hindistan", "kanada": "Kanada",
    "brezilya": "Brezilya", "avustralya": "Avustralya", "allah": "Allah",
    "akdeniz": "Akdeniz", "karadeniz": "Karadeniz", "ege": "Ege", "anadolu": "Anadolu",
}
# TRUECASE: korpustan öğrenilmiş özel-isim tablosu (_PROPER üstüne biner). Yükleme
# bir kez, uygulama çıkışta. truecase.json yoksa sadece _PROPER ile çalışır.
try:
    from truecase import TrueCaser
    _TRUECASER = TrueCaser(fallback=_PROPER)
except Exception:
    _TRUECASER = None


def _capitalize_tr(s):
    """Çıkış düzeltme (BPE küçük-harf ürettiği için): (1) özel isim (truecase tablosu,
    korpustan öğrenilmiş), (2) cümle başı büyük harf (Türkçe i->İ)."""
    if _TRUECASER is not None:
        s = _TRUECASER.fix(s)                              # önce isimler (küçük girişte)
    out = []
    cap = True
    for ch in s:
        if cap and ch.isalpha():
            ch = "İ" if ch == "i" else ("I" if ch == "ı" else ch.upper())
            cap = False
        elif ch in ".!?":
            cap = True
        out.append(ch)
    return "".join(out)


def _answer(tok, model, cfg, q):
    """Önce tool hook zinciri (calc vb.) dener — tool cevaplarsa model ATLANIR
    (502M aritmetik bilemez, hesap makinesi bilir). Yoksa normal generate."""
    try:
        from tools import run_pre_hooks
        a = run_pre_hooks(q)
        if a is not None:
            return a
    except ImportError:
        pass                                  # tools/ yoksa sohbet aynen çalışır
    return generate(tok, model, cfg, q)


def main():
    tok, model, cfg = load()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Claris hazır ({nparam/1e6:.1f}M param, bağlam {cfg['context']}). "
          f"Çıkmak için 'exit'.\n")
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        a = _answer(tok, model, cfg, q)
        print("Claris:", a)
        log_chat(q, a)
        return
    while True:
        try:
            q = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "kapat"):
            break
        a = _answer(tok, model, cfg, q)
        print("Claris:", a, "\n")
        log_chat(q, a)


if __name__ == "__main__":
    main()
