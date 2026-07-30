# -*- coding: utf-8 -*-
"""
Claris DPO (Direct Preference Optimization) — hizalama aşaması. Sıra: base -> SFT -> DPO.
SFT modeli "cevap vermeyi" öğrenir; DPO "hangi cevap daha iyi"yi öğretir (tercih hizalaması),
ayrı ödül modeli gerektirmeden. Model BitNet olsa da DPO değişmez (log-olasılık üzerinden).

Fikir: SFT ckpt'inden İKİ kopya —
  - politika (trainable): eğittiğimiz model.
  - referans (donuk): SFT ağırlıklarının dondurulmuş kopyası (çıpa).
Her tercih çifti (aynı prompt için chosen > rejected) üzerinde:

  loss = -log σ( β · [ (logπ_chosen - logπ_ref_chosen) - (logπ_rejected - logπ_ref_rejected) ] )

Veri: {"prompt", "chosen", "rejected"} satırlı jsonl (DATA_DIRS altında). Log-olasılık
SADECE cevap tokenlarında toplanır (prompt maskeli).

Çıktı: claris_dpo.pt. SFT/pretrain ckpt'lerine dokunmaz.
Veri/çıktı yolları env'den: CLARIS_INPUT_DIRS (: ile ayrık), CLARIS_OUT_DIR.

Not: SFT bittikten SONRA koşulur (tercih verisi + oturmuş bir SFT modeli gerekir).
"""

import os
import sys
import glob
import json
import time
import random

import torch
import torch.nn.functional as F

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
for _p in (ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INPUT_DIRS = [p for p in os.environ.get("CLARIS_INPUT_DIRS", "").split(":") if p.strip()]
for _d in INPUT_DIRS:
    if os.path.isdir(_d):
        for _root, _, _files in os.walk(_d):
            if "bpe.py" in _files and _root not in sys.path:
                sys.path.insert(0, _root)
                break

import bpe as bpemod
import train_claris as T   # BitNet model mimarisi + Transformer

DATA_DIRS = INPUT_DIRS + [os.path.join(ROOT, "data")]
OUT = os.environ.get("CLARIS_OUT_DIR") or os.path.join(ROOT, "models")
DPO_CKPT = os.path.join(OUT, "claris_dpo.pt")
START_NAMES = ("claris_sft.pt", "claris_model.pt")

# AYARLAR
BETA = 0.1                # KL kısıtı sıkılığı. Büyük = referansa daha bağlı.
DPO_LR = 5e-7             # DPO'da LR düşük; büyük LR politikayı hızla bozar.
DPO_EPOCHS = 1
ACCUM = 8
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
MAX_HOURS = 11.0          # uzak commit süre limiti (DPO genelde kısa)
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


def _find(name_or_glob):
    for d in ([OUT] + DATA_DIRS):
        if not os.path.isdir(d):
            continue
        hits = glob.glob(os.path.join(d, "**", name_or_glob), recursive=True)
        if hits:
            return hits[0]
    return None


def _encode_pair(tok, prompt, response, ctx):
    """(ids, resp_mask). Biçim: <bos><user>prompt<bot>response<eos>. Prompt maskeli.
    Kullanıcı girdisi allow_special=False (injection güvenliği); cevap allow_special=True."""
    bos, eos = tok.tok("<bos>"), tok.tok("<eos>")
    u, b = tok.tok("<user>"), tok.tok("<bot>")
    ids = [bos, u]
    ids += tok.encode(prompt, allow_special=False)
    ids.append(b)
    mask = [0] * len(ids)
    resp = tok.encode(response, allow_special=True) + [eos]
    ids += resp
    mask += [1] * len(resp)
    return ids[:ctx], mask[:ctx]


def build_pairs(tok, ctx):
    """DATA_DIRS'teki jsonl'lerden {"prompt","chosen","rejected"} çiftleri."""
    seen, pairs = set(), []
    for d in DATA_DIRS:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if path in seen:
                continue
            seen.add(path)
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    p, c, r = o["prompt"], o["chosen"], o["rejected"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if not (p and c and r) or c == r:
                    continue
                ic, mc = _encode_pair(tok, p, c, ctx)
                ir, mr = _encode_pair(tok, p, r, ctx)
                if sum(mc) < 1 or sum(mr) < 1:
                    continue
                pairs.append((ic, mc, ir, mr))
    return pairs


def seq_logprob(model, ids, mask):
    """Cevap tokenlarının toplam log-olasılığı. hedef = ids[1:], tahmin = logits[:-1]."""
    x = torch.tensor([ids[:-1]], device=device)
    tgt = torch.tensor([ids[1:]], device=device)
    m = torch.tensor([mask[1:]], device=device, dtype=torch.float32)
    logits = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (gathered * m).sum()


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    t_start = time.time()
    print(f"[dpo] donanım: {device}")

    bpe_path = _find("bpe.json") or os.path.join(OUT, "bpe.json")
    if not os.path.exists(bpe_path):
        print("[HATA] bpe.json yok — pretrain/SFT tokenizer'ını ekle.")
        return
    tok = bpemod.BPE().load(bpe_path)
    print(f"[BPE] {bpe_path} | sözlük {tok.size}")

    ckpt_path = None
    for name in START_NAMES:
        ckpt_path = _find(name)
        if ckpt_path:
            break
    if not ckpt_path:
        print("[HATA] başlangıç modeli yok (claris_sft.pt / claris_model.pt). SFT'yi önce koş.")
        return
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    T.N_EMBD, T.N_HEAD, T.N_LAYER, T.CONTEXT_LEN = (
        cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["context"])
    T.DROPOUT = 0.0
    ctx = cfg["context"]

    policy = T.Transformer(cfg["vocab"]).to(device)
    policy.load_state_dict(ck["model"])
    ref = T.Transformer(cfg["vocab"]).to(device)
    ref.load_state_dict(ck["model"])
    ref.eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)
    print(f"[model] başlangıç: {ckpt_path} | {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")

    pairs = build_pairs(tok, ctx)
    if len(pairs) < 20:
        print(f"[HATA] tercih çifti çok az ({len(pairs)}). {{prompt,chosen,rejected}} jsonl ekle.")
        return
    print(f"[dpo] {len(pairs)} tercih çifti")

    steps_total = (len(pairs) * DPO_EPOCHS) // ACCUM
    warmup = max(1, int(steps_total * WARMUP_FRAC))
    opt = torch.optim.AdamW(policy.parameters(), lr=DPO_LR, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < warmup:
            return DPO_LR * step / warmup
        import math
        prog = (step - warmup) / max(1, steps_total - warmup)
        return DPO_LR * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    policy.train()
    step = 0
    acc_loss = 0.0
    opt.zero_grad()
    for epoch in range(DPO_EPOCHS):
        random.shuffle(pairs)
        for i, (ic, mc, ir, mr) in enumerate(pairs):
            lp_c = seq_logprob(policy, ic, mc)
            lp_r = seq_logprob(policy, ir, mr)
            with torch.no_grad():
                rp_c = seq_logprob(ref, ic, mc)
                rp_r = seq_logprob(ref, ir, mr)
            margin = (lp_c - rp_c) - (lp_r - rp_r)
            loss = -F.logsigmoid(BETA * margin) / ACCUM
            loss.backward()
            acc_loss += loss.item()

            if (i + 1) % ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"  adım {step}/{steps_total}  loss {acc_loss:.4f}", flush=True)
                acc_loss = 0.0
                if (time.time() - t_start) > MAX_HOURS * 3600:
                    print("[dpo] süre limiti -> kaydet + dur")
                    break
        else:
            continue
        break

    os.makedirs(OUT, exist_ok=True)
    torch.save({"model": policy.state_dict(), "config": cfg,
                "base": os.path.basename(ckpt_path), "stage": "dpo"}, DPO_CKPT)
    print(f"[dpo] kaydedildi -> {DPO_CKPT}")


if __name__ == "__main__":
    main()
