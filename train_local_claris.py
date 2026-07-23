# -*- coding: utf-8 -*-
"""
Calisra YEREL eğitici — kendi cihazında (tek GPU / CPU) sıfırdan VEYA resume pretrain.

train_claris.py Kaggle'a göre ayarlıdır (2× T4, /kaggle yolları, 11.75h oto-dur).
Bu script onun YEREL, ESNEK (argparse), tek-GPU/CPU dostu sürümüdür. Model mimarisini
ve veri/tokenize fonksiyonlarını train_claris'dan İÇE AKTARIR (mimari tek yerde
kalır; burada yalnız eğitim döngüsü + yerel ayarlar var).

Öne çıkanlar:
  - Tek GPU veya CPU (otomatik algılar; --device ile zorla).
  - GRADIENT ACCUMULATION: düşük-VRAM GPU'da küçük batch ile büyük "efektif batch".
  - Süre/adım limiti, resume (kaldığı yerden), checkpoint.
  - --small: düşük-VRAM/CPU için minik deneme config'i (asıl modelle uyumsuz, ayrı bir şey).

Örnekler:
  python train_local.py                              # data/fetched -> models/, oto-config
  python train_local.py --batch 2 --grad-accum 16    # 8GB GPU (efektif batch 32)
  python train_local.py --small --device cpu --max-steps 200   # CPU'da hızlı deneme
  python train_local.py --max-hours 3                # 3 saat eğit, kaydet, dur
"""

import os
import sys
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn

import bpe as bpemod
import train_claris as T   # MODEL + veri/tokenize fonksiyonları buradan (kopya yok)


def parse_args():
    ap = argparse.ArgumentParser(description="Calisra yerel pretrain (tek GPU/CPU)")
    ap.add_argument("--data", default=os.path.join(T.ROOT, "data", "fetched"),
                    help="jsonl veri klasörü")
    ap.add_argument("--out", default=os.path.join(T.ROOT, "models"),
                    help="çıktı klasörü (bpe.json, claris_model.pt, token cache)")
    ap.add_argument("--batch", type=int, default=4, help="GPU/CPU batch (VRAM'a göre)")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="kaç batch biriktirip 1 adım atılsın (efektif batch = batch×bu)")
    ap.add_argument("--ctx", type=int, default=None, help="bağlam (vars. mimari değeri)")
    ap.add_argument("--lr", type=float, default=T.LR)
    ap.add_argument("--passes", type=int, default=1, help="veri üstünde kaç tam geçiş")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = passes'e göre otomatik")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = sınırsız (süre-oto-dur)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default=None, help="cuda / cpu (vars. otomatik)")
    ap.add_argument("--small", action="store_true",
                    help="düşük-VRAM/CPU için minik deneme config'i (asıl modelle uyumsuz)")
    ap.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2.x hız)")
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument("--force-overwrite", action="store_true",
                    help="config uyumsuz ckpt'in ÜZERİNE yazmaya izin ver (vars. güvenli dur)")
    return ap.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- train_claris global'lerini YEREL ayarlara göre geçersiz kıl ---
    # (veri/tokenize fonksiyonları bu global'leri okur; tek kaynak korunur)
    T.device = device
    T.DATA_DIRS = [a.data]
    T.OUT = a.out
    T.BPE_PATH = os.path.join(a.out, "bpe.json")
    T.CKPT_PATH = os.path.join(a.out, "claris_model.pt")
    T.TOKENS_BIN = os.path.join(a.out, "calisra_tokens.bin")
    T.TOKENS_META = os.path.join(a.out, "calisra_tokens.meta.json")
    T.BATCH_SIZE = a.batch
    if a.ctx:
        T.CONTEXT_LEN = a.ctx
    if a.small:                       # tiny DENEME config (CPU/düşük-VRAM) — AYRI model
        T.N_LAYER, T.N_EMBD, T.N_HEAD, T.CONTEXT_LEN = 6, 384, 6, (a.ctx or 256)
        print("[config] --small: minik deneme modeli (asıl model değil, ayrı bir şey)")
    os.makedirs(a.out, exist_ok=True)
    print(f"🎯 Donanım: {device} | batch {a.batch} × grad-accum {a.grad_accum} "
          f"= efektif {a.batch * a.grad_accum} | ctx {T.CONTEXT_LEN}")

    # --- 1) BPE: varsa yükle, yoksa veriden eğit ---
    tok = bpemod.BPE()
    if os.path.exists(T.BPE_PATH):
        tok.load(T.BPE_PATH)
        print(f"[BPE] yüklendi: {T.BPE_PATH} | sözlük {tok.size}")
    else:
        print("[BPE] sıfırdan eğitiliyor (yerel veriden)...")
        wf = bpemod.collect_word_freq(T.DATA_DIRS, max_lines=1_000_000)
        tok.train(wf, vocab_size=T.VOCAB_SIZE)
        tok.save(T.BPE_PATH)
        print(f"[BPE] sözlük {tok.size} -> {T.BPE_PATH}")

    # --- 2) token cache (memmap, artımlı, shard'lı) — train_claris'ın hattını kullan ---
    shard_paths, ntok = T.prepare_tokens(tok)
    data = T.open_token_data(shard_paths)
    split_idx = int(len(data) * (1 - T.VAL_FRAC))
    print(f"[veri] {len(data):,} token | sözlük {tok.size}")
    if len(data) < T.CONTEXT_LEN * 10:
        print("[HATA] Token çok az. Önce fetch_and_clean.py ile veri çek.")
        return

    # --- 3) model + optim ---
    model = T.Transformer(tok.size).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] katman {T.N_LAYER} d_model {T.N_EMBD} baş {T.N_HEAD} | {nparam/1e6:.1f}M param")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=T.WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    start_step, best_val = 0, float("inf")

    # --- 4) resume (mimari eşleşirse kaldığı yerden) ---
    cur_cfg = {"vocab": tok.size, "context": T.CONTEXT_LEN, "n_embd": T.N_EMBD,
               "n_head": T.N_HEAD, "n_layer": T.N_LAYER, "arch": "llama"}
    if os.path.exists(T.CKPT_PATH):
        try:
            # map_location="cpu": 3GB ckpt dict'i GPU'da ölü yer tutmasın (train_claris
            # ile aynı düzeltme — ilk resume OOM'unun kökü buydu).
            ck = torch.load(T.CKPT_PATH, map_location="cpu")
            cfg = ck.get("config", {})
            mism = {k: (cfg.get(k, "?" if k == "arch" else None), v)
                    for k, v in cur_cfg.items()
                    if cfg.get(k, "?" if k == "arch" else None) != v}
            if not mism:
                model.load_state_dict(ck["model"])
                if "optim" in ck:
                    # optim-state AYRI try (train_claris ile aynı desen): Kaggle
                    # 8-bit AdamW ckpt'i yerel fp32 AdamW'ye uymazsa TÜM resume çökerdi
                    # -> model ağırlığı da giderdi. Şimdi sadece optim sıfırdan başlar.
                    try:
                        opt.load_state_dict(ck["optim"])
                    except Exception as _oe:
                        print(f"[resume] optim-state yüklenemedi (format farklı?) -> optim "
                              f"sıfırdan, model ağırlığı KORUNDU: {str(_oe)[:50]}")
                start_step = int(ck.get("step", 0))
                best_val = float(ck.get("best_val", best_val))
                print(f"[resume] {start_step}. adımdan DEVAM (val {best_val:.3f})")
            else:
                # KLOBBER KORUMASI: config uyumsuz = save() mevcut ckpt'i EZER. --small
                # denemesi yanlışlıkla gerçek claris_model.pt'ye çevrilirse felaket ->
                # kullanıcı --force-overwrite demeden durup uyar.
                if not a.force_overwrite:
                    print(f"[resume] config farklı {mism} ve {T.CKPT_PATH} MEVCUT.\n"
                          f"  Devam etmek mevcut checkpoint'i EZERDİ. Ya --out ile farklı klasör\n"
                          f"  ver, ya da bilerek eziyorsan --force-overwrite ekle.")
                    sys.exit(1)
                print(f"[resume] config farklı {mism} -> SIFIRDAN (--force-overwrite: eski ckpt EZİLECEK)")
        except Exception as e:
            print(f"[resume] yüklenemedi, sıfırdan: {e}")

    if a.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("[hız] torch.compile aktif")

    steps_per_pass = max(1, len(data) // (a.batch * T.CONTEXT_LEN * a.grad_accum))
    total = a.max_steps if a.max_steps > 0 else start_step + steps_per_pass * a.passes
    warmup = min(T.WARMUP, max(10, total // 20))
    print(f"[plan] {start_step} -> {total} adım | ~{steps_per_pass}/geçiş | warmup {warmup}")

    def lr_at(step):
        if step < warmup:
            return a.lr * step / max(1, warmup)
        r = (step - warmup) / max(1, total - warmup)
        return 0.1 * a.lr + 0.5 * (a.lr - 0.1 * a.lr) * (1 + math.cos(math.pi * r))

    def save(step, best):
        torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod") else model
                              ).state_dict(),
                    "optim": opt.state_dict(), "step": step, "best_val": best,
                    "config": cur_cfg}, T.CKPT_PATH)

    # --- 5) eğitim döngüsü (gradient accumulation + süre/adım limiti) ---
    model.train()
    t0 = time.time()
    step = start_step
    for step in range(start_step + 1, total + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):                  # mikro-batch'leri biriktir
            x, y = T.get_batch(data, split_idx, is_val=False)
            loss = model(x, y)                     # forward hedefliyken SADECE skaler loss döner
            loss = (loss.mean() if loss.numel() > 1 else loss) / a.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), T.GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        if step % a.eval_every == 0 or step == total:
            model.eval()
            with torch.no_grad():
                vl = []
                for _ in range(10):
                    xv, yv = T.get_batch(data, split_idx, is_val=True)
                    l = model(xv, yv)
                    vl.append((l.mean() if l.dim() > 0 else l).item())
            val = sum(vl) / len(vl)
            best_val = min(best_val, val)
            model.train()
            print(f" adım {step}/{total}  loss {loss.item()*a.grad_accum:.3f}  "
                  f"val {val:.3f}  {time.time()-t0:.0f}s")
            save(step, best_val)

        if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
            save(step, best_val)
            print(f"⏰ {a.max_hours}h doldu -> kaydedildi, durdu (resume hazır).")
            return

    save(step, best_val)
    print(f"✅ Bitti. adım {step}  val {best_val:.3f} -> {T.CKPT_PATH}")


if __name__ == "__main__":
    main()
