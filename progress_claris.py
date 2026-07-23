# -*- coding: utf-8 -*-
"""
Calisra şu an ne durumda diye bakmak için küçük bir araç.
Modeli eğitmeye falan gerek yok; sadece kayıt dosyasını (.pt) açıp
"şu ana kadar kaç token gördü, kaç kere eğitildi, val kaç" gibi şeyleri söylüyor.

Kullanımı:  python progress_claris.py [models/claris_model.pt]
"""
import os
import sys
import torch

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

# Dosya adı verilmezse PRETRAIN ckpt'e bakar (ilerleme sayaçları — token/commit/val —
# sadece onda var; claris_sft.pt yalnız ağırlık+config taşır). SFT'ye bakmak istersen
# yolu elle ver: python progress_claris.py models/claris_sft.pt
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = next((os.path.join(ROOT, "models", n)
                 for n in ("claris_model.pt", "claris_sft.pt")
                 if os.path.exists(os.path.join(ROOT, "models", n))),
                os.path.join(ROOT, "models", "claris_model.pt"))
ck = torch.load(path, map_location="cpu")

# Kayıt dosyasının içindeki bilgileri çekiyoruz (yoksa 0/işaret dönsün)
tok = ck.get("tokens_seen", 0)       # şu ana kadar toplam kaç token eğitildi
com = ck.get("commits", 0)           # kaç kere (commit) eğitim yapıldı
step = ck.get("step", 0)             # kaçıncı adımda kaldı
val = ck.get("best_val", "?")        # en iyi val (düştükçe iyi)
cfg = ck.get("config", {})
sd = ck.get("model", {})
nparam = sum(v.numel() for v in sd.values())
# weight tying: embed_tokens + lm_head AYNI tensör ama state_dict'te İKİ anahtar ->
# çift sayma (551M görünürdü, gerçek ~502M). Tek say (calisra_chat ile tutarlı).
if "embed_tokens.weight" in sd and "lm_head.weight" in sd:
    nparam -= sd["lm_head.weight"].numel()

# Kabaca ne kadar doldu göstergesi. Hedef ~25B token (bkz. calisra.md §8: 25-30B).
TARGET_TOK = 25e9
pct = min(100, tok / TARGET_TOK * 100)

print("=" * 50)
print(f"📊 Claris NE DURUMDA  ({path})")
print("=" * 50)
print(f"  Toplam eğitilen : {tok/1e9:.2f}B token")
print(f"  Kaç commit      : {com}")
print(f"  Kaçıncı adım    : {step:,}")
print(f"  En iyi val      : {val}")
print(f"  Boyut           : {nparam/1e6:.1f}M param  | ctx {cfg.get('context','?')} | arch {cfg.get('arch','?')}")
print(f"  Dolgunluk (~25B): {pct:.1f}%")
print("=" * 50)
