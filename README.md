# Claris 🇹🇷 — 1.58-bit (BitNet) Türkçe mini model

**Calisra'nın kardeş projesi.** Aynı temiz Türkçe veri + aynı 32k BPE sözlük, ama tamamen
**BitNet b1.58** (ternary ağırlık {-1, 0, +1}) üzerine kurulu ~335M parametre. Amaç: CPU'da
çalışan, ~66MB'lık, hissedilmez footprint'li Türkçe edge sohbet/otomasyon modeli + yeni
teknoloji öğrenme aracı.

> **Calisra vs Claris:** Calisra = fp16 Llama (~502M, kaliteli). Claris = BitNet ternary
> (~335M, çok küçük deploy). Aynı veri, iki teknoloji — yan yana öğrenme + kıyas.

---

## Ne farklı? (fp16 → BitNet, sadece 3 şey)

BitNet b1.58 = Llama mimarisi + 3 değişiklik (bkz. [claris.md](claris.md)):
1. **BitLinear** ([bitlinear.py](bitlinear.py)) — tüm `nn.Linear` yerine. Ağırlık forward'da
   ternary'e (absmean), aktivasyon int8'e (absmax) kuantize; gradyan STE ile fp16 gölgeye akar.
2. **SubLN** — BitLinear'ın İÇİNDE RMSNorm (ayrı input/post-attention layernorm YOK).
3. **Squared ReLU (ReLU²)** FFN — SwiGLU/SiLU yerine.

**1.58-bit hızı SADECE çıkarımda.** Eğitimde fp16 gölge ağırlık tutulur → Calisra hızına yakın.

---

## Mimari

| Bileşen | Değer |
|--------|-------|
| Tip | BitNet b1.58 decoder-only (ternary {-1,0,+1}) |
| Parametre | **~335M** (16L→24L derin, d 1024) |
| Katman / d_model / baş | **24 / 1024 / 16** (head_dim 64) |
| FFN | Squared ReLU gated (down(relu(gate)²·up)), hidden 2688 |
| Norm | SubLN (RMSNorm BitLinear içinde) |
| Pozisyon | RoPE (θ=10000) |
| Sözlük | 32k BPE (Calisra ile AYNI, vocab 32000) |
| Bağlam | 2048 token |
| Deploy | ~66MB (ternary paketli, bitnet.cpp CPU) |
| Eğitim reçetesi | LR 6e-4 (2× fp16), opsiyonel 2-aşama LR/WD |

---

## Veri — SIFIR yeni iş (Calisra'nınki aynen)

**Önemli:** BitNet verisi standart tokenize metin — 1.58-bit olan AĞIRLIKLAR, veri değil.
Calisra'nın temiz 33B token bin'i + bpe.json **doğrudan** kullanılır:
- `models/` içindeki `calisra_tokens*.bin` + `bpe.json` = Calisra'ya **symlink** (66GB kopya YOK).
- Kaggle: Calisra'nın MEVCUT dataset'leri (`calisra-tokens/-001/-002` + `calisra-code`'un
  bpe.json'u) notebook'a **doğrudan eklenir**. Ayrı token dataset'i, re-tokenize, çevirme YOK.

---

## Kullanım

```bash
# yerel deneme (tiny config, CPU/GPU)
CLARIS_TRUST_CACHE=1 python train_local_claris.py --small --device cpu --max-steps 50

# Kaggle 2×T4 DDP (calisra-tokens + claris-code dataset'leri ekli):
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CLARIS_TRUST_CACHE=1 \
CLARIS_MAX_HOURS=6 torchrun --standalone --nproc_per_node=2 train_claris.py

# ilerleme / sohbet
python progress_claris.py
python claris_chat.py "merhaba"
```

**Süre env'den:** `CLARIS_MAX_HOURS=6` (Calisra da eğitimdeyse) / `=11.75` (hesap taze).
Diğer env: `CLARIS_LAYERS/DMODEL/HEADS/BATCH/ACCUM/CKPT_N/LR_TOTAL` — hepsi Calisra deseni.

---

## Kaggle akışı (Calisra ile PAYLAŞIMLI veri)

| Dataset | İçerik | Kaynak |
|---|---|---|
| **claris-code** (YENİ) | bitlinear.py + train_claris.py + bpe.py + bpe.json | `kaggle_push.py code` |
| **claris-resume** (YENİ) | claris_model.pt (ayrı beyin) | `kaggle_push.py model` |
| calisra-tokens / -001 / -002 | 33B bin (PAYLAŞ) | Calisra'nın mevcut dataset'i |

`kaggle_push.py tokens` bilinçli engelli (veri Calisra'dan paylaşılır). AFK upload:
`bash .claris_upload.sh` (ekran uyumaz, poweroff yok).

---

## Yol haritası
- [x] BitLinear (ternary+int8+STE+SubLN) + birim test
- [x] BitNet model (~335M, ReLU² FFN, SubLN) — forward/backward doğrulandı
- [x] Calisra optimizasyonları (DDP/8bit/ckpt/compile/chunked-CE/shard/bounded-cache) taşındı
- [x] Veri symlink (Calisra bin+bpe paylaşımı)
- [ ] Kaggle 2×T4 gerçek pretraining (env-süre commit + resume)
- [ ] ~5-10B token → akıcı Türkçe + Calisra fp16 ile kıyas
- [ ] SFT (Calisra hattı, loss SUM reduction)
- [ ] **v1.0 GGUF export** (bitnet.cpp, CPU'da torch'suz ~66MB)
