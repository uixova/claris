# Claris 🇹🇷 — 1.58-bit (BitNet) Türkçe mini model

**Calisra'nın kardeş projesi.** Aynı temiz Türkçe veri + aynı 32k BPE sözlük, ama tamamen
**BitNet b1.58** (ternary ağırlık {-1, 0, +1}) üzerine kurulu ~513M parametre. Amaç: CPU'da
çalışan, ~66MB'lık, hissedilmez footprint'li Türkçe edge sohbet/otomasyon modeli + yeni
teknoloji öğrenme aracı.

> **Calisra vs Claris:** Calisra = fp16 Llama (~502M, kaliteli). Claris = BitNet ternary
> (~513M, küçük deploy). Aynı veri, iki teknoloji — yan yana öğrenme + kıyas.

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
| Parametre | **517M** (ölçüldü, L24 d1280) |
| Katman / d_model / baş | **24 / 1280 / 20** (head_dim 64) |
| FFN | Squared ReLU gated (down(relu(gate)²·up)), hidden 2688 |
| Norm | SubLN (RMSNorm BitLinear içinde) |
| Pozisyon | RoPE (θ=10000) |
| Sözlük | 32k BPE (Calisra ile AYNI, vocab 32000) |
| Bağlam | 2048 token |
| Deploy | ~66MB (ternary paketli, bitnet.cpp CPU) |
| Eğitim reçetesi | LR 5.5e-4 (2× fp16), opsiyonel 2-aşama LR/WD |
| Kuantizasyon kapsamı | **169 BitLinear** — q/k/v/o, gate/up/down, lm_head. Kuantize edilmemiş `nn.Linear` **yok** (embedding hariç, o zaten lookup) |

---

## Veri — SIFIR yeni iş (Calisra'nınki aynen)

**Önemli:** BitNet verisi standart tokenize metin — 1.58-bit olan AĞIRLIKLAR, veri değil.
Calisra'nın temiz 33B token bin'i + bpe.json **doğrudan** kullanılır:
- `models/` içindeki `calisra_tokens*.bin` + `bpe.json` = Calisra'ya **symlink** (66GB kopya YOK).
- Kaggle: Calisra'nın MEVCUT dataset'leri (`calisra-tokens/-001/-002/-003` + `calisra-code`'un
  bpe.json'u) notebook'a **doğrudan eklenir**. Ayrı token dataset'i, re-tokenize, çevirme YOK.

> ### ⚠️ `CLARIS_BIN_RO=1` (varsayılan AÇIK) — kapatma
> Paylaşılan bin bir **symlink**. Claris onu sadece okur; cache uyumsuz görünürse yeniden
> tokenize etmeye kalkmaz, `RuntimeError` ile durur. Bu guard bedavaya gelmedi: bir kez
> symlink üzerinden yazıldı ve Calisra'nın 17.18GB shard 0'ı 0 byte'a düştü (jsonl'den tam
> rebuild gerekti).

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
| calisra-tokens / -001 / -002 / -003 | 33.3B bin, 4 shard (PAYLAŞ) | Calisra'nın mevcut dataset'i |

`kaggle_push.py tokens` bilinçli engelli (veri Calisra'dan paylaşılır). AFK upload:
`bash .claris_upload.sh` (ekran uyumaz, poweroff yok).

---

## Yol haritası
- [x] BitLinear (ternary+int8+STE+SubLN) + birim test
- [x] BitNet model (~513M, ReLU² FFN, SubLN) — forward/backward doğrulandı
- [x] Calisra optimizasyonları (DDP/8bit/ckpt/compile/chunked-CE/shard/bounded-cache) taşındı
- [x] Veri symlink (Calisra bin+bpe paylaşımı)
- [ ] Kaggle 2×T4 gerçek pretraining (env-süre commit + resume)
- [ ] ~5-10B token → akıcı Türkçe + Calisra fp16 ile kıyas
- [ ] SFT (Calisra hattı, loss SUM reduction)
- [ ] **v1.0 GGUF export** (bitnet.cpp, CPU'da torch'suz ~66MB)

---

## Lisans

**Apache-2.0** (bkz. [LICENSE](LICENSE)). Ağırlıklar bu repoda değil — kod GitHub'da,
ağırlık HuggingFace'te (gerekçe: Calisra'nın [YAYIN.md](https://github.com/uixova/calisra/blob/main/YAYIN.md)).

## Kaynaklar
- BitNet b1.58 2B4T teknik rapor — https://arxiv.org/abs/2504.12285
- Saf PyTorch BitLinear referansı — https://github.com/kevbuh/bitnet
- Küçük ağlarda ternary paritesi — https://arxiv.org/pdf/2407.09527
- bitnet.cpp (CPU çıkarım) — https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
