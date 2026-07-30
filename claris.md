# Claris — Teknik Rehber (BitNet b1.58)

Calisra'nın kardeşi: aynı veri + sözlük, ternary ağırlık. Bu belge Claris'in BitNet
farklarını + eğitim mekaniğini anlatır. Hızlı başlangıç → [README.md](README.md).

---

## 1. BitLinear — çekirdek ([bitlinear.py](bitlinear.py))

`nn.Linear`'ın drop-in yerine. Forward'da:
1. **SubLN**: `x = RMSNorm(x)` (kuantizasyondan önce, BitLinear içinde).
2. **Aktivasyon kuant**: per-token absmax → int8 (`act_quant`).
3. **Ağırlık kuant**: absmean → ternary {-1,0,+1} (`weight_quant`).
4. `F.linear(x_q, w_q)`.

**STE** (Straight-Through Estimator): `w_q = w + (quant(w) - w).detach()` → forward'da
kuantize değer, backward'da gradyan DÜZ geçer → fp16 gölge ağırlık güncellenir.

**bias YOK** (binarizasyon regularizasyon gibi davranır).

⚠️ **1.58-bit hızı yalnız ÇIKARIMDA.** Eğitimde fp16 gölge tutulur → Calisra hızına yakın.

---

## 2. Mimari farkları (fp16 Calisra → BitNet Claris)

| | Calisra (fp16) | Claris (BitNet) |
|---|---|---|
| Linear | `nn.Linear` | **BitLinear** (ternary+int8+STE) |
| Norm | input/post-attention RMSNorm (blok) | **SubLN** (BitLinear içinde; ayrı blok-norm YOK) |
| FFN | SwiGLU (SiLU) | **Squared ReLU** `down(relu(gate)²·up)` |
| lm_head | nn.Linear (tied) | BitLinear (tied, SubLN = final norm) |
| Boyut | 16L·1536d·24h (~502M) | 24L·1024d·16h (**~335M**) |
| Aynı | RoPE, MHA head_dim 64, vocab 32k, ctx 2048, weight tying | ← |

Blok: `x = x + attn(x); x = x + mlp(x)` (norm'lar BitLinear'ların SubLN'inde).

---

## 3. Eğitim reçetesi (BitNet)

- **LR 6e-4** (Calisra'nın 2×'i) — ternary ayrık adımlarla hareket eder, büyük güncelleme ister.
- **İki-aşama LR/WD** (opsiyonel, `CLARIS_LR_TOTAL=<adım>`): [0,mid] yüksek-tepe cosine + WD 0.1;
  [mid,T] ANİ düşük tepe + WD 0. Sabit ufuk (resume'da kaymaz). `=0` (vars.) → tek-aşama cosine.
- Gerisi Calisra'dan **aynen** (kanıtlı): DDP 2×GPU, 8-bit AdamW, seçici gradient checkpointing,
  torch.compile, chunked CE, shard'lı memmap, RAM-data, env MAX_HOURS oto-dur, resume+config-kilit
  (`arch:"bitnet"` → Calisra fp16 ckpt'i yanlışlıkla yüklenmez), bounded BPE cache, bf16-guard.

---

## 4. Veri — paylaşımlı + READ-ONLY (kritik güvenlik)

Claris kendi jsonl'ini TUTMAZ; Calisra'nın 33B bin'ini **symlink** ile OKUR.
- **`CLARIS_BIN_RO=1` (varsayılan AÇIK):** Claris paylaşılan bin'e ASLA yazamaz. Cache tam
  çözülmezse HATA verir (symlink'e "wb" yazıp Calisra'nın gerçek bin'ini truncate etmesin —
  bu koruma bir kez öğrenildi). Calisra yeni shard eklerse `models/`'teki symlink'i güncelle.
- Eğitim ortamı: Calisra'nın mevcut `calisra_tokens*` cache'i + bpe.json doğrudan eklenir.
  Ayrı token kaynağı, re-tokenize, çevirme YOK.

---

## 5. Yol haritası

- [x] BitLinear + birim test · [x] BitNet model (~513M, forward/backward + STE doğrulandı)
- [x] Eğitim loop (loss düşüyor) · [x] BIN_RO güvenlik · [x] uzak 2×GPU ilk commit (~8.7k→N=10)
- [ ] ~5-10B token, Calisra fp16 kıyas (aynı-token val)
- [ ] SFT (Calisra hattı, loss SUM reduction)
- [ ] **DPO — hizalama** (SFT sonrası; base→SFT→DPO). RLHF'in küçük-model kararlı hâli,
  tercih çiftleriyle (insan altın-standart + API ölçek). BitNet DPO'ya uyumlu — sadece
  policy güncellemesi, ternary mimari fark etmez. Detay: Calisra `calisra.md` §5.2.
- [ ] **v1.0 GGUF (bitnet.cpp, CPU ~100MB)**

---

## 6. Tek cümle

> **Claris = Calisra'nın verisi/sözlüğüyle, ternary {-1,0,+1} BitNet ~335M Türkçe mini modeli.
> Deploy ~66MB CPU; aynı veri iki teknoloji — öğrenme + kıyas projesi.**
