# -*- coding: utf-8 -*-
"""
Claris eğitici scripti — BitNet b1.58 (ternary {-1,0,+1} ağırlık), decoder-only,
Llama/Qwen tarzı bir Transformer: RMSNorm(SubLN) + RoPE (dönel pozisyon) + Squared-ReLU
(ReLU²) gated FFN. ~335M parametre, vocab 32000, sadece METİN (Türkçe).

Calisra'dan (fp16, 502M) tek farkı MİMARİ: her nn.Linear -> BitLinear (ternary ağırlık +
int8 aktivasyon, STE ile eğitim). VERİ AYNI — Calisra'nın 32k BPE'si ve token bin'i
symlink ile PAYLAŞILIYOR (re-tokenize YOK, kopya YOK). 1.58-bit kazancı ÇIKARIMDA:
CPU'da ~66MB'lık paketlenmiş ternary model.

Akış:
  1) BPE: Calisra'nınki aynen kullanılır (models/bpe.json -> symlink, vocab 32000).
     ASLA yeniden eğitme — vocab kayarsa tüm bin'ler + resume ölür.
  2) Token akışı: calisra_tokens*.bin symlink'i SADECE OKUNUR (CLARIS_BIN_RO=1
     varsayılan; yazma girişimi RuntimeError ile durur, Calisra'nın verisi korunur).
  3) Transformer'ı next-token tahminiyle eğit (context = CONTEXT_LEN token)
  4) models/claris_model.pt'yi kaydet (+ claris_chat.py ile çalıştırırsın)

Veri/çıktı yolları env'den: CLARIS_INPUT_DIRS (: ile ayrık), CLARIS_OUT_DIR.
"""

import os
# torch'tan ÖNCE set edilmeli: expandable_segments ayırıcıyı büyütülebilir yapar,
# parçalanma kaynaklı OOM'u düşürür.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import json
import glob
import math
import time
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint                       # gradient checkpointing (VRAM tasarrufu)
from torch.nn import functional as F

# Ampere+ (TF32) GPU'da bedava matmul hızı; eski GPU'da no-op. Şekil sabit olduğu için
# cudnn.benchmark ilk adımda en hızlı çekirdeği seçip kilitler.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
# torch._dynamo modül düzeyinde import edilmeli (main içinde import 'torch'u local yapar
# -> manual_seed UnboundLocalError). compile çeviremezse sessizce eager'a düşer.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

# hücreye yapıştırınca __file__ tanımsız -> getcwd
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
for _p in (ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Girdi dizinleri env'den (: ile ayrık); yerelde varsayılan data/. Uzak eğitim ortamı
# bu env'i kendi veri mount yoluna set eder.
INPUT_DIRS = [p for p in os.environ.get("CLARIS_INPUT_DIRS", "").split(":") if p.strip()]

# bpe.py girdi dizinlerinden birine eklenmişse otomatik bul
for _d in INPUT_DIRS:
    if os.path.isdir(_d):
        for _root, _, _files in os.walk(_d):
            if "bpe.py" in _files and _root not in sys.path:
                sys.path.insert(0, _root)
                break

import bpe as bpemod
from bitlinear import BitLinear   # BitNet çekirdeği: ternary ağırlık + int8 akt + STE + SubLN

# VERİ VE ÇIKTI YOLLARI
DATA_DIRS = INPUT_DIRS + [os.path.join(ROOT, "data")]
OUT = os.environ.get("CLARIS_OUT_DIR") or os.path.join(ROOT, "models")
BPE_PATH = os.path.join(OUT, "bpe.json")
# BEYİN AYRI, VERİ ORTAK:
#   - ckpt adı "claris_model.pt" -> Calisra'nın calisra_model.pt'siyle ASLA karışmaz,
#     üstüne yazmaz, yanlışlıkla resume etmez (mimariler farklı: bitnet vs llama).
#   - token bin adı "calisra_tokens" -> BİLEREK aynı: Calisra'nın 33B token bin'ini
#     symlink üzerinden PAYLAŞIYOR. Vocab ikisinde de 32000 olduğu için bin doğrudan
#     okunabilir (bin MODEL BOYUTUNA değil VOCAB'a kilitli). Re-tokenize/kopya YOK.
#   - Yazmaya karşı koruma: CLARIS_BIN_RO varsayılan AÇIK (aşağı bak) -> Claris bu bin'e
#     yazmaya kalkarsa RuntimeError. (Bir kez symlink üzerinden yazıp Calisra'nın 17GB
#     shard'ını sıfırladık; guard o yüzden var, KAPATMA.)
CKPT_PATH = os.path.join(OUT, "claris_model.pt")
CKPT_NAMES = ("claris_model.pt",)                   # resume ederken aranacak dosya
TOKENS_BIN = os.path.join(OUT, "calisra_tokens.bin")
TOKENS_META = os.path.join(OUT, "calisra_tokens.meta.json")
TOKEN_STEMS = ("calisra_tokens",)                   # token cache dosyasının adı (PAYLAŞILAN)
# TOKEN CACHE SHARD'LARI: tek dataset boyut sınırı yüzünden bin tek dosyada büyüyemez.
# Cache SHARD_CAP'e ulaşınca yeni parça açılır (calisra_tokens_001.bin, _002...). Eğitim
# tüm shard'ları tek sanal akış gibi memmap'ler; örnekleme havuzun tamamında rastgele
# (sırayla-shard-işleme yok -> order bias yok). Meta shard basename'lerini tutar -> yollar
# değişse de bulunur.
SHARD_CAP_BYTES = max(2, int(float(os.environ.get("CLARIS_SHARD_GB", "16")) * 1024**3)) & ~1


def _shard_name(i):
    return "calisra_tokens.bin" if i == 0 else f"calisra_tokens_{i:03d}.bin"

# MODEL MİMARİSİ (~513M param, BitNet b1.58, sadece METİN)
# 12*L*d^2 + vocab*d: 12*24*1280^2 + 32000*1280 = 472M + 41M = ~513M. head_dim = 64.
# NEDEN 513M (335M'den yükseltildi): 335M @ CKPT_N=0 8.7k tok/s ile beklenenin ÜSTÜNDE
# gitti -> bu hız fazlasını "zekaya" çevirdik. 335M BitNet ~= 250M-fp16 etkin; 513M BitNet
# ~= 370M-fp16 etkin (+120M etkin kapasite). Deploy farkı küçük (66MB->100MB, ikisi de
# "hissedilmez"), zeka farkı hissedilir. Hız ~6k'ya düşer (Calisra seviyesi, kabul edildi).
VOCAB_SIZE = 32000        # Calisra ile AYNI (paylaşılan bpe.json). ASLA DEĞİŞTİRME:
                          # vocab kayarsa paylaşılan bin okunamaz + resume ölür.
                          # 32000 < 65536 -> token cache uint16 kalır.
CONTEXT_LEN = 2048        # Calisra ile aynı. RoPE olduğu için ileride fine-tune ile uzatılır.
N_EMBD = 1280             # d_model. 335M'deki 1024'ten genişletildi (DERİNLİK sabit tutuldu ->
                          # reçete birebir transfer). head_dim = 1280/20 = 64 (standart).
N_HEAD = 20               # head_dim = 1280/20 = 64 (Calisra/335M ile aynı head_dim)
N_LAYER = 24              # DERİN çizgi (335M de L24'tü). Derinlik BitNet'in en çok kaybettiği
                          # ÇOK-ADIMLI MANTIĞI telafi eder; büyüme sadece GENİŞLİKTEN geldi.
DROPOUT = 0.1
# Toplam ~513M param. Mimari SABİT, resume aynı şekli yükler (config uyuşmazsa ckpt sessizce
# YOK SAYILIR ve SIFIRDAN başlar). DİKKAT: eski 335M claris_model.pt varsa (d1024) config
# uyuşmaz -> otomatik YOK SAYILIR, 513M SIFIRDAN başlar. Bu BEKLENEN (335M terk edildi).
# TOKEN BÜTÇESİ: 513M x ~20 (Chinchilla) = ~10B ama TAVAN DEĞİL -- SmolLM 135M'i 600B'yle
# eğitti (~4400 tok/param). Havuz 33B; 50B overtrain = 100 tok/param (sağlıklı). Tekrar ~4x.
# DÜRÜST NOT: BitNet küçük ölçekte ternary'den fp16'ya göre daha çok kaybeder (parite 3B+'da
# güçlü) -> 335M BitNet ≈ ~250M-fp16 etkin kalite. Aynı-token Calisra kıyası bunu gösterecek.

# EĞİTİM AYARLARI
# Toplam batch = PER_GPU_BATCH × GPU sayısı (device belli olunca hesaplanır). Ayarlar env'den,
# kodu elle düzenlemeye gerek yok. Bol VRAM'de: CLARIS_CKPT=0 CLARIS_BATCH=16 CLARIS_ACCUM=3.
# Not: BitLinear quant ara-tensörleri (act/weight_quant + STE) ekstra aktivasyon VRAM'i yer.
PER_GPU_BATCH = int(os.environ.get("CLARIS_BATCH", "3"))   # dar VRAM: 3 · bol VRAM: 16+
GRAD_ACCUM = int(os.environ.get("CLARIS_ACCUM", "5"))      # efektif batch = BATCH × N_GPU × ACCUM
GRAD_CKPT = os.environ.get("CLARIS_CKPT", "1") != "0"      # aktivasyon checkpoint (dar VRAM'de aç)
USE_8BIT = os.environ.get("CLARIS_8BIT", "1") != "0"       # 8-bit AdamW (VRAM -%75). 1=aç, 0=standart
# SEÇİCİ CHECKPOINT: kaç bloğu yeniden-hesapla (recompute). Az = daha hızlı ama daha çok VRAM.
# Sadece çalışma-zamanı, ağırlık değişmez -> resume-güvenli. Varsayılan 12 (son bloklar serbest).
GRAD_CKPT_N = int(os.environ.get("CLARIS_CKPT_N", "12"))
# HIZ BAYRAKLARI:
#   CLARIS_COMPILE=1 -> torch.compile füzyon. CLARIS_PARALLEL_TOK=1 -> çok-süreçli tokenize.
#   CLARIS_RAM_DATA=auto -> token .bin RAM'e. CLARIS_CKPT_N -> seçici recompute (yukarı bak).
MAX_ITERS = 0             # 0 = OTOMATİK: adım sayısı VERİ MİKTARINA göre (PASSES geçiş)
PASSES = 2                # veride kaç tam geçiş (epoch). Paylaşılan havuz 33B token / 335M param
                          # = ~98 tok/param -> tek geçiş bile MAX_HOURS'u kat kat aşıyor; süreyi
                          # pratikte CLARIS_MAX_HOURS kapatıyor, PASSES sadece emniyet tavanı.
DIALOG_REPEAT = 1         # 1 = tekrar YOK (oversample kapalı). Doğal+çeşitli bulk
                          # veriyle gerçek dil/mantık öğrenilir; tekrar = ezber/çöp.
EVAL_EVERY = 500
PROGRESS_EVERY = 20       # her bu kadar adımda HAFİF ilerleme (loss+tok/s) bas -> körlük yok.
                          # eval (pahalı) hâlâ EVAL_EVERY'de; bu sadece nabız + throughput ölçümü.
# SÜRE-TABANLI OTO-DUR: süre limitine ÇARPMADAN biraz önce durup normal kaydeder (timeout =
# "failed" = çıktı gider). Kirayı sen belirlersin -> env ile uzat (ör. 24h'de 23.5).
MAX_HOURS = float(os.environ.get("CLARIS_MAX_HOURS", "11.75"))
# 335M'de 6e-4 idi (ternary yüksek LR sever). 513M biraz daha büyük -> 5.5e-4: büyük
# model biraz düşük tepe LR ile daha kararlı yakınsar. Hâlâ Calisra fp16'nın (3e-4)
# ~1.8 katı (BitNet reçetesi yüksek LR ister). Env ile override edilebilir değil (sabit,
# resume-tutarlı); iki-aşama LR mantığı aşağıda LR_TOTAL ile.
LR = 5.5e-4
WARMUP = 1500             # ilk 1500 adımda LR yavaş yavaş artsın, model sağlam otursun diye
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_FRAC = 0.02
RESUME = True             # models/claris_model.pt VARSA sıfırdan değil ÜSTÜNE devam et
SEED = 42

# DDP mi DataParallel mı
# DDP'de GPU'lar eşit+bağımsız çalışır (GPU0 darboğazı yok, ~2x hız). torchrun ile açılır:
#   torchrun --nproc_per_node=2 train_claris.py   (WORLD_SIZE yoksa tek süreç = DataParallel)
# Not: kod bir hücreye yapıştırıldıysa torchrun çalışmaz (diskte .py yok) -> önce dosyaya yaz.
# Çok-GPU ortamında NCCL bazen sessiz takılır -> şu env'lerle başlat:
#   NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 torchrun --standalone --nproc_per_node=2 ...
DDP_RANK = int(os.environ.get("RANK", "-1"))
DDP_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DDP_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
USE_DDP = DDP_WORLD > 1 and torch.cuda.is_available()
IS_MAIN = (DDP_RANK <= 0)                        # rank0 (ya da DDP yok) = log + kayıt yetkilisi

if USE_DDP:
    device = "cuda:%d" % DDP_LOCAL_RANK          # her süreç KENDİ GPU'su
    N_GPU = DDP_WORLD                            # toplam süreç (her biri 1 GPU)
    BATCH_SIZE = PER_GPU_BATCH                   # her süreç kendi GPU'sunu besler (×GPU YOK)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_GPU = torch.cuda.device_count() if device == "cuda" else 1
    BATCH_SIZE = PER_GPU_BATCH * max(1, N_GPU)   # DP: tek süreç tüm GPU'ları besler
IS_CUDA = str(device).startswith("cuda")         # "cuda" veya "cuda:1" -> True

# AMP DTYPE: bf16 (sm80+: Ampere/Ada/Hopper) sayısal olarak fp16'dan güvenli — geniş exponent
# aralığı -> overflow yok -> GradScaler gerekmez. Karar compute capability >= 8.0 ile verilir,
# is_bf16_supported() ile DEĞİL: yeni PyTorch eski GPU'da da True dönebiliyor (emülasyon), ama
# sm<8.0'da bf16 = tensor-core yok + mem-efficient SDPA bf16 desteklemez -> attention matrisi
# ham açılır = OOM. sm80+ şartı bunu kapatır. Ağırlık fp32 master kalır -> dtype değişse de
# resume güvenli (sm<8.0'da fp16 eğit, sonra sm80+'da bf16'yla devam = sorunsuz).
# Elle: CLARIS_BF16=0 (kapat) / 1 (sm80+ ise aç).
try:
    _cap_ok = (IS_CUDA and torch.cuda.is_available()
               and torch.cuda.get_device_properties(
                   DDP_LOCAL_RANK if USE_DDP else 0).major >= 8)
    _bf_env = os.environ.get("CLARIS_BF16", "1") != "0"
    USE_BF16 = bool(_cap_ok and _bf_env)
except Exception:
    USE_BF16 = False
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# GPU-FARKINDALIK: env verilmediyse VRAM'e göre mantıklı varsayılanı seç.
# 16GB: 3/5/ckpt12. 22GB+: batch 16, accum 3, checkpoint kapalı. 40GB+: batch 24, accum 2.
# Elle CLARIS_BATCH/ACCUM/CKPT_N verildiyse dokunmaz.
if IS_CUDA and torch.cuda.is_available():
    try:
        _props = torch.cuda.get_device_properties(DDP_LOCAL_RANK if USE_DDP else 0)
        _gb = _props.total_memory / 1e9
        if "CLARIS_BATCH" not in os.environ:
            PER_GPU_BATCH = 24 if _gb >= 40 else (16 if _gb >= 22 else PER_GPU_BATCH)
        if "CLARIS_ACCUM" not in os.environ:
            GRAD_ACCUM = 2 if _gb >= 40 else (3 if _gb >= 22 else GRAD_ACCUM)
        if "CLARIS_CKPT_N" not in os.environ and _gb >= 22:
            GRAD_CKPT_N = 0                      # bol VRAM -> recompute tamamen kapalı
        if _props.major < 7 and IS_MAIN:
            # tensor core yok (sm<7.0) -> fp16 matmul çok yavaş; daha yeni bir GPU tercih et.
            print(f"[GPU] {_props.name} (sm{_props.major}{_props.minor}): tensor core yok -> yavaş.")
        # BATCH_SIZE yukarıda hesaplandı -> otomatik ayar sonrası tazele
        BATCH_SIZE = PER_GPU_BATCH if USE_DDP else PER_GPU_BATCH * max(1, N_GPU)
    except Exception:
        pass


# MODEL — Llama/Qwen tarzı blok, üç ana parçadan oluşuyor:
#   RMSNorm : LayerNorm gibi ama ortalama almıyor, sadece bölüyor -> biraz daha hızlı, donanım dostu
#   RoPE    : pozisyonu dönel şekilde veriyor, ayrı pozisyon parametresi yok -> bağlamı sonradan uzatabiliyoruz
#   SwiGLU  : normal MLP yerine kapılı MLP -> aynı parametreyle daha iyi öğreniyor
#   Bias yok (Llama tarzı).
RMS_EPS = 1e-6
ROPE_BASE = 10000.0


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (ortalama çıkarmaz; sadece RMS'e böler). fp32'de
    normalize -> fp16 autocast altında sayısal kararlılık."""
    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.to(dt)) * self.weight


def _build_rope(seqlen, head_dim, base=ROPE_BASE):
    """RoPE için cos/sin tablosu: (seqlen, head_dim). head_dim ÇİFT olmalı."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seqlen).float()
    freqs = torch.outer(t, inv_freq)               # (seq, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (seq, hd)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # q,k: (B, nh, T, hd) ; cos,sin: (T, hd) -> broadcast (1,1,T,hd)
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    """HuggingFace LlamaAttention ile BİREBİR: ayrı q_proj/k_proj/v_proj/o_proj,
    bias yok, RoPE (rotate_half), causal SDPA. Ayrı projeksiyonlar (fused qkv değil)
    -> state_dict anahtarları HF LlamaForCausalLM ile 1:1 eşleşir, GGUF dönüşümü
    (convert_hf_to_gguf.py) SIFIR ek işlemle çalışır. torch.compile 3 matmul'ü yatay
    füzyonlar (fused qkv'ye yakın hız). MHA: num_key_value_heads == num_attention_heads."""
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        # BitLinear: SubLN (RMSNorm) İÇERİDE -> ayrı input_layernorm YOK. q/k/v aynı x'i
        # kendi norm'larıyla ayrı normalize eder (SubLN felsefesi, BitNet standardı).
        self.q_proj = BitLinear(N_EMBD, N_EMBD, bias=False)
        self.k_proj = BitLinear(N_EMBD, N_EMBD, bias=False)
        self.v_proj = BitLinear(N_EMBD, N_EMBD, bias=False)
        self.o_proj = BitLinear(N_EMBD, N_EMBD, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        self.nh = N_HEAD
        self.hd = N_EMBD // N_HEAD
        # RoPE cos/sin: state_dict'e YAZILMAZ (persistent=False) -> HF ile 1:1 (HF de
        # RoPE'yi runtime hesaplar). CONTEXT büyürse yeniden kurulur (RoPE esnekliği).
        cos, sin = _build_rope(CONTEXT_LEN, self.hd)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _qkv(self, x, B, T):
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        return q, k, v

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self._qkv(x, B, T)
        # RoPE: T context'i aşarsa (ileride uzatma) tabloyu o uzunlukta yeniden kur
        if T > self.rope_cos.shape[0]:
            cos, sin = _build_rope(T, self.hd, ROPE_BASE)
            cos, sin = cos.to(x.device), sin.to(x.device)
        else:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q, k = _apply_rope(q, k, cos, sin)
        # nedensel (causal) öz-dikkat — geçmişe bakar
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=DROPOUT if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.o_proj(y))

    def forward_cached(self, x, cache_k, cache_v, pos):
        """KV-CACHE'Lİ çıkarım yolu (eğitim forward'ına DOKUNMAZ): yeni T token'ın
        k/v'si cache'in [pos:pos+T] dilimine yazılır, q SADECE yeni tokenlar için
        hesaplanır -> decode adımı O(T_yeni) olur (tam-bağlam forward değil).
        İki mod: prefill (pos=0, T=n, is_causal=True) ve tek-token decode (T=1,
        tüm geçmişe bakar, is_causal=False). Karışık mod (pos>0 ve T>1) SDPA'nın
        is_causal hizasıyla uyumsuz -> assert ile yasak."""
        B, T, C = x.shape
        assert pos == 0 or T == 1, "prefill (pos=0) veya tek-token decode (T=1)"
        q, k, v = self._qkv(x, B, T)
        if pos + T > self.rope_cos.shape[0]:       # bağlam uzatıldıysa tabloyu büyüt
            cos, sin = _build_rope(pos + T, self.hd, ROPE_BASE)
            cos, sin = cos.to(x.device), sin.to(x.device)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        q, k = _apply_rope(q, k, cos[pos:pos + T], sin[pos:pos + T])
        cache_k[:, :, pos:pos + T] = k
        cache_v[:, :, pos:pos + T] = v
        y = F.scaled_dot_product_attention(q, cache_k[:, :, :pos + T],
                                           cache_v[:, :, :pos + T],
                                           is_causal=(pos == 0))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)                      # çıkarım: dropout yok


class SwiGLU(nn.Module):
    """BitNet FFN: SwiGLU'nun SiLU'su yerine SQUARED ReLU (ReLU²) — down( relu(gate)²·up ).
    BitNet b1.58 2B4T reçetesi: kuantize rejimde ReLU² daha stabil. gate/up/down = BitLinear
    (SubLN içeride). İsimler (gate_proj/up_proj/down_proj) HF standardında -> GGUF köprüsü temiz."""
    def __init__(self):
        super().__init__()
        hidden = int(8 / 3 * N_EMBD)
        hidden = 64 * ((hidden + 63) // 64)        # 64'ün katına yuvarla (GPU dostu)
        self.gate_proj = BitLinear(N_EMBD, hidden, bias=False)
        self.up_proj = BitLinear(N_EMBD, hidden, bias=False)
        self.down_proj = BitLinear(hidden, N_EMBD, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        g = F.relu(self.gate_proj(x))
        return self.drop(self.down_proj((g * g) * self.up_proj(x)))   # ReLU² gated


class Block(nn.Module):
    """BitNet decoder bloğu: SubLN BitLinear'ın İÇİNDE olduğu için ayrı input_layernorm /
    post_attention_layernorm YOK. Blok sadece residual + alt-katman:
    x = x + attn(x) ; x = x + mlp(x). Norm'lama her BitLinear'ın kendi SubLN'inde yapılır."""
    def __init__(self):
        super().__init__()
        self.self_attn = CausalSelfAttention()
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.self_attn(x)          # SubLN q/k/v/o_proj içinde
        x = x + self.mlp(x)                # SubLN gate/up/down içinde
        return x

    def forward_cached(self, x, cache_k, cache_v, pos):
        x = x + self.self_attn.forward_cached(x, cache_k, cache_v, pos)
        x = x + self.mlp(x)
        return x


def _chunked_ce(logits, targets, chunk=256):
    """Parçalı cross-entropy: (B,T,V) logit'i T boyunca chunk'lara bölerek hesaplar
    (tek seferde view(-1,V) cross_entropy içinde dev anlık ara buffer açıyor -> OOM;
    parçalı hesap pik VRAM'i %60-70 düşürür). reduction="sum" + geçerli hedef SAYISINA
    bölme -> tam CE ile matematiksel BİREBİR. Eski stack().mean() chunk'ları eşit
    ağırlıklıyordu: maske (ignore_index=-1) eşitsiz dağılırsa loss kayıyordu ve
    TAMAMEN maskeli chunk'ta (SFT padding / uzun user turu) cross_entropy NaN
    dönüyordu -> sum tabanlı hesap ikisini de çözer. Akümülatör fp32 (fp16 autocast
    altında sum taşmasın); clamp(min=1) tümü-maskeli batch'te 0/0'ı engeller
    (loss 0, grad 0 -> GradScaler etkilenmez). Statik Python döngüsü + branch'siz
    clamp -> torch.compile dostu."""
    B, T, V = logits.shape
    total = logits.new_zeros((), dtype=torch.float32)
    count = torch.zeros((), dtype=torch.long, device=logits.device)
    for i in range(0, T, chunk):
        lc = logits[:, i:i + chunk, :].reshape(-1, V)
        tc = targets[:, i:i + chunk].reshape(-1)
        total = total + F.cross_entropy(lc, tc, ignore_index=-1, reduction="sum")
        count = count + (tc != -1).sum()
    return total / count.clamp(min=1).to(total.dtype)


class Transformer(nn.Module):
    """HF LlamaForCausalLM ile 1:1 state_dict: embed_tokens / layers.{i}.* / norm /
    lm_head (weight tying). export_hf.py sadece anahtarlara 'model.' ön eki ekler
    (lm_head hariç) -> ağırlık dönüşümü/yeniden şekillendirme YOK -> GGUF temiz."""
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.embed_tokens = nn.Embedding(vocab, N_EMBD)
        # pos_emb YOK -> pozisyon RoPE ile attention içinde verilir (parametre tasarrufu
        # + bağlam esnekliği). drop embedding üstünde kalır.
        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYER)])
        # lm_head = BitLinear -> SubLN İÇİNDE = final norm (ayrı self.norm YOK).
        self.lm_head = BitLinear(N_EMBD, vocab, bias=False)
        self.embed_tokens.weight = self.lm_head.weight   # weight tying (paylaşılan fp16 gölge)
        self.apply(self._init)
        # RESIDUAL-YOLU ÖLÇEKLİ INIT (GPT-2/Llama pratiği): residual'a eklenen son
        # projeksiyonlar (self_attn.o_proj + mlp.down_proj) 1/sqrt(2L) küçültülür ->
        # derinlikle aktivasyon varyansı patlamaz, ilk adımlar daha kararlı.
        for blk in self.layers:
            nn.init.normal_(blk.self_attn.o_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * len(self.layers)))
            nn.init.normal_(blk.mlp.down_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * len(self.layers)))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # autocast forward İÇİNDE -> DataParallel'in her GPU thread'inde fp16 çalışır
        # (autocast thread-local; dış context DP worker'larına yayılmaz).
        with torch.autocast(device_type=idx.device.type, dtype=AMP_DTYPE,
                            enabled=(idx.device.type == "cuda")):
            x = self.drop(self.embed_tokens(idx))  # RoPE pozisyonu attention'da verir
            for i, blk in enumerate(self.layers):
                # SEÇİCİ GRADIENT CHECKPOINTING: ilk GRAD_CKPT_N bloğu checkpoint'le (backward'da
                # recompute -> VRAM tasarrufu), kalanı NORMAL çalıştır (recompute YOK -> hız).
                # GRAD_CKPT_N=N_LAYER -> hepsi (mevcut). Düşür -> daha hızlı + daha çok VRAM.
                # use_reentrant=False = modern/DP-uyumlu, torch.compile ile uyumlu.
                if GRAD_CKPT and self.training and i < GRAD_CKPT_N:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            logits = self.lm_head(x)       # BitLinear SubLN final norm'u yapar (ayrı norm yok)
            loss = None
            if targets is not None:
                # CHUNKED CROSS-ENTROPY (bkz. _chunked_ce): parçalı hesap -> pik VRAM
                # %60-70 düşer; sum/count tabanlı -> tam CE ile BİREBİR ve maskeli
                # (ignore_index=-1) SFT batch'lerinde NaN-güvenli.
                loss = _chunked_ce(logits, targets)
        # DataParallel DENGE FİX: eğitimde SADECE skaler loss döndür. logits döndürürsek DP
        # onu GPU0'a TOPLAR (batch×seq×vocab ~yüz MB + grad grafı) -> GPU0 şişer, GPU1 boş,
        # biri OOM olur (gördüğün 9.5GB vs 4.4GB). Loss-only -> DP sadece skaleri toplar.
        if loss is not None:
            return loss                    # EĞİTİM/SFT: skaler loss (DP-dengeli)
        return logits                      # ÇIKARIM (generate): logits

    @torch.no_grad()
    def forward_infer(self, idx, caches=None, pos=0):
        """KV-CACHE'Lİ çıkarım: caches=None ise tahsis edip prompt'u prefill'ler;
        sonra her çağrıda yeni token(lar)ı işler. (son_pozisyon_logits(B,V), caches)
        döner. Cache'ler LOCAL tensör (buffer/module DEĞİL) -> state_dict'e asla
        girmez, resume etkilenmez. dtype: cuda'da fp16 (autocast ile aynı), cpu fp32.
        Eğitim forward'ı bayt-bayt aynen durur (compile grafiği değişmez)."""
        B, T = idx.shape
        assert pos + T <= CONTEXT_LEN, "KV cache dolu: bağlam aşıldı"
        if caches is None:
            nh, hd = self.layers[0].self_attn.nh, self.layers[0].self_attn.hd
            dt = AMP_DTYPE if idx.device.type == "cuda" else torch.float32
            caches = [(torch.zeros(B, nh, CONTEXT_LEN, hd, dtype=dt, device=idx.device),
                       torch.zeros(B, nh, CONTEXT_LEN, hd, dtype=dt, device=idx.device))
                      for _ in self.layers]
        with torch.autocast(device_type=idx.device.type, dtype=AMP_DTYPE,
                            enabled=(idx.device.type == "cuda")):
            x = self.embed_tokens(idx)             # çıkarım: dropout yok
            for blk, (ck, cv) in zip(self.layers, caches):
                x = blk.forward_cached(x, ck, cv, pos)
            logits = self.lm_head(x[:, -1, :])     # BitLinear SubLN final norm + son pozisyon (B, V)
        return logits.float(), caches


# VERİYİ TOKEN AKIŞINA ÇEVİRME
def _jsonl_paths():
    seen, out = set(), []
    for d in DATA_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _file_ids():
    """Tüm jsonl dosyalarının kimliği: [tam_yol, boyut]. Tam yol -> basename
    çakışması yok (her cycle dosyalar claris_diyalog_01.. olsa bile farklı dataset)."""
    out = []
    for p in _jsonl_paths():
        try:
            out.append([p, os.path.getsize(p)])
        except OSError:
            pass
    return out


def _tokenize_paths(tok, paths, fileobj):
    """Verilen jsonl yollarını tokenize edip fileobj'e (uint16) EKLER. (n_token, n_belge)."""
    import array
    bos, eos = tok.tok("<bos>"), tok.tok("<eos>")
    u, b, s = tok.tok("<user>"), tok.tok("<bot>"), tok.tok("<sys>")
    buf = array.array("H")
    n = ndocs = 0
    t0 = time.time()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "messages" in obj:                     # SOHBET: rol tokenlı
                seq = [bos]
                for m in obj["messages"]:
                    role = m.get("role", "user")
                    rtok = b if role == "assistant" else (s if role == "system" else u)
                    seq.append(rtok)
                    seq.extend(tok.encode(m.get("content", "")))
                seq.append(eos)
                for _ in range(DIALOG_REPEAT):
                    buf.extend(seq)
                    n += len(seq)
            elif "text" in obj:                       # DÜZ metin (web/altyazı/kitap)
                seq = [bos] + tok.encode(obj["text"]) + [eos]
                buf.extend(seq)
                n += len(seq)
            else:
                continue
            ndocs += 1
            if len(buf) >= 8_000_000:                 # ~16MB parça -> diske boşalt
                buf.tofile(fileobj)
                buf = array.array("H")
            if ndocs % 200000 == 0:
                print(f"   tokenize {ndocs:,} belge | {n:,} token | {time.time()-t0:.0f}s")
    if buf:
        buf.tofile(fileobj)
    return n, ndocs


# --- PARALEL TOKENIZE (VERİ İŞLEME HIZI) -----------------------------------
# Tek süreçte tokenize CPU-bound + GIL'li (saf-python BPE) -> yavaş (~1B token ~16dk).
# Korpusu dosya-gruplarına bölüp ÇOK SÜREÇLİ tokenize edersek 4 vCPU'da ~4x hızlanır.
# Eğitimde örnekler RASTGELE pencere -> token AKIŞ SIRASI önemsiz; cache kimliği dosya-kümesi
# + boyut ile doğrulanır (içerik-sırası değil) -> paralel SONUÇ birebir geçerli. ÇÖKMEZ:
# herhangi bir hata olursa SESSİZCE seri (_tokenize_paths) moda düşer (eski davranış).
def _tok_worker_chunk(args):
    """Bir işçi: verilen dosya grubunu KENDİ geçici .bin'ine tokenize eder. (tmp, n, ndoc)."""
    tok, paths, tmp_path = args
    with open(tmp_path, "wb") as f:
        n, ndocs = _tokenize_paths(tok, paths, f)
    return (tmp_path, n, ndocs)


def _tokenize_paths_fast(tok, paths, out_fileobj):
    """_tokenize_paths'in PARALEL sarmalı. Aynı imza/sonuç. out_fileobj'e (uint16) EKLER."""
    import shutil
    use_par = os.environ.get("CLARIS_PARALLEL_TOK", "1") != "0"
    nproc = min(len(paths), (os.cpu_count() or 1), 8)
    if not use_par or nproc < 2:
        return _tokenize_paths(tok, paths, out_fileobj)         # seri (tek dosya/CPU)
    # Dosyaları nproc gruba round-robin dağıt (boyut dengesi) -> her grup bir işçi
    groups = [paths[i::nproc] for i in range(nproc)]
    tasks, tmp_paths = [], []
    for gi, g in enumerate(groups):
        if not g:
            continue
        tp = os.path.join(OUT, ".claris_tok_tmp_%d_%d.bin" % (os.getpid(), gi))
        tmp_paths.append(tp)
        tasks.append((tok, g, tp))
    try:
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("spawn")               # spawn = temiz süreç (DDP/CUDA fork sorunu YOK)
        results = []
        with cf.ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as ex:
            for r in ex.map(_tok_worker_chunk, tasks):   # ex.map giriş SIRASINI korur
                results.append(r)
    except Exception as e:                          # paralel patladı -> out_fileobj'e DOKUNULMADI
        print(f"[veri] paralel tokenize başarısız ({str(e)[:60]}) -> SERİ moda düşülüyor")
        for tp in tmp_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
        return _tokenize_paths(tok, paths, out_fileobj)         # güvenli seri yedek
    # Paralel BAŞARILI -> geçici bin'leri SIRAYLA out_fileobj'e birleştir (tek yazıcı yol)
    total_n = total_doc = 0
    for (tp, n, ndocs) in results:
        with open(tp, "rb") as tf:
            shutil.copyfileobj(tf, out_fileobj, length=16 * 1024 * 1024)
        total_n += n
        total_doc += ndocs
        try:
            os.remove(tp)
        except OSError:
            pass
    print(f"[veri] PARALEL tokenize: {len(tasks)} işçi | {total_n:,} token | {total_doc:,} belge")
    return total_n, total_doc


def _collect_token_metas():
    """Aday meta dosyaları: yerel TOKENS_META + girdi dizinlerindeki hepsi.
    Shard'lar farklı kaynaklarda olabileceği için meta ile bin aynı klasörde aranmaz;
    bin çözümünü _resolve_shards yapar (ad + boyut eşleşmesi)."""
    metas = [TOKENS_META]
    for d in INPUT_DIRS:
        if not os.path.isdir(d):
            continue
        for r, _, fs in os.walk(d):
            for stem in TOKEN_STEMS:
                if stem + ".meta.json" in fs:
                    metas.append(os.path.join(r, stem + ".meta.json"))
    return metas


def _bin_index():
    """OUT + girdi dizinlerindeki tüm calisra_tokens*.bin dosyaları: ad -> [yol,...]."""
    idx = {}
    roots = [OUT] + [d for d in INPUT_DIRS if os.path.isdir(d)]
    for root in roots:
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.startswith(TOKEN_STEMS) and f.endswith(".bin"):
                    idx.setdefault(f, []).append(os.path.join(r, f))
    return idx


def _meta_shards(meta):
    """meta -> [[ad, n_token], ...]. Eski (shard'sız) meta = tek parça (geriye uyum)."""
    if meta.get("shards"):
        return [[str(n), int(t)] for n, t in meta["shards"]]
    return [[_shard_name(0), int(meta.get("n_tokens", 0))]]


def _resolve_shards(shards, bin_idx, meta_dir):
    """Her shard'ı diskte bul: ad eşleşir + boyut TAM 2×n_token (bozuk/yarım -> geçersiz).
    Bulunamayan tek shard bile cache'i geçersiz kılar. Döner: [yol,...] ya da None."""
    out = []
    for name, ntok in shards:
        want = int(ntok) * 2
        cands = [os.path.join(meta_dir, name)] + bin_idx.get(name, [])
        p = next((c for c in cands
                  if os.path.exists(c) and os.path.getsize(c) == want), None)
        if p is None:
            return None
        out.append(p)
    return out


class _ShardWriter:
    """uint16 token akışını SHARD_CAP_BYTES'lık calisra_tokens*.bin parçalarına yazar.
    Mevcut shard listesi ([yol, n_token]) devralınır -> son parçaya EKLER, dolunca
    yeni parça açar. write() 2-bayt (token) hizasını korur; array.tofile ve
    shutil.copyfileobj bu nesneye doğrudan yazabilir."""

    def __init__(self, out_dir, shards=None, cap=None):
        self.out_dir = out_dir
        self.cap = max(2, int(cap if cap is not None else SHARD_CAP_BYTES)) & ~1
        self.shards = [list(s) for s in (shards or [])]     # [yol, n_token]
        self.f, self.size = None, 0
        if self.shards:
            p = self.shards[-1][0]
            sz = os.path.getsize(p)
            if sz < self.cap:                               # son shard'da yer var -> ekle
                self.f, self.size = open(p, "ab"), sz

    def _roll(self):
        if self.f:
            self.f.close()
        p = os.path.join(self.out_dir, _shard_name(len(self.shards)))
        self.shards.append([p, 0])
        self.f, self.size = open(p, "wb"), 0

    def write(self, b):
        mv = memoryview(b)
        done = 0
        while done < len(mv):
            if self.f is None or self.size >= self.cap:
                self._roll()
            take = min(len(mv) - done, self.cap - self.size) & ~1
            assert take > 0, "tek bayt kaldı — uint16 akışı bozuk (çift bayt beklenir)"
            self.f.write(mv[done:done + take])
            self.shards[-1][1] += take // 2
            self.size += take
            done += take
        return done

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
        if not self.shards:                 # hiç veri yazılmadı -> boş shard0 (eski davranış)
            p = os.path.join(self.out_dir, _shard_name(0))
            open(p, "wb").close()
            self.shards.append([p, 0])


class ShardedTokens:
    """Birden çok uint16 memmap'i TEK sanal diziymiş gibi sunar (sadece okuma).
    get_batch'in pencere dilimleri shard sınırını aşarsa parçalar birleştirilir
    (2048 token'lık kopya = önemsiz). Havuzun tamamında rastgele örnekleme korunur."""

    def __init__(self, arrs):
        self.arrs = arrs
        self.offs = np.cumsum([0] + [len(a) for a in arrs])
        self.n = int(self.offs[-1])

    def __len__(self):
        return self.n

    def __getitem__(self, sl):
        start, stop = sl.start or 0, min(sl.stop if sl.stop is not None else self.n, self.n)
        i = int(np.searchsorted(self.offs, start, side="right")) - 1
        pieces = []
        while start < stop:
            a, base = self.arrs[i], int(self.offs[i])
            e = min(stop - base, len(a))
            pieces.append(a[start - base:e])
            start, i = base + e, i + 1
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def to_array(self):
        """RAM kopyası (CLARIS_RAM_DATA yolu) — np.array = gerçek kopya."""
        return np.concatenate([np.array(a, dtype=np.uint16) for a in self.arrs])


def open_token_data(shard_paths):
    """Shard yollarını okunur veri olarak aç: tek shard -> düz memmap (eski davranış),
    çok shard -> ShardedTokens sanal akış."""
    if len(shard_paths) == 1:
        return np.memmap(shard_paths[0], dtype=np.uint16, mode="r")
    return ShardedTokens([np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths])


def prepare_tokens(tok):
    """jsonl -> DİSKE (uint16, SHARD'lı). ARTIMLI: önceki cache'i bulur, SADECE YENİ
    dosyaları ekler (tüm korpusu baştan tokenize ETMEZ). Cache SHARD_CAP_BYTES'a
    ulaşınca yeni calisra_tokens_NNN.bin parçası açılır (tek dataset boyut sınırı).
    Döner: ([shard_yolları], n_token_toplam)."""
    assert tok.size <= 65536, "VOCAB 65536'yı aşarsa uint16 yetmez (token dtype büyüt)."
    base = {"vocab": int(tok.size), "dialog_repeat": DIALOG_REPEAT}
    cur = _file_ids()
    # KİMLİK = (basename, boyut) -> yollar oturumlar arası değişse bile eşleşir
    # (tam yol kullanınca her oturum yeni yol -> cache asla tutmaz = baştan tokenize).
    def _nrm(e):
        return (os.path.basename(e[0]), int(e[1]))
    cur_norm = {_nrm(e) for e in cur}
    os.makedirs(OUT, exist_ok=True)

    # CLARIS_TRUST_CACHE=1: alt-küme şartını GEVŞET — cache'i tek doğruluk kaynağı say.
    # Amacı: depolama kotası. Normalde eski jsonl dataset'leri hep ekli
    # kalmalı (alt-küme şartı); 50-100 cycle'da bu ~100GB+ jsonl birikir. Cache .bin
    # zaten TÜM eski veriyi taşıyor (uint16, jsonl'in ~1/3'ü) -> bu bayrakla eski jsonl
    # dataset'leri notebook'tan çıkarılabilir; veri cache'ten aynen eğitilmeye devam eder.
    # DİKKAT: bu moddayken cache'te olan bir jsonl'i SONRADAN tekrar ekleme (meta dosya
    # listesi birleşim tutar, çift eklemez — ama farklı boyutlu kopyası çift sayılır).
    TRUST_CACHE = os.environ.get("CLARIS_TRUST_CACHE", "0") == "1"
    # ⚠️ VERİ KORUMASI — VARSAYILAN AÇIK, KAPATMA.
    # models/calisra_tokens*.bin Calisra'ya giden SYMLINK'tir (33B token, haftalarca emek).
    # Claris onu SADECE OKUR. Bu bayrak açıkken cache eksik/uyumsuz görünürse Claris
    # yeniden tokenize etmeye kalkmaz, RuntimeError ile DURUR.
    # NEDEN VAR: bir kez symlink üzerinden open(...,"wb") yapıldı ve Calisra'nın 17.18GB
    # shard 0'ı 0 byte'a düştü (jsonl'den tam rebuild gerekti). Bir daha olmasın.
    # Kapatman gereken TEK durum: Claris'e AYRI bir bin ürettirmek istersen (vocab aynı
    # olduğu sürece buna gerek YOK -- paylaşılan bin doğrudan çalışır).
    BIN_RO = os.environ.get("CLARIS_BIN_RO", "1") == "1"

    # En çok dosya kapsayan, geçerli, dosyaları current'in ALT KÜMESİ olan cache'i seç.
    # Shard'lar meta'dan bağımsız dataset'lerde olabilir -> önce ad->yol indeksi kur.
    bin_idx = _bin_index()
    best = None   # (kapsanan_dosya, shard_yolları, shard_listesi, meta, cached_set)
    for metap in _collect_token_metas():
        if not os.path.exists(metap):
            continue
        try:
            meta = json.load(open(metap, encoding="utf-8"))
        except Exception:
            continue
        if meta.get("vocab") != base["vocab"] or meta.get("dialog_repeat") != base["dialog_repeat"]:
            continue
        shards = _meta_shards(meta)
        if sum(n for _, n in shards) != int(meta.get("n_tokens", -1)):
            continue                                   # meta iç tutarsız -> atla
        paths = _resolve_shards(shards, bin_idx, os.path.dirname(metap))
        if paths is None:
            continue                                   # shard eksik/bozuk/yarım -> atla
        cset = {_nrm(e) for e in meta.get("files", [])}   # eski meta tam-yol olsa da basename'e iner
        if cset and (TRUST_CACHE or cset <= cur_norm) and (best is None or len(cset) > best[0]):
            best = (len(cset), paths, shards, meta, cset)

    if best:
        _, paths, shards, meta, cset = best
        if TRUST_CACHE and not (cset <= cur_norm) and IS_MAIN:
            print(f"[veri] TRUST_CACHE: cache'teki {len(cset - cur_norm)} eski dosya artık "
                  f"görünmüyor -> tokenleri cache'ten AYNEN kullanılıyor (veri kaybı yok)")
        new_paths = [p for p, sz in cur if _nrm([p, sz]) not in cset]
        if not new_paths:                              # TAM uyum -> tokenize YOK
            print(f"[veri] token CACHE tam (tokenize YOK): {len(paths)} shard | "
                  f"{int(meta['n_tokens']):,} token | {paths[0]}")
            return paths, int(meta["n_tokens"])
        if BIN_RO:                                     # paylaşımlı bin: artımlı YAZMA yasak
            raise RuntimeError(
                f"[CLARIS_BIN_RO] {len(new_paths)} yeni jsonl var ama bin PAYLAŞIMLI (read-only). "
                f"Claris paylaşılan bin'e yazamaz — Calisra tarafında build_tokens çalıştır.")
        # ARTIMLI: dondurulmuş shard'lar YERİNDE kalır (kopya yok); sadece SON shard'a
        # eklenir. Son shard başka (read-only) girdi klasöründeyse OUT'a kopyalanır;
        # zaten doluysa kopya da yok, direkt yeni shard açılır.
        print(f"[veri] ARTIMLI tokenize: {len(cset)} dosya CACHE'ten ({len(paths)} shard), "
              f"{len(new_paths)} YENİ ekleniyor...")
        local = [[p, n] for p, (_, n) in zip(paths, shards)]
        if os.path.getsize(local[-1][0]) < SHARD_CAP_BYTES:
            dst = os.path.join(OUT, os.path.basename(local[-1][0]))
            if os.path.abspath(local[-1][0]) != os.path.abspath(dst):
                import shutil
                shutil.copyfile(local[-1][0], dst)
            local[-1][0] = dst
        w = _ShardWriter(OUT, local)
        add_n, add_doc = _tokenize_paths_fast(tok, new_paths, w)
        w.close()
        n_total = int(meta["n_tokens"]) + add_n
        # meta dosya listesi = eski cache kümesi ∪ şu an görünenler -> TRUST_CACHE modunda
        # görünmeyen eski dosyaların kaydı KAYBOLMAZ (tekrar eklenirse çift tokenize olmaz).
        files_out = sorted(cset | cur_norm) if TRUST_CACHE else [_nrm(e) for e in cur]
        json.dump({**base, "n_tokens": n_total, "files": [list(t) for t in files_out],
                   "shards": [[os.path.basename(p), n] for p, n in w.shards]},
                  open(TOKENS_META, "w", encoding="utf-8"))
        print(f"[veri] artımlı bitti: +{add_n:,} token ({add_doc:,} yeni belge) -> "
              f"toplam {n_total:,} ({len(w.shards)} shard)")
        return [p for p, _ in w.shards], n_total

    # FULL: hiç cache yok / uymuyor (ör. vocab değişti) -> baştan
    if BIN_RO:                                          # paylaşımlı bin: YAZMA yasak (truncate ETME)
        raise RuntimeError(
            "[CLARIS_BIN_RO] Uygun paylaşılan bin cache'i BULUNAMADI ve read-only mod. "
            "Yazmıyorum (Calisra bin'ini bozmamak için). Kontrol: TÜM calisra_tokens*.bin "
            "+ calisra_tokens.meta.json + bpe.json Claris/models'e symlink'li mi? "
            "(Calisra yeni shard eklediyse symlink'i güncelle.)")
    print("[veri] tam tokenize (uygun cache yok)...")
    w = _ShardWriter(OUT)
    n, ndocs = _tokenize_paths_fast(tok, [p for p, _ in cur], w)
    w.close()
    # önceki koşudan kalmış, artık referanssız üst-indeksli shard'ları temizle
    keep = {os.path.basename(p) for p, _ in w.shards}
    for stale in glob.glob(os.path.join(OUT, "calisra_tokens_*.bin")):
        if os.path.basename(stale) not in keep:
            try:
                os.remove(stale)
            except OSError:
                pass
    json.dump({**base, "n_tokens": n, "files": [_nrm(e) for e in cur],
               "shards": [[os.path.basename(p), t] for p, t in w.shards]},
              open(TOKENS_META, "w", encoding="utf-8"))
    print(f"[veri] tokenize bitti: {ndocs:,} belge | {n:,} token -> {len(w.shards)} shard")
    return [p for p, _ in w.shards], n


def get_batch(data, split_idx, is_val):
    """data = np.memmap | ShardedTokens | ndarray (uint16). Sadece batch kadar dilim
    okunur -> RAM dostu. Shard'lıda pencere sınır aşarsa parçalar birleştirilir."""
    lo, hi = (split_idx, len(data)) if is_val else (0, split_idx)
    # Dilim CONTEXT'ten kısaysa lo'yu AŞAĞI çek. (Eskiden hi yukarı zorlanıyordu ->
    # memmap sonundan KISA okuma -> np.stack boyut hatası. Küçük val diliminde çökerdi.)
    if hi - lo < CONTEXT_LEN + 2:
        lo = max(0, hi - (CONTEXT_LEN + 2))
    if hi - CONTEXT_LEN - 1 <= lo:
        raise RuntimeError(f"veri çok küçük: {len(data):,} token, bağlam {CONTEXT_LEN} "
                           f"pencereye yetmiyor -> veri ekle ya da CONTEXT_LEN düşür")
    ix = np.random.randint(lo, hi - CONTEXT_LEN - 1, size=BATCH_SIZE)
    x = np.stack([np.asarray(data[i:i + CONTEXT_LEN], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1:i + 1 + CONTEXT_LEN], dtype=np.int64) for i in ix])
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


# BitNet İKİ-AŞAMA reçetesi (env ile, opsiyonel). CLARIS_LR_TOTAL=0 (vars.) -> tek-aşama
# cosine (Calisra gibi, güvenli, continual-resume dostu; 2× LR zaten BitNet için). >0 verilirse:
# o SABİT ufka göre BitNet 2-aşama -> [0,mid] yüksek-tepe cosine + WD 0.1; [mid,T] ANİ düşük
# tepeye in + cosine ~0 + WD 0 (raporun "abrupt decay + WD kapat" = geç-dönem hızlanması).
# Sabit ufuk şart: resume'da per-commit total değişir, 2-aşama kayardı; LR_TOTAL sabit tutar.
LR_TOTAL = int(os.environ.get("CLARIS_LR_TOTAL", "0"))
LR_STAGE2_PEAK = 0.2 * LR


def lr_at(step, total):
    if LR_TOTAL <= 0:                              # TEK-AŞAMA (varsayılan, güvenli)
        warm = min(WARMUP, max(1, total // 2))
        if step < warm:
            return LR * step / max(1, warm)
        r = min(max((step - warm) / max(1, total - warm), 0.0), 1.0)
        return 0.1 * LR + 0.5 * (LR - 0.1 * LR) * (1 + math.cos(math.pi * r))
    # İKİ-AŞAMA (sabit ufuk LR_TOTAL)
    T = LR_TOTAL
    mid = T // 2
    warm = min(WARMUP, max(1, T // 10))
    if step < warm:
        return LR * step / max(1, warm)
    if step < mid:                                # AŞAMA 1: yüksek tepe -> cosine
        r = min(max((step - warm) / max(1, mid - warm), 0.0), 1.0)
        return 0.1 * LR + 0.5 * (LR - 0.1 * LR) * (1 + math.cos(math.pi * r))
    r = min(max((step - mid) / max(1, T - mid), 0.0), 1.0)   # AŞAMA 2: ANİ düşük tepe -> ~0
    lo = 0.02 * LR
    return lo + 0.5 * (LR_STAGE2_PEAK - lo) * (1 + math.cos(math.pi * r))


def wd_at(step):
    """BitNet: ilk yarı WD 0.1, ikinci yarı 0 (geç-dönem loss düşüşünü hızlandırır).
    LR_TOTAL=0 iken sabit WEIGHT_DECAY (tek-aşama)."""
    if LR_TOTAL <= 0:
        return WEIGHT_DECAY
    return WEIGHT_DECAY if step < LR_TOTAL // 2 else 0.0


def main():
    torch.manual_seed(SEED)
    session_t0 = time.time()              # süreye göre otomatik durmak için (tokenize dahil)

    # DDP başlat: her rank ayrı süreç, torchrun başlatıyor
    if USE_DDP:
        import torch.distributed as dist
        from datetime import timedelta
        torch.cuda.set_device(DDP_LOCAL_RANK)
        # TIMEOUT UZUN (3h): rank0 tokenize/BPE ederken (DAKİKALARCA, cache yoksa ~10dk+)
        # rank1 broadcast/barrier'da bekler. NCCL varsayılan timeout 10dk -> uzun tokenize'da
        # rank1 TIMEOUT eder + çöker (gördüğün hata: BROADCAST ran for 600061ms). 3h = bol pay.
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(hours=3))
        np.random.seed(SEED + DDP_RANK)   # her rank FARKLI veri örneklesin (çeşitlilik)
        if IS_MAIN:
            print(f"[DDP] {DDP_WORLD} süreç (NCCL) | rank {DDP_RANK} -> {device} | EŞİT yük")
    else:
        np.random.seed(SEED)
    if IS_MAIN:
        print(f"🎯 Donanım: {device}{' (DDP)' if USE_DDP else ' (DataParallel)' if N_GPU>1 else ''}")

    # 1) BPE — VARSA yeniden eğitme (aynı sözlük şart, yoksa resume bozulur).
    # Çıktı dizini oturumlar arası silinebilir; önceki bpe.json'u girdi olarak eklersen bulunur.
    # DDP: tokenize aşamasındaki ile AYNI desen — SADECE rank0 eğitir/yazar (çifte-eğitim +
    # dosyaya eş zamanlı yazma yarışı önlenir), diğer rank'ler barrier'da bekler, rank0
    # bpe.json yolunu broadcast eder, hepsi AYNI dosyayı yükler.
    tok = bpemod.BPE()
    bpe_existing = None
    if IS_MAIN:
        bpe_existing = BPE_PATH if os.path.exists(BPE_PATH) else None
        if bpe_existing is None:
            for _d in INPUT_DIRS:
                if not os.path.isdir(_d):
                    continue
                for _r, _, _fs in os.walk(_d):
                    if "bpe.json" in _fs:
                        bpe_existing = os.path.join(_r, "bpe.json")
                        break
                if bpe_existing:
                    break
        if bpe_existing:
            tok.load(bpe_existing)
            print(f"[BPE] mevcut yüklendi (yeniden eğitilmedi): {bpe_existing} | sözlük {tok.size}")
        else:
            # GURABA UYARISI: checkpoint VAR ama bpe.json YOK -> taze BPE eğitmek vocab'ı
            # KAYDIRIR (31999 vs 31997, BPE eğitimi deterministik değil) -> resume KIRILIR,
            # her sefer SIFIRDAN başlar (soy birikmez). Bu hatayı yakala, yüksek sesle uyar.
            _ck = next((h for d in DATA_DIRS for n in CKPT_NAMES
                        for h in glob.glob(os.path.join(d, "**", n), recursive=True)), None)
            if _ck:
                print("=" * 64)
                print("⚠️  DUR! checkpoint VAR ama bpe.json YOK -> taze BPE eğitilecek!")
                print(f"⚠️  Bulunan checkpoint: {_ck}")
                print("⚠️  Taze BPE vocab'ı KAYDIRIR -> resume KIRILIR -> SIFIRDAN başlar (soy gider).")
                print("⚠️  ÇÖZÜM: checkpoint'i eğiten bpe.json'u DATASET olarak yükle, tekrar çalıştır.")
                print("⚠️  (İLK eğitimse bu uyarıyı yok say — bpe.json doğal olarak şimdi oluşur.)")
                print("=" * 64)
            print("[BPE] sıfırdan eğitiliyor...")
            # 32k vocab için daha zengin kelime örneği (Türkçe form çeşitliliği yakalansın)
            wf = bpemod.collect_word_freq(DATA_DIRS, max_lines=1_500_000)
            tok.train(wf, vocab_size=VOCAB_SIZE)
            os.makedirs(OUT, exist_ok=True)
            tok.save(BPE_PATH)
            bpe_existing = BPE_PATH
            print(f"[BPE] sözlük {tok.size} -> {BPE_PATH}")
    if USE_DDP:
        import torch.distributed as dist
        obj = [bpe_existing]
        dist.broadcast_object_list(obj, src=0)     # rank0 -> herkese bpe.json yolu
        bpe_existing = obj[0]
        dist.barrier()                              # rank0 yazmayı bitirene kadar diğerleri bekler
        if not IS_MAIN:
            tok.load(bpe_existing)                  # diğer rank'ler AYNI dosyayı okur (eğitmez)

    # 2) token akışı -> DİSKE (memmap), RAM'e değil.
    # DDP: SADECE rank0 tokenize eder (çifte-yazma/yarış önle); diğerleri barrier'da bekler,
    # rank0 bin yolunu broadcast eder, hepsi aynı bin'i (paylaşılan çıktı dizini) memmap açar.
    if IS_MAIN:
        print("[veri] tokenize / cache kontrol...")
    shard_paths = None
    if IS_MAIN:
        shard_paths, _ntok = prepare_tokens(tok)
    if USE_DDP:
        import torch.distributed as dist
        obj = [shard_paths]
        dist.broadcast_object_list(obj, src=0)     # rank0 -> herkese shard yolları
        shard_paths = obj[0]
        dist.barrier()
    data = open_token_data(shard_paths)            # 1 shard = düz memmap, N shard = sanal akış
    # 30GB RAM'i KULLAN: token .bin disk yerine RAM'e -> rastgele okumada disk page-fault
    # titremesi yok, batch beslemesi pürüzsüz. DDP'de her süreç KENDİ kopyasını yükler ->
    # toplam = boyut × süreç. ~18GB güvenli pay altında kalırsa otomatik aç; aşarsa memmap kalır.
    _ram = os.environ.get("CLARIS_RAM_DATA", "auto")
    _world = (DDP_WORLD if USE_DDP else 1)
    _nbytes = len(data) * 2
    if _ram == "1" or (_ram == "auto" and _nbytes * _world < 18 * 1024**3):
        if IS_MAIN:
            print(f"[veri] {_nbytes/1e9:.2f}GB token RAM'e yükleniyor (×{_world} süreç) -> sabit hızlı okuma")
        data = (data.to_array() if isinstance(data, ShardedTokens)
                else np.array(data, dtype=np.uint16))   # np.array = GERÇEK kopya (asarray kopyalamaz!)
    split_idx = int(len(data) * (1 - VAL_FRAC))
    if IS_MAIN:
        _src = ("RAM" if isinstance(data, np.ndarray) and not isinstance(data, np.memmap)
                else f"memmap×{len(shard_paths)}")
        print(f"[veri] {len(data):,} token ({_src}) | sözlük {tok.size}")
    if len(data) < CONTEXT_LEN * 10:
        if IS_MAIN:
            print("[HATA] Token çok az. Önce fetch_and_clean ile bol veri çek.")
        return

    # Geçiş başına adım. token/adım = (her-rank batch) × (rank sayısı DDP'de) × CTX × ACCUM.
    # DP'de BATCH_SIZE zaten ×N_GPU içeriyor -> GLOBAL_MULT=1. DDP'de BATCH_SIZE per-rank -> ×N_GPU.
    GLOBAL_MULT = N_GPU if USE_DDP else 1
    tokens_per_step = BATCH_SIZE * GLOBAL_MULT * CONTEXT_LEN * GRAD_ACCUM
    steps_per_pass = max(1, len(data) // tokens_per_step)

    # 3) model + optim
    model = Transformer(tok.size).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    if IS_MAIN:
        print(f"[model] Claris (BitNet 1.58-bit)  katman {N_LAYER}  d_model {N_EMBD}  baş {N_HEAD}  param {nparam/1e6:.1f}M")
        print(f"[amp] autocast dtype = {'bf16 (GradScaler kapalı)' if USE_BF16 else 'fp16 (GradScaler açık)'}")
    # OPTIMIZER: bitsandbytes 8-bit AdamW VARSA kullan -> optimizer state VRAM ~-%75
    # (550M'de ~3GB boşalır -> PER_GPU_BATCH 4'e çıkıp ~1.5-2x hızlanabilirsin).
    # OTOMATİK KURULUM: kütüphane yoksa + GPU varsa kendisi pip install eder
    # (sen unutsan da çalışır). Kuramazsa standart AdamW'ye düşer (çökmez).
    # ÖNEMLİ: TUTARLI ol — hep 8-bit ya da hep fp32. Format değişirse optim-state resume
    # edilmez (model ağırlığı yine yüklenir, aşağıda ayrı try ile korunur).
    def _make_optimizer(params):
        if not USE_8BIT or not IS_CUDA:            # CPU/yerelde 8-bit yok -> standart (DDP "cuda:1" de IS_CUDA)
            # fused=True: AdamW adımını TEK CUDA çekirdeğinde toplar -> param-başına launch yok,
            # büyük modelde optimizer adımı belirgin hızlanır.
            return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
                                     fused=IS_CUDA), False
        bnb = None
        try:
            import bitsandbytes as bnb              # zaten kuruluysa
        except Exception:
            # DDP: SADECE rank0 kurar (tüm rank'ler aynı anda pip = yarış/bozulma);
            # diğerleri barrier'da bekler, sonra hepsi import eder.
            if IS_MAIN:
                print("[optim] bitsandbytes yok -> OTOMATİK kuruluyor (pip install)...")
                try:
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                                   check=True, timeout=600)
                except Exception as ie:
                    print(f"[optim] bitsandbytes kurulamadı ({str(ie)[:50]}) -> standart AdamW")
            if USE_DDP:
                import torch.distributed as dist
                dist.barrier()                     # diğer rank'ler kurulumu bekler
            try:
                import bitsandbytes as bnb
            except Exception:
                bnb = None
        if bnb is not None:
            try:
                o = bnb.optim.AdamW8bit(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
                print("[optim] bitsandbytes AdamW8bit (VRAM -%75). PER_GPU_BATCH 4 deneyebilirsin.")
                return o, True
            except Exception as oe:
                print(f"[optim] AdamW8bit kurulamadı ({str(oe)[:50]}) -> standart AdamW")
        return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
                                 fused=IS_CUDA), False

    opt, _is8bit = _make_optimizer(model.parameters())
    # GradScaler yalnız fp16'da gerekir (overflow önleme). bf16'da fp32-exponent -> overflow
    # yok -> scaler KAPALI (scale/unscale/step/update no-op geçer, döngü aynen çalışır).
    scaler = torch.amp.GradScaler("cuda", enabled=(IS_CUDA and not USE_BF16))
    best_val = float("inf")
    start_step = 0
    prev_tokens = 0          # ŞİMDİYE KADAR eğitilen TOPLAM token (commit'ler arası birikir)
    prev_commits = 0         # kaç commit/session yapıldı (resume'da checkpoint'ten yüklenir)

    # RESUME: kayıtlı model varsa SIFIRDAN değil, ÜSTÜNE devam et.
    # Çıktı dizini oturum başında boşsa önceki claris_model.pt'yi girdi olarak ekle
    # bir veri seti olarak ekle -> DATA_DIRS altında bulunur, devam edilir.
    resume_path = CKPT_PATH if os.path.exists(CKPT_PATH) else None
    if resume_path is None:
        for name in CKPT_NAMES:                    # kayıtlı claris_model.pt'yi ara (resume için)
            for d in DATA_DIRS:
                hits = glob.glob(os.path.join(d, "**", name), recursive=True)
                if hits:
                    resume_path = hits[0]
                    break
            if resume_path:
                break
    if RESUME and resume_path:
        try:
            # KRİTİK: map_location="cpu" — checkpoint (fp32 ağırlık ~2GB + optim state ~1GB)
            # GPU'ya DEĞİL CPU RAM'e açılır. Eskiden device'a yükleniyordu -> dict eğitim
            # boyunca GPU'da ölü ~3GB tutuyordu -> İLK RESUME'da OOM (ilk commit sıfırdan
            # başladığı için bug hiç görünmemişti). load_state_dict CPU'dan GPU'ya tensör
            # tensör kopyalar (pik ~200MB); optimizer state'i torch zaten param cihazına taşır.
            ck = torch.load(resume_path, map_location="cpu")
            cfg = ck.get("config", {})
            now = {"vocab": tok.size, "context": CONTEXT_LEN, "n_embd": N_EMBD,
                   "n_head": N_HEAD, "n_layer": N_LAYER, "arch": "bitnet"}
            # Uyumsuz bir checkpoint (farklı boyut ya da arch bilgisi olmayan) yanlışlıkla
            # yüklenmesin diye config'i karşılaştırıyoruz; tutmuyorsa sıfırdan başlıyoruz.
            mismatch = {k: (cfg.get(k, "?" if k == "arch" else None), v)
                        for k, v in now.items() if cfg.get(k, "?" if k == "arch" else None) != v}
            if not mismatch:
                model.load_state_dict(ck["model"])
                if "optim" in ck:
                    # optim-state AYRI try: 8-bit<->fp32 format değişirse model ağırlığı
                    # KAYBOLMASIN (sadece optim sıfırdan başlar, warmup tekrar eder).
                    try:
                        opt.load_state_dict(ck["optim"])
                    except Exception as _oe:
                        if IS_MAIN:
                            print(f"[resume] optim-state yüklenemedi (format değişti?) -> optim sıfırdan, "
                                  f"model ağırlığı KORUNDU: {str(_oe)[:50]}\n"
                                  f"         NOT: momentum kaybı = ilk adımlarda loss SIÇRAYABİLİR "
                                  f"(normal, toparlar). Tutarlı kal: hep 8-bit ya da hep fp32.")
                start_step = int(ck.get("step", 0))
                best_val = float(ck.get("best_val", best_val))
                # tokens_seen YOKSA (eski checkpoint, sayaç eklenmeden önce) -> step'ten tahmin
                # et (config aynıysa = doğru). Böylece sayaç eski beyinden devam ederken sıfırlanmaz.
                prev_tokens = int(ck.get("tokens_seen", start_step * tokens_per_step))
                prev_commits = int(ck.get("commits", 0))       # birikmiş commit sayısı
                if IS_MAIN:
                    print(f"[resume] {start_step}. adımdan DEVAM (val {best_val:.3f}) — sıfırdan değil")
                    print(f"[birikim] şimdiye kadar {prev_tokens/1e9:.2f}B token | {prev_commits} commit")
            else:
                # Mimari/sözlük değişti -> eski ağırlık ŞEKLİ uymaz, SIFIRDAN (çökme YOK)
                if IS_MAIN:
                    print(f"[resume] mimari/sözlük değişti {mismatch} -> SIFIRDAN başlanıyor "
                          f"(eski .pt kullanılmadı). Resume istiyorsan ayarları eski .pt ile aynı yap.")
        except Exception as e:
            if IS_MAIN:
                print(f"[resume] yüklenemedi, sıfırdan: {e}")
        # ckpt dict'ini HEMEN bırak + CUDA cache'i boşalt: ilk forward (compile pikiyle)
        # maksimum boş VRAM ile başlasın. ck referansı kalsa 3GB eğitim boyunca ölü dururdu.
        try:
            del ck
        except NameError:
            pass
        import gc
        gc.collect()
        if IS_CUDA:
            torch.cuda.empty_cache()

    # ÇOKLU GPU: model_fw = forward sarmalı. 'model' HAM kalır (resume/save/eval onu kullanır).
    #   DDP  -> her GPU eşit/bağımsız, gradyan all-reduce ile senkron (boş VRAM kullanılır).
    #   DP   -> tek süreç batch'i böler (GPU0 darboğaz, fallback).
    #
    # ⚡ torch.compile: HAM 'model' derlenir -> RMSNorm/RoPE/SwiGLU/residual element-wise
    # çekirdekleri füzyonlanır, kernel-launch sayısı düşer (~1.2-1.5x). Param PAYLAŞIMLI:
    # 'model' HAM kalır -> state_dict anahtarları TEMİZ (_orig_mod/module YOK) -> RESUME birebir.
    # DDP, derlenmiş çekirdeği SARAR -> no_sync/all-reduce STANDART çalışır (grad-birikim doğru).
    # suppress_errors=True: derleme bir grafı çeviremezse SESSİZCE eager'a düşer -> ASLA çökmez,
    # en kötü ihtimalle "hız kazancı yok" (resume/ağırlık bozulmaz).
    eff_batch = BATCH_SIZE * (N_GPU if USE_DDP else 1) * GRAD_ACCUM
    COMPILE = os.environ.get("CLARIS_COMPILE", "1") != "0"
    core = model
    if COMPILE and IS_CUDA and hasattr(torch, "compile") and not (N_GPU > 1 and not USE_DDP):
        try:
            # NOT: 'import torch._dynamo' BURADA YAPILMAZ -> 'torch'u local yapıp main başını kırar.
            # suppress_errors zaten modül düzeyinde set edildi (yukarı bak).
            core = torch.compile(model, mode="default", dynamic=False)
            if IS_MAIN:
                print("[model] ⚡ torch.compile AKTİF (mode=default, hata-güvenli eager fallback)")
        except Exception as e:
            core = model
            if IS_MAIN:
                print(f"[model] torch.compile atlandı: {str(e)[:60]}")

    model_fw = core
    if USE_DDP:
        from torch.nn.parallel import DistributedDataParallel as DDP
        try:
            model_fw = DDP(core, device_ids=[DDP_LOCAL_RANK], output_device=DDP_LOCAL_RANK,
                           find_unused_parameters=False)
        except Exception as e:               # DDP derlenmiş modülü saramazsa -> HAM model ile DDP
            if IS_MAIN:
                print(f"[GPU] DDP(compiled) başarısız -> HAM model: {str(e)[:50]}")
            model_fw = DDP(model, device_ids=[DDP_LOCAL_RANK], output_device=DDP_LOCAL_RANK,
                           find_unused_parameters=False)
        if IS_MAIN:
            print(f"[GPU] DDP {N_GPU}×GPU EŞİT | mikro-batch {BATCH_SIZE}/GPU × {N_GPU} GPU "
                  f"× accum {GRAD_ACCUM} = EFEKTİF batch {eff_batch} | checkpoint {GRAD_CKPT} (N={GRAD_CKPT_N if GRAD_CKPT else 0}/{N_LAYER})")
    elif N_GPU > 1:
        model_fw = nn.DataParallel(model)    # DP + compile riskli kombinasyon -> HAM model sar
        print(f"[GPU] {N_GPU} GPU (DataParallel) | mikro-batch {BATCH_SIZE} ({PER_GPU_BATCH}/GPU) "
              f"× accum {GRAD_ACCUM} = EFEKTİF batch {eff_batch} | checkpoint {GRAD_CKPT} (N={GRAD_CKPT_N if GRAD_CKPT else 0}/{N_LAYER})")
    else:
        print(f"[GPU] 1 GPU | mikro-batch {BATCH_SIZE} × accum {GRAD_ACCUM} = efektif {eff_batch} "
              f"| checkpoint {GRAD_CKPT} (N={GRAD_CKPT_N if GRAD_CKPT else 0}/{N_LAYER})")

    # Toplam adım: resume'dan SONRA -> bu session'da PASSES geçiş DAHA yap.
    # (Aksi halde cycle 2'de start_step >= total_iters olur, döngü boş kalır, hiç eğitmez.)
    total_iters = MAX_ITERS if MAX_ITERS > 0 else start_step + steps_per_pass * PASSES
    if IS_MAIN:
        print(f"[plan] {start_step}. adımdan başla | geçiş başına ~{steps_per_pass} adım | "
              f"hedef {total_iters} adım ({'elle' if MAX_ITERS>0 else f'+{PASSES} geçiş'})")

    def save_ckpt(step, best):
        os.makedirs(OUT, exist_ok=True)
        # TOPLAM token = önceki birikim + bu session'da işlenen (adım × token/adım).
        # commit = önceki + 1 (bu session). Beyni SIFIRLAMAZ, sadece metadata ekler.
        tokens_seen = prev_tokens + (step - start_step) * tokens_per_step
        torch.save({
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "step": step,
            "best_val": best,
            "tokens_seen": tokens_seen,        # ŞİMDİYE KADAR eğitilen toplam token
            "commits": prev_commits + 1,       # toplam commit/session sayısı
            "config": {"vocab": tok.size, "context": CONTEXT_LEN,
                       "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
                       "arch": "bitnet"},
        }, CKPT_PATH)

    t0 = time.time()
    step = start_step                      # döngü boş kalırsa (edge) NameError olmasın
    _wd = wd_at(start_step)
    for step in range(start_step + 1, total_iters + 1):
        _lr = lr_at(step, total_iters)
        _wd = wd_at(step)                          # BitNet 2-aşama WD (LR_TOTAL=0 -> sabit)
        for g in opt.param_groups:
            g["lr"] = _lr
            g["weight_decay"] = _wd
        # GRADYAN BİRİKTİRME: opt.step ÖNCESİ GRAD_ACCUM mikro-batch biriktir (efektif
        # batch büyür, VRAM sabit). loss / GRAD_ACCUM -> biriken gradyan doğru ölçekte.
        opt.zero_grad(set_to_none=True)
        loss_step = 0.0
        for _micro in range(GRAD_ACCUM):
            x, y = get_batch(data, split_idx, is_val=False)
            last_micro = (_micro == GRAD_ACCUM - 1)
            # DDP: ara mikro-batch'lerde all-reduce YOK (no_sync) -> sadece SON mikro-batch'te
            # senkronize ol. Yoksa her backward() GPU'lar arası iletişim tetikler (GRAD_ACCUM
            # kat fazla all-reduce) -> DDP beklenen hızı vermez (DP'den bile yavaş olabilir).
            sync_ctx = model_fw.no_sync() if (USE_DDP and not last_micro) else contextlib.nullcontext()
            with sync_ctx:
                loss = model_fw(x, y)                   # SADECE loss döner (DP-dengeli, logits gather yok)
                loss = loss.mean() / GRAD_ACCUM         # DP: her GPU'dan loss ortala; sonra accum böl
                scaler.scale(loss).backward()           # gradyanlar BİRİKİR (zero_grad döngü dışında)
            loss_step = loss_step + loss.detach()   # .item() YOK -> mikro-batch başına GPU sync tetiklenmez (print'te float())
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        # HAFİF İLERLEME (eval değil) -> körlük yok + tok/s ölçümü. SADECE rank0 basar.
        # tok/s GLOBAL (tokens_per_step zaten ×N_GPU içeriyor). flush=True: log buffer beklemesin.
        if IS_MAIN and (step % PROGRESS_EVERY == 0 or step <= start_step + 3):
            done = step - start_step
            tps = done * tokens_per_step / max(time.time() - t0, 1e-6)
            print(f"  adım {step}/{total_iters}  loss {float(loss_step):.3f}  "
                  f"{tps/1000:.1f}k tok/s  {time.time()-t0:.0f}s", flush=True)

        if step % EVAL_EVERY == 0 or step == total_iters:
            # EVAL: SADECE rank0 (HAM model -> DDP collective YOK). Diğerleri barrier'da bekler.
            if IS_MAIN:
                model.eval()
                with torch.no_grad():
                    vl = []
                    for _ in range(20):
                        xv, yv = get_batch(data, split_idx, is_val=True)
                        l = model(xv, yv)          # HAM model (model_fw değil) -> sync yok
                        vl.append(l.mean().item())
                val = sum(vl) / len(vl)
                model.train()
                best_val = min(best_val, val)
                _toplam = prev_tokens + (step - start_step) * tokens_per_step
                print(f" [EVAL] adım {step}/{total_iters}  train {float(loss_step):.3f}  val {val:.3f}  "
                      f"| TOPLAM {_toplam/1e9:.2f}B token ({prev_commits+1}. commit)  "
                      f"{time.time()-t0:.0f}s", flush=True)
                save_ckpt(step, best_val)          # LATEST kaydet (resume edilebilir)
            if USE_DDP:
                import torch.distributed as dist
                dist.barrier()                     # diğer rank'ler rank0 eval+kayıt'ı bekler

        # SÜRE-TABANLI OTO-DUR: MAX_HOURS doldu. DDP'de tüm rank'ler AYNI adımda durmalı (yoksa
        # all-reduce kilitlenir) -> rank0 karar verir, broadcast eder.
        stop = 0
        if IS_MAIN and (time.time() - session_t0) > MAX_HOURS * 3600:
            stop = 1
        if USE_DDP:
            import torch.distributed as dist
            t = torch.tensor([stop], device=device)
            dist.broadcast(t, src=0)               # rank0 -> herkes (senkron dur)
            stop = int(t.item())
        if stop:
            if IS_MAIN:
                save_ckpt(step, best_val)
                print(f"✅ Bu commit TAMAM: {MAX_HOURS}h eğitildi, {step}. adımda kaydedildi "
                      f"(output güvende). Sonraki commit kaldığı yerden devam eder.", flush=True)
            break

    if IS_MAIN:
        save_ckpt(step, best_val)
        _toplam = prev_tokens + (step - start_step) * tokens_per_step
        print(f"✅ Bitti. adım {step}  val {best_val:.3f}  | TOPLAM {_toplam/1e9:.2f}B token "
              f"({prev_commits+1}. commit) -> {CKPT_PATH}  (resume hazır)", flush=True)
    if USE_DDP:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()