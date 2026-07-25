# ============================================================================
# CLARIS — Kaggle eğitim hücresi (2×T4 DDP, BitNet b1.58)
# ----------------------------------------------------------------------------
# Notebook'ta TEK hücre. Bu dosyayı olduğu gibi yapıştır, çalıştır.
#
# GEREKEN DATASET'LER (notebook'a ekle, "Add Input"):
#   1) claris-code      -> bitlinear.py + train_claris.py + bpe.py + bpe.json
#   2) calisra-tokens   -> calisra_tokens.bin (+ meta)      [Calisra ile PAYLAŞ]
#   3) calisra-tokens-001 / -002 / -003                     [33B'nin diğer shard'ları]
#   4) claris-resume    -> claris_model.pt   [İLK commit'te YOK; 2. commit'ten itibaren ekle]
#
# NOT: Claris, Calisra'nın token bin'ini AYNEN okur (vocab ikisinde de 32000).
#      Ayrı token dataset'i YOK, re-tokenize YOK. Beyin (claris_model.pt) ayrı.
#
# SÜREYİ HER COMMIT'TE SEN AYARLA (aşağıda CLARIS_MAX_HOURS):
#   - Hesap TAZE (30h kotanın başı)     -> "11.75"
#   - Calisra da bu hesapta eğitiliyorsa -> kalan saate göre "6" / "5" ...
# ============================================================================
import glob, os, shutil, subprocess

# --- 1) Kodu /kaggle/working'e kopyala (torchrun diskte .py ister) ---
# DİKKAT: Claris 3 dosya ister — train_claris.py bitlinear.py'ye BAĞIMLI
# (from bitlinear import BitLinear). Biri eksikse import patlar.
GEREKLI = ("train_claris.py", "bitlinear.py", "bpe.py")
for f in GEREKLI:
    hits = glob.glob(f"/kaggle/input/**/{f}", recursive=True)
    assert hits, f"{f} bulunamadı — claris-code dataset'i ekli mi?"
    # claris-code'u tercih et (ileride başka kod dataset'i olursa yanlış sürüm gelmesin)
    hits.sort(key=lambda p: (0 if "claris-code" in p else 1, p))
    shutil.copy(hits[0], f"/kaggle/working/{f}")
    print("kopyalandı:", hits[0])

# --- 2) Ortam + hız ayarları ---
env = dict(
    os.environ,
    # DDP kararlılığı (Kaggle sanal 2×T4'te NCCL sessiz takılmasını önler)
    NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", NCCL_SHM_DISABLE="1",
    # Bin-only akış: jsonl olmadan cache kabul (veri zaten tokenize, Calisra'dan paylaşılıyor)
    CLARIS_TRUST_CACHE="1",
    # ⚠️ VERİ KORUMASI — AÇIK BIRAK. Paylaşılan bin'e yazma girişimini RuntimeError ile durdurur
    # (Calisra'nın 33B verisi symlink; bir kez üstüne yazıp shard sıfırlamıştık, guard o yüzden).
    CLARIS_BIN_RO="1",
    # HIZ / VRAM AYARI — CLARIS_CKPT_N = kaç blok recompute edilecek (i < N checkpoint'li).
    # Düşük N = daha çok aktivasyon bellekte = daha çok VRAM + daha hızlı.
    #   N=24 tam checkpoint (en az VRAM) ... N=0 hiç recompute (en çok VRAM, en hızlı).
    # ⚠️ 513M MODEL: 335M'de N=0 (13.7GB) sığıyordu ama 513M daha geniş (d1280) ->
    # N=0 OOM olur (~19GB). N=8: ilk 8 blok recompute, 16 serbest -> ~13-14GB'a oturması
    # beklenir (~6k tok/s, Calisra seviyesi). İLK COMMİT'İ İZLE:
    #   OOM olursa   -> N=12, olmazsa N=16, hâlâ olmazsa N=24 (tam, en güvenli)
    #   bol yer varsa -> N=6, N=4 indirip hız kazan (VRAM 15'e yaklaşana kadar)
    CLARIS_CKPT_N="8",
    # SÜRE — HER COMMIT'TE BURAYI AYARLA (yukarıdaki nota bak)
    CLARIS_MAX_HOURS="11.5",
)

# --- 3) Eğit (2×T4 DDP) ---
subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=2", "train_claris.py"],
    cwd="/kaggle/working", env=env, check=True,
)
