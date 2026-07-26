# -*- coding: utf-8 -*-
"""
KAGGLE YÜKLEYİCİ — düz terminalden dataset güncelleme (tarayıcı/VS Code RAM'i yok).

Kurulum (bir kez):  pip install kaggle   (CLI >= 2.x, yeni token sistemi)

KİMLİK (Kaggle'ın YENİ access-token sistemi, 2025+):
  Settings -> API -> "Generate New Token" artık kaggle.json DEĞİL, KGAT_... diye
  tek satırlık token verir. Ana hesap için Kaggle'ın gösterdiği komut aynen doğru:
    mkdir -p ~/.kaggle && echo 'KGAT_...' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
  Token'da KULLANICI ADI YOK -> bu script adı şu sırayla çözer:
    1) KAGGLE_USERNAME=... env   2) legacy kaggle.json   3) CLI'a sorar (config view)
  (Legacy yol hâlâ var: Settings -> API -> "Legacy API Credentials" -> kaggle.json.)

ÇOK HESAP (bayrak yarışı): her hesabın token'ı AYRI DOSYAYA (yerelde kalır):
  echo 'KGAT_hesap2token' > ~/.kaggle/hesap2.token && chmod 600 ~/.kaggle/hesap2.token
  KAGGLE_API_TOKEN=~/.kaggle/hesap2.token KAGGLE_USERNAME=hesap2adi python kaggle_push.py model
  (KAGGLE_API_TOKEN dosya yolu da kabul eder — CLI dosyaysa içinden okur.
   DİKKAT: KAGGLE_CONFIG_DIR yalnız legacy kaggle.json'a işler; access_token
   dosyasının yeri CLI'da sabit ~/.kaggle/access_token'dır.)
Not: dataset'i 5 hesaba ayrı ayrı YÜKLEME — bir hesap sahip olur, diğerleri
Kaggle'da dataset "Settings -> Collaborators" ile eklenir (private paylaşım, tek upload).

Kullanım:
  python kaggle_push.py tokens   # DEĞİŞEN token shard'ları -> <sen>/calisra-tokens(-NNN)
  python kaggle_push.py tokens --force   # değişiklik takibini atla, tüm shard'ları it
  python kaggle_push.py model    # claris_model.pt               -> <sen>/claris-resume
  python kaggle_push.py code     # bpe.py + bitlinear.py + train_claris.py + bpe.json -> <sen>/claris-code

TOKEN SHARD'LARI: Kaggle tek-dataset sınırı 20GB -> token cache CLARIS_SHARD_GB'lık
(vars. 16GB) parçalara bölünür (build_tokens.py otomatik yapar). Her shard AYRI dataset:
  calisra_tokens.bin      -> calisra-tokens
  calisra_tokens_001.bin  -> calisra-tokens-001   (doğunca notebook'a EKLEMEYİ UNUTMA)
meta.json her zaman SON (büyüyen) shard'ın dataset'iyle gider; eski dataset'lerdeki
bayat meta kopyaları zararsız (eğitim en çok dosya kapsayan GEÇERLİ meta'yı seçer).
Değişiklik takibi .claris_push_state.json'da (dosya adı -> son itilen boyut) ->
dondurulmuş 16GB shard'lar bir daha YÜKLENMEZ; veri günü sadece son shard gider.

İlk koşuda dataset YOKSA oluşturur (private), VARSA "New Version" atar.
Dosyalar hardlink ile hazırlanır (kopya yok -> 16GB bin için ekstra disk/süre sıfır).
DÜRÜST NOT: CLI bant genişliği kazandırmaz (hız = upload hattın); kazancı RAM yememesi,
sekme/oturum derdi olmaması ve script'lenebilmesi. Kesilirse komutu tekrar çalıştır.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# "kaggle" binary'si çoğu kurulumda PATH dışında (~/.local/bin) -> python -m kaggle
# her zaman çalışır (pip paketi yeter, PATH derdi yok).
KAGGLE = [sys.executable, "-m", "kaggle"]

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
M = os.path.join(ROOT, "models")
STATE = os.path.join(ROOT, ".claris_push_state.json")

TARGETS = {
    "model":  {"slug": "claris-resume",
               "files": [os.path.join(M, "claris_model.pt")]},
    "code":   {"slug": "claris-code",
               "files": [os.path.join(ROOT, "bpe.py"),
                         os.path.join(ROOT, "bitlinear.py"),
                         os.path.join(ROOT, "train_claris.py"),
                         os.path.join(M, "bpe.json")]},
}
# NOT: Claris VERİYİ Calisra'nın MEVCUT dataset'lerinden alır (calisra-tokens/-001/-002 +
# bpe.json calisra-code'dan). Ayrı claris-tokens YOK -> `tokens` komutu bilinçli engelli.


def _username():
    """Dataset id'si (<kullanici>/<slug>) için kullanıcı adı. Yeni access-token'da
    (KGAT_...) ad YOK -> sırayla: env -> legacy kaggle.json -> CLI'a sor."""
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    cfg = os.environ.get("KAGGLE_CONFIG_DIR") or os.path.expanduser("~/.kaggle")
    p = os.path.join(os.path.expanduser(cfg), "kaggle.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)["username"]
    except Exception:
        pass
    try:                                    # token/OAuth girişliyse CLI adı kendisi bilir
        r = subprocess.run(KAGGLE + ["config", "view"],
                           capture_output=True, text=True, timeout=60)
        m = re.search(r"username:\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
        if m:
            return m.group(1)
    except Exception:
        pass
    print("[HATA] Kaggle kullanıcı adı bulunamadı. Şunlardan biri lazım:\n"
          "  - komut önüne KAGGLE_USERNAME=kullaniciadi\n"
          "  - ~/.kaggle/access_token dosyasına token (Settings -> API -> Generate New Token)\n"
          "  - ya da legacy kaggle.json (Settings -> API -> Legacy API Credentials)")
    sys.exit(1)


def _stage(files, meta):
    """Geçici klasöre hardlink + dataset-metadata.json (kopya yok, disk yok)."""
    stage = tempfile.mkdtemp(prefix=".kaggle_push_", dir=ROOT)
    for f in files:
        dst = os.path.join(stage, os.path.basename(f))
        try:
            os.link(f, dst)                     # aynı disk -> anlık, 0 ekstra yer
        except OSError:
            shutil.copy2(f, dst)                # farklı disk/fs -> kopyaya düş
    with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return stage


def _run_live(cmd):
    """Alt-süreci çalıştır: çıktıyı CANLI ekrana yaz (tqdm ilerleme çubuğu görünür)
    AYNI ANDA tampona biriktir (create-fallback için 404/not-found taraması).
    capture_output=True 17GB upload'da çubuğu gizliyordu -> bu onu düzeltir."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         bufsize=0)
    buf = []
    while True:
        chunk = p.stdout.read(4096)             # \r'li çubuklar için satır değil chunk
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        buf.append(chunk)
    p.wait()
    return p.returncode, b"".join(buf).decode("utf-8", "replace")


def _dataset_exists(ds_id):
    """Dataset var mı? UCUZ metadata sorgusu (upload YOK). None = kararsız (ağ hatası).
    KRİTİK: bu kontrol olmadan ilk push'ta `version` 17GB'ı yükleyip 404 alıyordu,
    sonra `create` AYNI 17GB'ı BAŞTAN yüklüyordu = çift yükleme (2 saat boşa)."""
    rc, out = _run_live_quiet(KAGGLE + ["datasets", "status", ds_id])
    low = out.lower()
    if "not found" in low or "404" in low or "does not exist" in low:
        return False
    if rc == 0 and out.strip():
        return True
    return None


def _run_live_quiet(cmd):
    """_run_live gibi ama SESSİZ (metadata sorgusu — ekrana basma)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 1, str(e)


def _push_dataset(user, slug, files):
    """Bir dataset'e New Version at (yoksa oluştur). True = başarı.
    Yüklemeden ÖNCE varlığı kontrol eder -> 17GB'ı ASLA iki kez yüklemez."""
    ds_id = f"{user}/{slug}"
    total = sum(os.path.getsize(f) for f in files)
    meta = {"title": slug, "id": ds_id,
            "licenses": [{"name": "CC0-1.0"}]}   # private dataset'te kozmetik alan
    stage = _stage(files, meta)
    t0 = time.time()
    try:
        exists = _dataset_exists(ds_id)
        if exists is True:
            print(f"[push] {ds_id}  ({total/1e9:.2f} GB, {len(files)} dosya) — New Version...")
            msg = time.strftime("push %Y-%m-%d %H:%M")
            rc, out = _run_live(KAGGLE + ["datasets", "version", "-p", stage, "-m", msg])
        else:   # False (yok) veya None (kararsız) -> create; zaten varsa create hızlı hata verir
            if exists is None:
                print(f"[push] {ds_id}: varlık kontrolü belirsiz -> create denenecek")
            print(f"[push] {ds_id}  ({total/1e9:.2f} GB, {len(files)} dosya) — oluşturuluyor (private)...")
            rc, out = _run_live(KAGGLE + ["datasets", "create", "-p", stage])
            # nadir yarış: "already exists" -> version'a düş (create'in yüklemesi boşa değil,
            # Kaggle hash-cache'ler; yine de version doğru yol)
            if rc != 0 and "already exist" in out.lower():
                print("[push] dataset zaten var -> New Version...")
                msg = time.strftime("push %Y-%m-%d %H:%M")
                rc, out = _run_live(KAGGLE + ["datasets", "version", "-p", stage, "-m", msg])
        if rc == 0:
            mins = (time.time() - t0) / 60
            print(f"[push] TAMAM: {ds_id}  ({total/1e9:.2f} GB, {mins:.1f} dk)")
            return True
        print(f"[push] BAŞARISIZ (rc={rc}) — çıktıya bak, komutu tekrar dene.")
        return False
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def _push_tokens(user, force):
    """Token shard'larını iter: shard i -> calisra-tokens(-NNN). Sadece boyutu
    değişenler gider (state dosyası); meta SON shard'ın dataset'iyle taşınır."""
    metap = os.path.join(M, "calisra_tokens.meta.json")
    if not os.path.exists(metap):
        print(f"[HATA] {metap} yok — önce: python build_tokens.py")
        sys.exit(1)
    meta = json.load(open(metap, encoding="utf-8"))
    shards = meta.get("shards") or [["calisra_tokens.bin", int(meta.get("n_tokens", 0))]]
    paths = [os.path.join(M, name) for name, _ in shards]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[HATA] shard dosyası yok: {missing}")
        sys.exit(1)

    state = _load_state()
    last = len(paths) - 1
    todo = [i for i, p in enumerate(paths)
            if force or state.get(os.path.basename(p)) != os.path.getsize(p)]
    if not todo:
        print("[push] değişen shard yok — yüklenecek bir şey yok. (--force ile zorla)")
        return
    print(f"[push] {len(todo)}/{len(paths)} shard değişmiş -> yüklenecek")
    ok_all = True
    for i in todo:
        slug = "calisra-tokens" if i == 0 else f"calisra-tokens-{i:03d}"
        files = [paths[i]] + ([metap] if i == last else [])
        if _push_dataset(user, slug, files):
            state[os.path.basename(paths[i])] = os.path.getsize(paths[i])
            json.dump(state, open(STATE, "w", encoding="utf-8"))
            if i == last and i > 0:
                print(f"[push] NOT: {slug} YENİ dataset ise Kaggle notebook'una eklemeyi unutma!")
        else:
            ok_all = False
    if not ok_all:
        sys.exit(1)


def main():
    tgt = sys.argv[1] if len(sys.argv) > 1 else ""
    force = "--force" in sys.argv
    if tgt == "tokens":
        print("[Claris] tokens YÜKLENMEZ — Claris veriyi Calisra'nın MEVCUT dataset'lerinden\n"
              "  alır (notebook'a calisra-tokens/-001/-002 + calisra-code'un bpe.json'unu ekle).\n"
              "  Sadece: python kaggle_push.py code | model")
        sys.exit(1)
    if tgt not in TARGETS:
        print("kullanım: python kaggle_push.py code | model   (tokens YOK — calisra paylaşır)")
        sys.exit(1)
    if importlib.util.find_spec("kaggle") is None:   # import ETME: paket import'ta
        print("[HATA] kaggle paketi yok:  pip install kaggle")   # auth ister/çıkar
        sys.exit(1)
    user = _username()

    t = TARGETS[tgt]
    missing = [f for f in t["files"] if not os.path.exists(f)]
    if missing:
        print(f"[HATA] dosya yok: {missing}")
        sys.exit(1)
    if not _push_dataset(user, t["slug"], t["files"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
