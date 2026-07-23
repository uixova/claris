# -*- coding: utf-8 -*-
"""
Calisra'nun BPE (Byte-Pair Encoding) tokenizer'ı... Qwen/Llama'daki mantığın aynısı ve
model bunun üstüne kuruluyor.

Kelime kelime bir sözlük tutmuyoruz. BPE metni sık geçen küçük parçalara bölüyor:
  "geliyorum" -> ["gel", "iyor", "um</w>"]   gibi.
Böyle olunca Türkçe ekleri (-iyor, -um, -lar falan) kendiliğinden öğreniliyor, sözlük
küçük kalıyor (~8k-32k) ve bilinmeyen kelime neredeyse hiç çıkmıyor.

Tamamen saf Python, torch falan yok... yani yerelde rahatça eğitip test edebilirsin.

Kullanımı:
  python bpe.py train [vocab_size]   # data/ klasöründen eğitir -> models/bpe.json
  python bpe.py test                 # encode/decode gidip gelme denemesi
"""

import os
import re
import sys
import json
import glob
from collections import Counter

try:                                  # Kaggle hücresine yapıştırınca __file__ yok
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
BPE_PATH = os.path.join(ROOT, "models", "bpe.json")

# Özel + rol tokenları (sohbet yapısı için).
# İLK 7: temel (pad/bos/eos/unk + rol user/bot/sys) — DEĞİŞMEZ sıra.
# SONRAKİLER (gelecek yetenekler için REZERVE — şimdi ekleniyor çünkü vocab'a token
# eklemek = vocab_size değişir = model şekli değişir = SIFIRDAN eğitim. Fresh restart
# = bunları eklemek için TEK temiz pencere; sonradan eklemek her şeyi kırar):
#   <think></think>       -> ileride akıl-yürütme SFT'i (zincirleme düşünce; DeepSeek-R1 tarzı)
#   <tool></tool>         -> tool çağrısı/sonucu (tools/: calc/rag/memory yol haritası)
#   <reserved_0/1>        -> bilinmeyen gelecek ihtiyaç (Llama/Qwen de rezerve tutar)
# Eğitim verisinde geçmezlerse embedding'leri init'te kalır (zararsız; üretimde maskelenir),
# ama gün gelince SFT hazır bulur -> sıfırdan eğitime gerek kalmaz.
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<bot>", "<sys>",
            "<think>", "</think>", "<tool>", "</tool>", "<reserved_0>", "<reserved_1>"]
END = "</w>"   # kelime sonu işareti

# Türkçe duyarlı küçük harf
_TR_LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
_TOKEN_RE = re.compile(r"[a-zçğıöşüâîû0-9]+|[^\sa-zçğıöşüâîû0-9]", re.UNICODE)


def normalize(text):
    return text.translate(_TR_LOWER).lower()


def pre_tokenize(text):
    """Metni kelime ve tekil noktalama parçalarına ayırır."""
    return _TOKEN_RE.findall(normalize(text))


class BPE:
    def __init__(self):
        self.merges = {}      # (a, b) -> rank
        self.vocab = {}       # symbol -> id
        self.id2sym = {}      # id -> symbol
        self._cache = {}

    def train(self, word_freq, vocab_size=16000, min_pair=2, verbose=True):
        """HIZLI (incremental) BPE: her merge'de TÜM korpusu değil, yalnız o çifti
        içeren kelimeleri günceller. word_freq: Counter(kelime -> sıklık)."""
        import heapq
        from collections import defaultdict
        # Kelimeleri sembol listesi olarak tut (id ile)
        seqs, freqs = [], []
        base = set()
        for w, f in word_freq.items():
            if not w:
                continue
            sq = list(w) + [END]
            seqs.append(sq)
            freqs.append(f)
            base.update(sq)

        # Çift istatistikleri + indeks (çift -> içeren kelime id'leri)
        pairs = defaultdict(int)
        where = defaultdict(set)
        for i, sq in enumerate(seqs):
            f = freqs[i]
            for j in range(len(sq) - 1):
                p = (sq[j], sq[j + 1])
                pairs[p] += f
                where[p].add(i)

        # En sık çifti O(log) bulmak için heap (lazy silme: bayat girdiler atlanır)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)

        def _push(p):
            heapq.heappush(heap, (-pairs[p], p))

        target = max(0, vocab_size - len(SPECIALS) - len(base))
        merges = []
        for step in range(target):
            best = None
            while heap:                            # geçerli (bayat olmayan) en sık çift
                negc, p = heapq.heappop(heap)
                if pairs.get(p, 0) == -negc:
                    best = p
                    break
            if best is None or pairs[best] < min_pair:
                break
            cnt = pairs[best]
            merges.append(best)
            a, b = best
            ab = a + b
            touched = set()
            for i in list(where[best]):            # SADECE bu çifti içeren kelimeler
                sq = seqs[i]
                f = freqs[i]
                for j in range(len(sq) - 1):       # eski çiftleri istatistikten düş
                    p = (sq[j], sq[j + 1])
                    pairs[p] -= f
                    where[p].discard(i)
                    touched.add(p)
                ns, j, L = [], 0, len(sq)           # çifti birleştir
                while j < L:
                    if j < L - 1 and sq[j] == a and sq[j + 1] == b:
                        ns.append(ab)
                        j += 2
                    else:
                        ns.append(sq[j])
                        j += 1
                seqs[i] = ns
                for j in range(len(ns) - 1):       # yeni çiftleri ekle
                    p = (ns[j], ns[j + 1])
                    pairs[p] += f
                    where[p].add(i)
                    touched.add(p)
            pairs.pop(best, None)
            where.pop(best, None)
            touched.discard(best)
            for p in touched:                      # değişen çiftleri heap'e güncel it
                if pairs.get(p, 0) <= 0:
                    pairs.pop(p, None)
                else:
                    _push(p)
            if verbose and (step % 1000 == 0 or step == target - 1):
                print(f"   merge {step+1}/{target}  son: {a!r}+{b!r} ({cnt})")

        self.merges = {m: i for i, m in enumerate(merges)}
        allsyms = set(base)
        for sq in seqs:
            allsyms.update(sq)
        syms = list(SPECIALS) + sorted(s for s in allsyms if s not in SPECIALS)
        self.vocab = {s: i for i, s in enumerate(syms)}
        self.id2sym = {i: s for s, i in self.vocab.items()}
        self._cache = {}

    # encode-cache üst sınırı: sınırsız cache 224M satırlık korpusta milyonlarca benzersiz
    # Türkçe kelime formu biriktirir -> paralel tokenize'da işçi başına GB'larca RAM -> OOM.
    # 1M girdi (~200MB) Zipf gereği metnin ~%99'unu zaten kapsar; dolunca temizle (nadir,
    # ucuz — sık kelimeler anında yeniden dolar). Hız etkisi ihmal edilebilir.
    _CACHE_MAX = 1_000_000

    def _encode_word(self, w):
        c = self._cache.get(w)
        if c is not None:
            return c
        seq = list(w) + [END]
        while len(seq) > 1:
            best_rank = None
            best_i = None
            for j in range(len(seq) - 1):
                r = self.merges.get((seq[j], seq[j + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = j
            if best_i is None:
                break
            seq[best_i:best_i + 2] = [seq[best_i] + seq[best_i + 1]]
        unk = self.vocab.get("<unk>", 3)
        ids = [self.vocab.get(s, unk) for s in seq]
        if len(self._cache) >= self._CACHE_MAX:    # RAM sınırı: dolunca sıfırla
            self._cache.clear()
        self._cache[w] = ids
        return ids

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.vocab["<bos>"])
        for w in pre_tokenize(text):
            ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.vocab["<eos>"])
        return ids

    def tok(self, name):
        """Özel/rol token id'si (ör. tok('<user>'))."""
        return self.vocab.get(name)

    def decode(self, ids):
        out = []
        for i in ids:
            s = self.id2sym.get(int(i), "")
            if s in SPECIALS:
                continue
            if s.endswith(END):
                out.append(s[: -len(END)] + " ")
            else:
                out.append(s)
        text = re.sub(r"\s+([.,!?;:])", r"\1", "".join(out).strip())
        # Kesme işareti Türkçede bitişik ("ankara ' dır" -> "ankara'dır"). Alıntı-tırnak
        # kullanımı korpusta nadir; baskın kullanım özel-isim eki. SADECE görüntüleme —
        # token id'leri/vocab değişmez (resume güvenli).
        return re.sub(r"\s*(['’])\s*", r"\1", text)

    @property
    def size(self):
        return len(self.vocab)

    def save(self, path=BPE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": ["%s\t%s" % (a, b) for (a, b) in self.merges],
            }, f, ensure_ascii=False)

    def load(self, path=BPE_PATH):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.vocab = d["vocab"]
        self.id2sym = {i: s for s, i in self.vocab.items()}
        self.merges = {}
        for rank, line in enumerate(d["merges"]):
            a, b = line.split("\t")
            self.merges[(a, b)] = rank
        self._cache = {}
        return self

# VERİ -> kelime sıklıkları (data/ klasöründeki jsonl'lerden)
def collect_word_freq(dirs, max_lines=400000):
    wf = Counter()
    n = 0
    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
            with open(path, encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    texts = []
                    if "messages" in obj:
                        texts = [m.get("content", "") for m in obj["messages"]]
                    elif "text" in obj:
                        texts = [obj["text"]]
                    for t in texts:
                        for w in pre_tokenize(t):
                            wf[w] += 1
                    n += 1
                    if n >= max_lines:
                        return wf
    return wf

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    dirs = [os.path.join(ROOT, "data", "fetched"),
            os.path.join(ROOT, "data", "generated"),
            os.path.join(ROOT, "data")]
    if cmd == "train":
        vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
        print(f"[BPE] kelime sıklıkları toplanıyor...")
        wf = collect_word_freq(dirs)
        print(f"[BPE] benzersiz kelime: {len(wf)} | hedef sözlük: {vocab_size}")
        bpe = BPE()
        bpe.train(wf, vocab_size=vocab_size)
        bpe.save()
        print(f"[BPE] sözlük boyutu: {bpe.size} -> {BPE_PATH}")
        for s in ["Merhaba nasılsın bugün?", "Geliyorum birazdan görüşürüz."]:
            ids = bpe.encode(s)
            print(f"   '{s}' -> {len(ids)} token -> {bpe.decode(ids)!r}")
    else:
        # küçük demo eğitim (data/generated üstünde)
        wf = collect_word_freq(dirs, max_lines=8000)
        if not wf:
            print("Veri yok. Önce data/generated veya fetch_and_clean çalıştır.")
            return
        bpe = BPE()
        bpe.train(wf, vocab_size=2000, verbose=False)
        print(f"[BPE test] sözlük: {bpe.size}")
        for s in ["Merhaba nasılsın bugün?", "Geliyorum birazdan.", "Çok teşekkür ederim dostum."]:
            ids = bpe.encode(s, add_bos=True, add_eos=True)
            dec = bpe.decode(ids)
            print(f"   {len(s)} kar -> {len(ids)} token | decode: {dec!r}")


if __name__ == "__main__":
    main()
