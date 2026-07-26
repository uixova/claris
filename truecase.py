# -*- coding: utf-8 -*-
"""
TRUECASE — özel isimlerin büyük harfini KORPUSTAN öğrenir (elle liste yok).

Sorun: BPE her şeyi küçük harfe indirir (sözlük SABİT, yeniden eğitilemez) -> model
fiziksel olarak büyük harf üretemez; "istanbul osmanlı ahmet" yazar. Elle sözlük
(claris_chat._PROPER) ölçeklenmez — on binlerce özel isim var.

Çözüm (klasik truecasing): çekilen veri ORİJİNAL harfleriyle durur. Bir kelime
korpusta CÜMLE ORTASINDA ağırlıklı büyük yazılıyorsa (İstanbul, Ahmet, TBMM) özel
isimdir; ağırlıklı küçükse (masa, gitmek) değildir. Cümle BAŞI sayılmaz (orada her
kelime büyük -> sinyal kirli). Böylece "mantık" veriden kendiliğinden öğrenilir,
veri büyüdükçe sözlük de büyür.

Kullanım:
  python truecase.py build            # data/ -> models/truecase.json (bir kez / veri büyüyünce)
  python truecase.py test "istanbul'da ahmet ile buluştuk"

claris_chat.py çıktıda otomatik uygular (models/truecase.json varsa).
Not: bu ÇIKARIM katmanı düzeltmesidir; token/eğitim/resume'a dokunmaz. Modelin
ağırlık içinde büyük harf bilmesi ancak byte-level BPE ile olur (vocab değişir =
sıfırdan eğitim) -> gelecek major sürüm işi (bkz. claris.md tokenizer kısıtı).
"""

import glob
import json
import os
import re
import sys
from collections import Counter

from bpe import normalize as _tr_lower   # Türkçe küçültme BPE ile AYNI (İ->i, I->ı)

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
TC_PATH = os.path.join(ROOT, "models", "truecase.json")
DATA_DIRS = [os.path.join(ROOT, "data", "fetched"),
             os.path.join(ROOT, "data", "books"),
             os.path.join(ROOT, "data", "sft")]

# Kelime (orijinal case, Türkçe harfler dahil). Rakamlı tokenlar isim değil -> dışarıda.
_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")
# Bu karakterlerden sonra gelen kelime CÜMLE BAŞI sayılır (istatistiğe girmez).
_SENT_END = set(".!?…:;\n\"'«»()-–—•*")

MIN_COUNT = 4        # cümle-ortası en az bu kadar görülmeli (tek görüş = gürültü)
MIN_RATIO = 0.75     # baskın biçim oranı (%75+ büyükse özel isim say)
MAX_LINES = 2_000_000  # vars. tarama sınırı (RAM/süre); --limit 0 = tümü


def _iter_texts(max_lines):
    """data/ altındaki jsonl'lerden metin akışı ({"text"} + {"messages"})."""
    n = 0
    for d in DATA_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(path, encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if "text" in obj:
                        yield obj["text"]
                    elif "messages" in obj:
                        for m in obj["messages"]:
                            c = m.get("content")
                            if c:
                                yield c
                    n += 1
                    if max_lines and n >= max_lines:
                        return


def build(max_lines=MAX_LINES):
    """Korpusu tara -> kelime başına cümle-ortası biçim sayımı -> truecase sözlüğü."""
    stats = {}                     # lower_kelime -> Counter(orijinal_biçim)
    texts = 0
    for text in _iter_texts(max_lines):
        texts += 1
        prev_end = len(text)       # ilk kelime = cümle başı gibi davran (aşağıda kontrol)
        for m in _WORD_RE.finditer(text):
            w = m.group(0)
            if len(w) < 2:
                continue
            # cümle başı mı? kelimeden önceki İLK boşluk-olmayan karaktere bak
            j = m.start() - 1
            while j >= 0 and text[j] == " ":
                j -= 1
            if j < 0 or text[j] in _SENT_END:
                continue           # cümle başı -> herkes büyük, sinyal yok
            key = _tr_lower(w)
            c = stats.get(key)
            if c is None:
                c = stats[key] = Counter()
            c[w] += 1
        if texts % 200_000 == 0:
            print(f"  ... {texts:,} satır tarandı ({len(stats):,} benzersiz kelime)")

    table = {}
    for key, forms in stats.items():
        total = sum(forms.values())
        if total < MIN_COUNT:
            continue
        form, cnt = forms.most_common(1)[0]
        # baskın biçim küçük halinin AYNISI ise normal kelime -> tabloya girmez
        if form == key or cnt / total < MIN_RATIO:
            continue
        table[key] = form
    os.makedirs(os.path.dirname(TC_PATH), exist_ok=True)
    with open(TC_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    print(f"[truecase] {texts:,} satır -> {len(table):,} özel isim/kısaltma -> {TC_PATH}")
    return table


class TrueCaser:
    """models/truecase.json'u yükler, küçük-harf model çıktısındaki özel isimleri
    korpustan öğrenilmiş biçimine çevirir. Dosya yoksa fallback tabloyla çalışır."""

    _APPLY_RE = re.compile(r"[a-zçğıöşüâîû]+")   # model çıktısı hep küçük harf

    def __init__(self, path=TC_PATH, fallback=None):
        self.table = dict(fallback or {})
        try:
            with open(path, encoding="utf-8") as f:
                self.table.update(json.load(f))
        except Exception:
            pass                    # build edilmemiş -> fallback yeter, sohbet bozulmaz

    def fix(self, text):
        # "istanbul'da" -> regex "istanbul" + "da" ayrı yakalar; ek KENDİLİĞİNDEN
        # küçük kalır (Türkçe imla: İstanbul'da). Eşleşme yoksa kelime aynen kalır.
        return self._APPLY_RE.sub(lambda m: self.table.get(m.group(0), m.group(0)), text)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        limit = MAX_LINES
        if "--limit" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        build(limit)
    elif cmd == "test":
        tc = TrueCaser()
        text = " ".join(sys.argv[2:]) or "istanbul'da ahmet ile osmanlı tarihini konuştuk"
        print(f"[truecase] {len(tc.table):,} girdili tablo")
        print("  girdi :", text)
        print("  çıktı :", tc.fix(text))
    else:
        print("kullanım: python truecase.py build [--limit N] | test [metin]")


if __name__ == "__main__":
    main()
