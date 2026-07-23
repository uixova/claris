# -*- coding: utf-8 -*-
"""
RAG İSKELETİ (calisra.md §9.5 — henüz bağlı değil): model bilmediği bir şey
sorulunca Python araya girer, dışarıdan (web araması / yerel PDF-TXT-JSON)
ilgili parçayı çeker, bağlama enjekte eder. Model ezberlemez; önündeki metni
Türkçe akıcılığıyla özetler/cevaplar.

Enjeksiyon sözleşmesi (calisra_chat.generate ile):
  ids = [<bos>, <sys>] + encode("Aşağıdaki bilgilere göre cevapla: " + parçalar)
        + [<user>] + encode(soru) + [<bot>]

Kısıt: 2048 bağlam -> uzun belge için chunk + retrieval (anahtar kelime ya da
embedding ile en ilgili parçayı seç). RoPE ile bağlam uzatma kapasiteyi açar.
"""


class Rag:
    """Belge/web getirici. retrieve() sorguya en ilgili k metin parçasını döner.
    Şimdilik iskelet: ilk sürümde anahtar-kelime skoru (BM25-vari), sonra embedding."""

    def __init__(self, sources=None):
        self.sources = sources or []         # dosya yolları / URL listesi

    def add_source(self, path_or_url):
        """Kaynak ekle (yerel dosya yolu ya da URL)."""
        self.sources.append(path_or_url)

    def retrieve(self, query, k=3):
        """Sorguya en ilgili k parçayı (list[str]) döndürür. Henüz gerçeklenmedi:
        chunk'lama (≤~400 token) + anahtar-kelime eşleme planlanıyor; web tarafı
        DuckDuckGo ilk-N sonuç çekme olarak ayrı adım (yol haritası §9.5)."""
        raise NotImplementedError("RAG retrieval henüz gerçeklenmedi (yol haritası §9.5)")
