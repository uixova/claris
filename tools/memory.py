# -*- coding: utf-8 -*-
"""
Memory Buffer İSKELETİ (calisra.md §9.5 — henüz bağlı değil): 2048 bağlam sınırını
"Summary Memory + Sliding Window" ile aşmak için arayüz sözleşmesi.

Plan:
  [Eski sohbet (JSON'da kayıtlı)]
     -> Python arkada geçmişi 1-2 cümleyle ÖZETLER (model ya da kural tabanlı)
  [Özet: "Kullanıcı adı Ahmet, bir kedi besliyor."]
     -> her yeni soruda <sys> tokenıyla context'in BAŞINA enjekte edilir
  [Yeni girdi (fiziksel 2048 token TEMİZ kalır)]

Enjeksiyon sözleşmesi (calisra_chat.generate ile):
  ids = [<bos>, <sys>] + encode(memory.summary()) + [<user>] + encode(soru) + [<bot>]
Model fiziksel 2048 ile çalışmaya devam eder (VRAM sabit) ama özet sayesinde
"asla unutmayan" gibi davranır. SFT'den SONRA bağlanacak (özet formatını model
tutarlı kullansın diye SFT verisiyle öğretmek gerekir).
"""


class Memory:
    """Sohbet hafızası. add() her turu kaydeder, summary() enjekte edilecek
    kısa özeti döner. Şimdilik iskelet: alt sınıf/ileriki sürüm doldurur."""

    def __init__(self, max_turns=50):
        self.turns = []                      # [(user, bot), ...]
        self.max_turns = max_turns

    def add(self, user, bot):
        """Bir sohbet turunu kaydet (kayan pencere: en eski turlar düşer)."""
        self.turns.append((user, bot))
        self.turns = self.turns[-self.max_turns:]

    def summary(self):
        """<sys> ile enjekte edilecek 1-2 cümlelik Türkçe özet döndürür.
        Henüz gerçeklenmedi: kural-tabanlı (isim/tercih yakalama) ya da modelin
        kendisiyle özetleme (ayrı kısa generate çağrısı) planlanıyor."""
        raise NotImplementedError("memory özeti henüz gerçeklenmedi (yol haritası §9.5)")
