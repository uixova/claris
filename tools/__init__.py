# -*- coding: utf-8 -*-
"""
Calisra tool-agent katmanı (calisra.md §9.5): PARAMETRESİZ ekstra yetenekler.

Bilgiyi modelin paramında tutmak yerine gerektiğinde dışarıdan getirip modele
okutuyoruz. Bunlar ÇIKARIM eklentisi — eğitim/resume'a dokunmaz.

Kullanım (calisra_chat.py):
    from tools import run_pre_hooks
    ans = run_pre_hooks(kullanici_metni)
    if ans is not None:  # tool doğrudan cevapladı, model atlanır
        print(ans)

Sıra (yol haritası): calc (hazır) -> RAG (iskelet) -> memory (iskelet) -> web.
"""

# Üretim ÖNCESİ hook zinciri: her hook (user_text) alır, ya doğrudan cevap (str)
# ya None döner. İlk non-None cevap kazanır -> model hiç çağrılmaz (hız + doğruluk:
# 502M model aritmetiği güvenilir yapamaz, hesap makinesi yapar).
PRE_GENERATE_HOOKS = []


def run_pre_hooks(user_text):
    """Hook zincirini sırayla dener; ilk non-None cevabı döner, yoksa None.
    Hook hatası sohbeti ASLA bozmaz (sessizce sonrakine geçer)."""
    for hook in PRE_GENERATE_HOOKS:
        try:
            ans = hook(user_text)
        except Exception:
            continue
        if ans is not None:
            return ans
    return None


from .calc import hook as _calc_hook          # noqa: E402  (kayıt import'un sonunda)
PRE_GENERATE_HOOKS.append(_calc_hook)
