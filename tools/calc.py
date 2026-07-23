# -*- coding: utf-8 -*-
"""
Matematik tool'u (calisra.md §9.5): küçük model aritmetiği güvenilir yapamaz ->
hesap makinesi yapar. "22*10/2+5" gibi ifadeyi regex yakalar, GÜVENLİ ast-tabanlı
değerlendirici hesaplar (ASLA eval() değil), sonuç doğrudan cevaba döner.

Güvenlik: sadece sayı + aritmetik operatör düğümlerine izin var (Name/Call/
Attribute RED), üs ve sonuç büyüklüğü sınırlı -> 9**9**9 gibi DoS ifadeleri kesilir.
"""

import ast
import operator
import re

# İzinli ikili operatörler (ast düğüm tipi -> fonksiyon)
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_MAX_ABS = 1e15          # ara/son sonuç tavanı (taşma/DoS koruması)
_MAX_POW_EXP = 12        # üs tavanı
_MAX_POW_BASE = 1e6      # üs tabanı tavanı

# Aritmetik ifade yakalayıcı: sayıyla başlar, en az bir operatör içerir.
_EXPR_RE = re.compile(r"\d[\d\s.,]*(?:[+\-*/%()]+[\d\s.,()+\-*/%]*)+")
# Tetik kelimeleri: ifade cümle İÇİNDEYSE bunlardan biri de olmalı ("2024-2025
# sezonu" gibi masum metni ele geçirmesin). İfade tek başınaysa kelime gerekmez.
_TRIGGER_RE = re.compile(r"\b(kaç|kaçtır|hesapla|hesaplar\smısın|sonucu|eder|etmez)\b")
_ONLY_EXPR_RE = re.compile(r"^[\d\s.,()+\-*/%]+$")


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > _MAX_POW_EXP or abs(a) > _MAX_POW_BASE):
            raise ValueError("üs çok büyük")
        r = _OPS[type(node.op)](a, b)
        if abs(r) > _MAX_ABS:
            raise ValueError("sonuç çok büyük")
        return r
    raise ValueError("izin verilmeyen ifade düğümü: %s" % type(node).__name__)


def evaluate(expr):
    """Aritmetik ifadeyi güvenle hesaplar. Sadece sayı+operatör; başka her şey
    ValueError. Bölme/mod sıfır hatası vb. çağırana yükselir."""
    expr = expr.replace(",", ".")            # Türkçe ondalık virgül -> nokta
    return _ev(ast.parse(expr.strip(), mode="eval"))


def _fmt_tr(x):
    """Sonucu Türkçe biçimle: tam sayıysa düz, değilse virgüllü ondalık."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, int):
        return str(x)
    return ("%.6f" % x).rstrip("0").rstrip(".").replace(".", ",")


def hook(user_text):
    """Pre-generate hook: metinde hesaplanabilir ifade varsa doğrudan cevap döner,
    yoksa None (model devam eder). İfade cümle içindeyse tetik kelimesi arar."""
    if not user_text:
        return None
    low = user_text.lower().strip()
    m = _EXPR_RE.search(low)
    if not m:
        return None
    expr = m.group(0).strip()
    # en az 2 sayı + 1 operatör içermeli ("2024" tek sayı -> tetiklenmez)
    if len(re.findall(r"\d+(?:[.,]\d+)?", expr)) < 2 or not re.search(r"[+\-*/%]", expr):
        return None
    if not _ONLY_EXPR_RE.match(low) and not _TRIGGER_RE.search(low):
        return None                          # cümle içinde ama hesap NİYETİ yok
    try:
        result = evaluate(expr)
    except ZeroDivisionError:
        # None dönsek model uydurma cevap verirdi; hesap NİYETİ belli -> doğru bilgiyi ver.
        return "Hesap: %s -> sıfıra bölme tanımsızdır." % expr.replace(" ", "")
    except Exception:
        return None                          # bozuk ifade -> modele bırak
    return "Hesap: %s = %s" % (expr.replace(" ", ""), _fmt_tr(result))
