# -*- coding: utf-8 -*-
"""
Claris BitLinear — BitNet b1.58'in çekirdeği. nn.Linear'ın YERİNE geçen katman:
ağırlık forward'da ternary {-1,0,+1}'e, aktivasyon int8'e kuantize edilir; gradyan
Straight-Through Estimator (STE) ile fp16 "gölge" ağırlığa akar.

Kaynak: BitNet b1.58 2B4T teknik raporu (arxiv 2504.12285) + kevbuh/bitnet.
- Ağırlık: absmean ölçekle -> yuvarla -> {-1,0,+1}. Dequant için ölçeğe böl.
- Aktivasyon: per-token absmax -> int8 (-128..127). Dequant için ölçeğe böl.
- SubLN: BitLinear'ın İÇİNDE RMSNorm (kuantizasyondan ÖNCE) -> kararlılık.
- bias YOK (binarizasyon regularizasyon gibi davranır).

⚠️ 1.58-bit HIZI sadece ÇIKARIMDA (ternary paketli + toplama). EĞİTİMDE fp16 gölge
ağırlık + quant yükü var -> Calisra fp16'dan adım başına biraz daha yavaş.

Test:  python bitlinear.py test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


RMS_EPS = 1e-6
Q_EPS = 1e-5


class RMSNorm(nn.Module):
    """Calisra ile BİREBİR: ortalama çıkarmaz, RMS'e böler, fp32'de normalize."""
    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dt) * self.weight


def weight_quant(w):
    """fp16 ağırlık -> ternary {-1,0,+1} (dequant ölçeğiyle). absmean scaling.
    gamma = mean(|w|); w/gamma yuvarla+clip {-1,0,1}; sonra gamma ile ölçekle (dequant)."""
    scale = 1.0 / w.abs().mean().clamp(min=Q_EPS)
    return (w * scale).round().clamp_(-1, 1) / scale


def act_quant(x):
    """Aktivasyon -> per-token int8 (dequant ölçeğiyle). absmax scaling (son eksen)."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=Q_EPS)
    return (x * scale).round().clamp_(-128, 127) / scale


class BitLinear(nn.Linear):
    """nn.Linear drop-in. bias=False. SubLN (RMSNorm) İÇERİDE. STE ile gradyan geçer.
    Eğitimde self.weight = fp16 gölge; forward'da ternary'e kuantize edilir."""
    def __init__(self, in_features, out_features, bias=False, norm_eps=RMS_EPS):
        super().__init__(in_features, out_features, bias=False)   # bias YOK (BitNet)
        self.norm = RMSNorm(in_features, eps=norm_eps)            # SubLN

    def forward(self, x):
        x = self.norm(x)                                  # SubLN (kuantizasyondan önce)
        # STE: forward'da kuantize değeri kullan, backward'da gradyan DÜZ geçsin
        x = x + (act_quant(x) - x).detach()
        w = self.weight
        w = w + (weight_quant(w) - w).detach()
        return F.linear(x, w)


# --------------------------- BİRİM TEST ---------------------------
def _test():
    torch.manual_seed(0)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
        ok = ok and cond

    # 1) weight_quant çıktısı ternary mi (dequant öncesi round değerleri {-1,0,1})
    w = torch.randn(256, 128)
    scale = 1.0 / w.abs().mean().clamp(min=Q_EPS)
    ternary = (w * scale).round().clamp(-1, 1)
    uniq = set(ternary.unique().tolist())
    check("weight ternary ∈ {-1,0,1}", uniq <= {-1.0, 0.0, 1.0}, str(sorted(uniq)))

    # 2) act_quant int8 aralığında (dequant öncesi)
    x = torch.randn(4, 32, 128) * 5
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=Q_EPS)
    q = (x * s).round().clamp(-128, 127)
    check("act int8 aralığında", q.min() >= -128 and q.max() <= 127,
          f"[{int(q.min())},{int(q.max())}]")

    # 3) STE gradyan akıyor mu (kuantizasyon non-diff ama grad geçmeli)
    lin = BitLinear(128, 256)
    xin = torch.randn(2, 16, 128, requires_grad=True)
    out = lin(xin)
    out.sum().backward()
    check("STE: girdi gradyanı var", xin.grad is not None and xin.grad.abs().sum() > 0)
    check("STE: ağırlık gradyanı var", lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0)
    check("forward şekli doğru", tuple(out.shape) == (2, 16, 256), str(tuple(out.shape)))

    # 4) bias YOK (BitNet)
    check("bias yok", lin.bias is None)

    # 5) quant round-trip makul (dequant ~ orijinal büyüklük mertebesi)
    wd = weight_quant(w)
    rel = (wd - w).abs().mean() / w.abs().mean()
    check("weight quant sapması makul (<1.0)", rel.item() < 1.0, f"rel={rel.item():.3f}")

    # 6) forward fp16 autocast altında çalışıyor (T4 yolu)
    if torch.cuda.is_available():
        lin.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            o = lin(torch.randn(2, 16, 128, device="cuda"))
        check("fp16 autocast forward", o.dtype in (torch.float16, torch.float32))
    else:
        print("  (CUDA yok — fp16 autocast testi atlandı)")

    print("\n" + ("HEPSİ GEÇTİ ✓" if ok else "BAŞARISIZ ✗"))
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(0 if _test() else 1)
    print("kullanım: python bitlinear.py test")
