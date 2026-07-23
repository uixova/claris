# -*- coding: utf-8 -*-
"""
Claris → GGUF (bitnet.cpp) export — v1.0 DEPLOY yolu. ŞİMDİLİK İSKELET + yöntem notu.

BitNet'in deploy'u Calisra'nınkinden (export_hf.py → llama.cpp) FARKLI: ternary ağırlıklar
Microsoft'un **bitnet.cpp** çekirdeğiyle paketlenir (4 ternary değer → 1 int8, CPU'da
toplama-tabanlı matmul). Adımlar (v1.0'da doldurulacak):

1) claris_model.pt (fp16 gölge ağırlık) yüklenir.
2) Her BitLinear.weight -> weight_quant ile ternary'e SABİTLENİR (artık gölge değil, kalıcı).
3) HF-BitNet formatına yazılır (config.json: model_type="bitnet", squared_relu, subln flag'leri;
   safetensors ternary-packed). NOT: iç isimler zaten HF-benzeri (q/k/v/o_proj, gate/up/down_proj)
   ama SubLN'in ekstra norm'ları + ReLU² Microsoft BitNet config'iyle eşlenmeli.
4) bitnet.cpp / convert ile GGUF: `python convert-helper-bitnet.py models/claris_hf --outtype tl1`
   (bitnet.cpp'nin ternary-lookup formatı). Sonuç: ~66MB, CPU'da torch'suz.

Kaynaklar:
- bitnet.cpp: https://github.com/microsoft/BitNet
- HF BitNet formatı: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

Tokenizer kısıtı Calisra ile AYNI: kelime-BPE + Türkçe lowercase -> native GGUF tokenizer'a
birebir çevrilmez; external tokenization (bpe.py) ile çalışır (bkz. Calisra run_gguf.py).
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    print("\n[v1.0 İSKELET] Gerçek export bitnet.cpp entegrasyonuyla v1.0'da eklenecek.")
    print("Şu an: model pretraining aşamasında; deploy plato/SFT sonrası.")
    sys.exit(0)
