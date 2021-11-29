# Identity-aware Fingerprint Database (IAF) models

## Download: https://drive.google.com/drive/folders/1ZppbRqviDI_6HSFVCrUmGZOxjA-jQCCu?usp=sharing

Dependences: 
```bash
$ pip install lightweight-gan==0.20.4 torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html 
```

Inference example: 
```bash
$ CUDA_VISIBLE_DEVICES=0 lightweight_gan --name LOOP_RIGHT --load-from 138 --generate --generate-types default --num-image-tiles 1000
```
