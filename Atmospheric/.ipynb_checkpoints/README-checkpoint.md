# README — Distortion-Aware Image Restoration (Dehazing: SOTS Outdoor + Real Fog Fine-tune)

This repository contains our implementation for **road-scene dehazing** using **Restormer** within the **BasicSR** training framework. It supports:

* **Pre-training** on **RESIDE SOTS Outdoor (paired)**.
* **Fine-tuning** on a **custom real fog dataset** (paired or pseudo-paired).
* **Inference** on custom images.
* **Evaluation** with PSNR/SSIM on paired datasets.

## 1. Repository Structure

Key files and folders used in this project:

```text
Restormer/
  basicsr/                                  # BasicSR framework (from Restormer repo)
  Dehazing/
    Options/
      Dehazing_SOTS_Outdoor_Restormer.yml    # SOTS Outdoor training config
      Dehazing_SOTS_Outdoor_Restormer_big.yml# (optional) bigger Restormer config
      Finetune_RealFog_Restormer.yml         # fine-tune config on Real Fog paired/pseudo-paired
    tools/
      gen_pseudo_gt_dcp.py                   # generate pseudo-GT for real fog images (optional)
      split_real_fog_dataset.py              # split paired dataset into train/val; handle fog suffix pairing
    test_dehaze_custom.py                    # inference on custom images (save outputs)
    test_dehaze_eval.py                      # inference + evaluation (PSNR/SSIM with gt_dir)
  Dehazing/
    Datasets/                                # dataset root (see below)
  Dehazing/experiments/                      # training outputs (checkpoints/logs)
```

## 2. Environment Setup

### 2.1 Create conda env

```bash
conda create -n ECE253 python=3.8 -y
conda activate ECE253
```


### 2.2 Verify GPU

```bash
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Torch:", torch.__version__)
EOF
```
## 3. Data

### 3.1 RESIDE SOTS Outdoor (Paired Dehazing Dataset)

We use **RESIDE SOTS Outdoor** for supervised pre-training.

Expected folder layout:

```text
Dehazing/Datasets/SOTS_Outdoor/
  train/
    hazy/
    gt/
  val/
    hazy_mod8/
    gt_mod8/
```

Notes:

* `*_mod8` validation folders are **cropped/padded** so that H/W are multiples of 8 (required by Restormer).
* If your original val images are not mod-8, preprocess them with the provided crop script (or equivalent).

**Download links (if dataset not included due to size):**

* RESIDE dataset homepage / mirrors (provide your link here)
* SOTS Outdoor subset (provide your link here)

> Replace the above with the actual links you used (Google Drive / official mirror).

---

### 3.2 Custom Real Fog Dataset (Fine-tuning)

We fine-tune on a custom dataset. Two supported cases:

#### Case A — Paired real dataset (preferred)

```text
Dehazing/Datasets/myDataset/Real/
  train/
    hazy/
    gt/
  val/
    hazy/
    gt/
```

#### Case B — Unpaired real dataset (no GT)

We generate pseudo-GT using DCP:

* Input hazy: `Real/Fog/`
* Pseudo GT output: `Real/pseudo_gt_dcp/`

Then split into train/val (paired):

```text
Dehazing/Datasets/myDataset/Real/
  train/
    hazy/
    gt/
  val/
    hazy/
    gt/
```

## 4. Pretrained Models / Checkpoints

Training outputs are saved under:

```text
Dehazing/experiments/<experiment_name>/
  models/                # .pth checkpoints
  log/                   # train logs + tensorboard
  visual_validation/     # validation images (if enabled)
```

Examples:

* Pre-trained (SOTS):
  `Dehazing/experiments/Dehazing_SOTS_Outdoor_Restormer_big/models/net_g_80000.pth`
* Fine-tuned (Real Fog):
  `Dehazing/experiments/Finetune_RealFog_Restormer/models/net_g_20000.pth`

**If checkpoints are large and not committed**, provide download links:

* SOTS pre-trained checkpoint: (link here)
* Real Fog fine-tuned checkpoint: (link here)

## 5. Training

All training commands should be run from the repo root.

### 5.1 Pre-train on SOTS Outdoor

```bash
conda activate ECE253
cd /path/to/Restormer

PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt Dehazing/Options/Dehazing_SOTS_Outdoor_Restormer_big.yml
```

Outputs:

* checkpoints: `Dehazing/experiments/Dehazing_SOTS_Outdoor_Restormer_big/models/`
* logs: `Dehazing/experiments/Dehazing_SOTS_Outdoor_Restormer_big/log/`


### 5.2 (Optional) Generate Pseudo-GT for Real Fog with DCP

If real data has no GT:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
python Dehazing/tools/gen_pseudo_gt_dcp.py
```

This generates:

```text
Dehazing/Datasets/myDataset/Real/pseudo_gt_dcp/
```


### 5.3 Split Real Fog dataset into train/val (paired)

If hazy filenames include an extra suffix `fog` (e.g., `xxx_fog.png`) while gt is `xxx.png`,
use:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
python Dehazing/tools/split_real_fog_dataset.py
```

This produces:

```text
Dehazing/Datasets/myDataset/Real/train/{hazy,gt}/
Dehazing/Datasets/myDataset/Real/val/{hazy,gt}/
```


### 5.4 Fine-tune on Real Fog

Edit `Dehazing/Options/Finetune_RealFog_Restormer.yml`:

* `datasets.train.dataroot_lq / dataroot_gt`
* `datasets.val.dataroot_lq / dataroot_gt`
* `path.pretrain_network_g` → point to the SOTS pretrained checkpoint

Then run:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt Dehazing/Options/Finetune_RealFog_Restormer.yml
```

---

## 6. Inference (Dehazing on Custom Images)

Run inference and save outputs:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
python Dehazing/test_dehaze_custom.py \
  --input_dir Dehazing/Datasets/myDataset/Real/Fog \
  --output_dir Dehazing/Datasets/myDataset/Real/FogOutput \
  --ckpt Dehazing/experiments/Finetune_RealFog_Restormer/models/net_g_20000.pth \
  --opt Dehazing/Options/Finetune_RealFog_Restormer.yml \
  --gpu 0
```

---

## 7. Evaluation (PSNR / SSIM on Paired Data)

If you have paired `hazy` and `gt` folders (same filenames), run:

```bash
PYTHONPATH="./:${PYTHONPATH}" \
python test_dehaze_eval.py \
  --input_dir Dehazing/Datasets/myDataset/Real/eval/hazy \
  --gt_dir    Dehazing/Datasets/myDataset/Real/eval/gt \
  --output_dir Dehazing/Datasets/myDataset/Real/eval/dehazed_eval \
  --ckpt Finetune_RealFog_Restormer/models/net_g_20000.pth \
  --opt Dehazing/Options/Finetune_RealFog_Restormer.yml \
  --gpu 0 \
  --save_csv Dehazing/Datasets/myDataset/Real/eval/metrics_eval.csv
```

The script prints average PSNR/SSIM across the dataset and saves dehazed outputs.

---

## 8. Included Source Code (All)

Relevant source code used in our submission:

* Training framework: `basicsr/` (from Restormer repo)
* Dehazing configs: `Dehazing/Options/*.yml`
* Fine-tune utilities:

  * `Dehazing/tools/gen_pseudo_gt_dcp.py`
  * `Dehazing/tools/split_real_fog_dataset.py`
* Inference / evaluation:

  * `Dehazing/test_dehaze_custom.py`
  * `Dehazing/test_dehaze_eval.py`

---

## 9. Notes / Common Issues

* **Input size constraint**: Restormer requires image H/W divisible by 8.
  We handle this via padding in the test scripts and via `*_mod8` for validation.

* **Disk quota exceeded**: training outputs can be large (checkpoints + visualizations).
  Reduce `save_img`, increase `save_checkpoint_freq`, or delete unused `visual_validation/` and old checkpoints.

---

## 10. Reproducibility Checklist (Report Requirement)

* [x] Code to train (pre-train + fine-tune)
* [x] Code to test/infer on custom images
* [x] Code to evaluate PSNR/SSIM on paired datasets
* [x] Dataset directory layout + download links placeholder
* [x] Pretrained model paths + download links placeholder

