import os
import random
import shutil
from pathlib import Path

random.seed(42)

# 1) 修改这里，指向你下载的 SOTS outdoor 目录
SOTS_OUTDOOR_ROOT = Path(".")  # ★★★ 改成你自己的路径
HAZY_DIR = SOTS_OUTDOOR_ROOT / "hazy"
GT_DIR = SOTS_OUTDOOR_ROOT / "clear"

OUT_ROOT = Path("SOTS_outdoor_pairs")
TRAIN_GT = OUT_ROOT / "train_gt"
TRAIN_LQ = OUT_ROOT / "train_lq"
VAL_GT = OUT_ROOT / "val_gt"
VAL_LQ = OUT_ROOT / "val_lq"

for d in [TRAIN_GT, TRAIN_LQ, VAL_GT, VAL_LQ]:
    d.mkdir(parents=True, exist_ok=True)

# 2) 根据 GT 去找对应 hazy，规则可按自己数据情况微调
gt_files = sorted([p for p in GT_DIR.iterdir()
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

pairs = []

for gt_path in gt_files:
    base = gt_path.stem  # 如 "1_1"
    # 一般 hazy 命名里包含 base，例如 "1_1_0.8_0.2"
    hazy_candidates = [p for p in HAZY_DIR.iterdir()
                       if p.stem.startswith(base)
                       and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if not hazy_candidates:
        print(f"[WARN] no hazy found for GT: {gt_path.name}")
        continue

    # 如果每个 GT 有多个 haze，可以随机选一个；也可以全用（会重复 GT）
    hazy = random.choice(hazy_candidates)
    pairs.append((hazy, gt_path))

print(f"total pairs: {len(pairs)}")

# 3) 划分 train/val（如 90% / 10%）
random.shuffle(pairs)
val_ratio = 0.1
val_size = max(1, int(len(pairs) * val_ratio))

val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]

def copy_pairs(pairs, lq_dir, gt_dir):
    for idx, (hazy, gt) in enumerate(pairs):
        new_name = f"{idx:06d}.png"  # 统一命名 000000.png
        shutil.copy(gt, gt_dir / new_name)
        shutil.copy(hazy, lq_dir / new_name)

copy_pairs(train_pairs, TRAIN_LQ, TRAIN_GT)
copy_pairs(val_pairs, VAL_LQ, VAL_GT)

print(f"train: {len(train_pairs)}, val: {len(val_pairs)}")
print("done.")
