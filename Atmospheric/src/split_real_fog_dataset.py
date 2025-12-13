from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# ========= 路径配置，根据你实际情况改 =========
root = Path("./Dehazing/Datasets/myDataset/Real/Finetune")
hazy_dir = root / "hazy"      # 有雾图
gt_dir   = root / "gt"    # 清晰图 (GT)

out_train_hazy = root / "train/hazy"
out_train_gt   = root / "train/gt"
out_val_hazy   = root / "val/hazy"
out_val_gt     = root / "val/gt"

# 如果之前已经跑过，建议先手动确认 train/val 目录是空的，避免混淆

for p in [out_train_hazy, out_train_gt, out_val_hazy, out_val_gt]:
    p.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

pairs = []   # (hazy_path, gt_path, common_name)

for hazy in hazy_dir.glob("*"):
    if hazy.suffix.lower() not in exts:
        continue

    name = hazy.stem   # 例如 0016E5_08021_fog
    if not name.endswith("fog"):
        print(f"跳过非 hazy 文件: {hazy.name}")
        continue

    base = name[:-4]               # 去掉 'fog' -> 0016E5_08021
    gt_name = base + hazy.suffix   # 0016E5_08021.png
    gt_path = gt_dir / gt_name

    if gt_path.exists():
        pairs.append((hazy, gt_path, gt_name))
    else:
        print(f"[警告] 找不到 GT: {gt_path}, 跳过 {hazy.name}")

print(f"\n有效配对数量: {len(pairs)}")

# 8:2 划分 train / val
train_pairs, val_pairs = train_test_split(
    pairs, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

def copy_pairs(pairs, hazy_out, gt_out):
    for hazy, gt, common_name in pairs:
        # GT 直接用自己文件名
        shutil.copy2(gt, gt_out / common_name)
        # hazy 拷贝时重命名成和 GT 一样的文件名
        shutil.copy2(hazy, hazy_out / common_name)

copy_pairs(train_pairs, out_train_hazy, out_train_gt)
copy_pairs(val_pairs,   out_val_hazy,   out_val_gt)

print("\n划分完成！目录如下：")
print("Train hazy:", out_train_hazy)
print("Train gt  :", out_train_gt)
print("Val hazy  :", out_val_hazy)
print("Val gt    :", out_val_gt)
