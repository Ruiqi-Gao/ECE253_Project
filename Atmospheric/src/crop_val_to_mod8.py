from pathlib import Path
import cv2

# 这里改成你自己的 val 目录
ROOT = Path("Dehazing/Datasets/SOTS_Outdoor/val")

for split in ["gt", "hazy"]:
    folder = ROOT / split
    out_folder = ROOT / f"{split}_mod8"
    out_folder.mkdir(parents=True, exist_ok=True)

    for p in sorted(folder.glob("*.*")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("skip (failed to read):", p)
            continue

        h, w = img.shape[:2]
        new_h = h - (h % 8)   # 向下取最近的 8 的倍数
        new_w = w - (w % 8)

        cropped = img[:new_h, :new_w]
        out_path = out_folder / p.name
        cv2.imwrite(str(out_path), cropped)

        print(f"{p.name}: ({h},{w}) -> ({new_h},{new_w})")
