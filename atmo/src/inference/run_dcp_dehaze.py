"""
Batch DCP dehazing script (Modified to process only the first 50 images).

Usage example:

python -m src.inference.run_dcp_dehaze \
    --input_dir ./data/Phygital2025_Atmo/real_eval/input \
    --output_dir ./outputs/dcp_real_eval \
    --save_t
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ..classical.dcp_dehaze import DCPDehazer, DCPConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DCP dehazing (first 50 images only)")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing hazy images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save dehazed results.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png,jpg,jpeg",
        help="Comma-separated list of allowed image extensions.",
    )
    parser.add_argument(
        "--save_t",
        action="store_true",
        help="Whether to save transmission maps as grayscale images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple("." + e.lower() for e in args.ext.split(","))

    cfg = DCPConfig()
    dehazer = DCPDehazer(cfg)

    # --- Load all images ---
    img_paths = [p for p in input_dir.glob("*") if p.suffix.lower() in exts]
    img_paths.sort()

    if not img_paths:
        print(f"[WARN] No images found in {input_dir} with extensions {exts}")
        return

    # --- Only keep the first 50 images ---
    img_paths = img_paths[:50]

    print(f"[INFO] Processing first {len(img_paths)} images from {input_dir} ...")

    for i, img_path in enumerate(img_paths, start=1):
        print(f"[{i}/{len(img_paths)}] Processing {img_path.name} ...")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [WARN] Failed to read image: {img_path}")
            continue

        try:
            J_bgr, t, A = dehazer.dehaze(bgr)
        except Exception as e:
            print(f"  [ERROR] Dehazing failed for {img_path.name}: {e}")
            continue

        out_name = img_path.stem + "_dcp.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), J_bgr)

        # Optionally save transmission map
        if args.save_t:
            import numpy as np
            t_vis = (t * 255.0).astype(np.uint8)
            t_vis = cv2.applyColorMap(t_vis, cv2.COLORMAP_JET)
            t_name = img_path.stem + "_t.png"
            t_path = output_dir / t_name
            cv2.imwrite(str(t_path), t_vis)

    print(f"[INFO] Finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
