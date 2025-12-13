import os
import sys
import glob
import argparse
import csv

import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

# skimage 用于 SSIM（更稳妥）
from skimage.metrics import structural_similarity as ssim_fn

# 让 Python 能找到当前仓库里的 basicsr
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from basicsr.models.archs.restormer_arch import Restormer


def load_opt(opt_path):
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)
    return opt


def build_model(opt, ckpt_path, device):
    """复用 test_dehaze_custom.py：从 yml 的 network_g 直接构建 Restormer。"""
    net_opt = opt["network_g"].copy()
    net_opt.pop("type", None)

    model = Restormer(**net_opt)

    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def pad_img(img, factor=8):
    """padding 到 factor 的倍数"""
    h, w = img.shape[:2]
    H = int(np.ceil(h / factor) * factor)
    W = int(np.ceil(w / factor) * factor)
    pad_h = H - h
    pad_w = W - w
    img_pad = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101
    )
    return img_pad, h, w


def maybe_resize(img_rgb, max_side):
    """
    如果 max_side > 0 且图像最长边大于 max_side，则按比例缩放。
    返回 (resized_img, scale)。
    """
    h, w = img_rgb.shape[:2]
    if max_side is None or max_side <= 0:
        return img_rgb, 1.0

    long_side = max(h, w)
    if long_side <= max_side:
        return img_rgb, 1.0

    scale = max_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_small, scale


def calc_psnr_uint8(pred_rgb_u8, gt_rgb_u8, eps=1e-10):
    """PSNR on uint8 RGB images."""
    pred = pred_rgb_u8.astype(np.float32)
    gt = gt_rgb_u8.astype(np.float32)
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return 99.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calc_ssim_uint8(pred_rgb_u8, gt_rgb_u8):
    """SSIM on uint8 RGB images, computed on RGB channels (average)."""
    # skimage SSIM expects channel_axis for color images
    return ssim_fn(
        gt_rgb_u8,
        pred_rgb_u8,
        channel_axis=2,
        data_range=255
    )


def build_gt_map(gt_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    gt_paths = [p for p in glob.glob(os.path.join(gt_dir, "*")) if os.path.splitext(p)[1].lower() in exts]
    return {os.path.basename(p): p for p in gt_paths}


def inference_and_eval(
    input_dir,
    gt_dir,
    output_dir,
    ckpt,
    opt_path,
    gpu_id=0,
    max_side=0,              # 评测建议默认不缩放
    save_csv=None
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    opt = load_opt(opt_path)
    model = build_model(opt, ckpt, device)

    gt_map = build_gt_map(gt_dir)
    if len(gt_map) == 0:
        raise RuntimeError(f"No GT images found in: {gt_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if os.path.splitext(p)[1].lower() in exts
    ])
    print(f"Found {len(img_paths)} hazy images in {input_dir}")
    print(f"Found {len(gt_map)} gt images in {gt_dir}")

    if max_side is not None and max_side > 0:
        print(f"[Warn] Evaluation with downscaling enabled (max_side={max_side}). "
              f"Metrics will be computed at (possibly) resized resolution.")

    amp_enable = device.type == "cuda"

    rows = []
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Dehazing+Eval"):
            imgname = os.path.basename(img_path)

            if imgname not in gt_map:
                print(f"[Skip] No matched GT for {imgname}")
                continue

            # ---- read hazy ----
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[Skip] Failed to read hazy: {img_path}")
                continue
            hazy_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = hazy_rgb.shape[:2]

            # ---- read gt ----
            gt_bgr = cv2.imread(gt_map[imgname], cv2.IMREAD_COLOR)
            if gt_bgr is None:
                print(f"[Skip] Failed to read gt: {gt_map[imgname]}")
                continue
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

            # 保证同尺寸（若你的 gt/hazy 本来就同名同尺寸，这里不会改变）
            if gt_rgb.shape[:2] != (h_orig, w_orig):
                # 优先按 hazy 尺寸对齐 gt（最常见：gt 与 hazy 同尺寸）
                gt_rgb = cv2.resize(gt_rgb, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

            # ---- optional resize (for memory) ----
            hazy_proc, scale = maybe_resize(hazy_rgb, max_side=max_side)
            h_proc, w_proc = hazy_proc.shape[:2]

            # gt 同步 resize，保证计算同尺度指标
            gt_proc = gt_rgb
            if scale != 1.0:
                gt_proc = cv2.resize(gt_rgb, (w_proc, h_proc), interpolation=cv2.INTER_CUBIC)

            hazy_f = hazy_proc.astype(np.float32) / 255.0
            hazy_pad, h_pad, w_pad = pad_img(hazy_f, factor=8)

            img_t = torch.from_numpy(hazy_pad).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=amp_enable):
                out = model(img_t)
            if isinstance(out, (list, tuple)):
                out = out[0]

            out = out[:, :, :h_pad, :w_pad]

            out_np = (
                out.clamp(0, 1)
                .cpu()
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )

            out_u8 = (out_np * 255.0 + 0.5).astype(np.uint8)

            # 保存输出（与 hazy 同名）
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, imgname), out_bgr)

            # 计算指标（与 gt_proc 对齐）
            psnr = calc_psnr_uint8(out_u8, gt_proc.astype(np.uint8))
            ssim = calc_ssim_uint8(out_u8, gt_proc.astype(np.uint8))

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            rows.append([imgname, psnr, ssim])

    if len(psnr_list) == 0:
        raise RuntimeError("No valid paired images were evaluated. "
                           "Check that hazy/gt filenames match exactly.")

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    print("========================================")
    print(f"Evaluated pairs: {len(psnr_list)}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print("========================================")

    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True) if os.path.dirname(save_csv) else None
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "psnr", "ssim"])
            writer.writerows(rows)
            writer.writerow(["__AVERAGE__", avg_psnr, avg_ssim])
        print(f"[Saved] CSV metrics to: {save_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Dehazing inference + evaluation (PSNR/SSIM) for paired datasets"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with hazy images (LQ)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder with gt images (GT), same filenames as hazy")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save dehazed images")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--opt", type=str, required=True, help="Path to training yml (network_g config)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument("--max_side", type=int, default=0, help="Max longer side during eval; <=0 disables downscaling")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save per-image metrics csv")
    args = parser.parse_args()

    inference_and_eval(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        ckpt=args.ckpt,
        opt_path=args.opt,
        gpu_id=args.gpu,
        max_side=args.max_side,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
