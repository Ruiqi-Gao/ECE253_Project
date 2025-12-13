import os
import sys
import glob
import argparse

import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

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
    """
    根据 yml 中的 network_g 配置构建 Restormer，
    不再手动写 dim / num_blocks。
    """
    net_opt = opt["network_g"].copy()
    net_opt.pop("type", None)  # 去掉 type: Restormer

    model = Restormer(**net_opt)

    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # 兼容 BasicSR/Restormer 的多种保存格式
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
    如果 max_side > 0 且图像最长边大于 max_side，
    则按比例缩放到最长边 = max_side；否则不缩放。
    返回 (resized_img, scale) 其中 scale 是缩放比例 (<1 表示缩小)。
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
    img_small = cv2.resize(
        img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
    )
    return img_small, scale


def inference_folder(input_dir, output_dir, ckpt, opt_path, gpu_id=0, max_side=720):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print("Use device:", device)

    opt = load_opt(opt_path)
    model = build_model(opt, ckpt, device)

    img_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
    print(f"Found {len(img_paths)} images in {input_dir}")
    if max_side is not None and max_side > 0:
        print(f"[Info] Will downscale images whose longer side > {max_side} px.")

    amp_enable = device.type == "cuda"

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Dehazing"):
            imgname = os.path.basename(img_path)

            # 1) 读图 (BGR) -> RGB，float32, [0,1]
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Warning: failed to read {img_path}")
                continue
            rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = rgb_orig.shape[:2]

            # 2) 如果图太大，先整体缩小（避免 OOM）
            rgb_proc, scale = maybe_resize(rgb_orig, max_side=max_side)
            h_proc, w_proc = rgb_proc.shape[:2]

            rgb_proc = rgb_proc.astype(np.float32) / 255.0  # HWC, [0,1]

            # 3) padding 到 8 的倍数
            rgb_pad, h_pad, w_pad = pad_img(rgb_proc, factor=8)

            # 4) HWC -> NCHW tensor
            img_t = (
                torch.from_numpy(rgb_pad)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            # 5) forward (AMP 可减少显存占用)
            with torch.cuda.amp.autocast(enabled=amp_enable):
                out = model(img_t)
            if isinstance(out, (list, tuple)):
                out = out[0]

            # 6) 去 padding
            out = out[:, :, :h_pad, :w_pad]

            # 7) tensor -> HWC RGB [0,1]
            out = (
                out.clamp(0, 1)
                .cpu()
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )

            # 8) 如果之前缩小过，这里再插值回原尺寸
            if scale != 1.0:
                out = cv2.resize(
                    out,
                    (w_orig, h_orig),
                    interpolation=cv2.INTER_CUBIC,
                )

            # 9) [0,1] -> uint8，RGB -> BGR 保存
            out = (out * 255.0 + 0.5).astype(np.uint8)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_dir, imgname)
            cv2.imwrite(save_path, out_bgr)

    print(f"Done! Results are saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Custom dehazing with Restormer (SOTS Outdoor model)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder with hazy images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save dehazed images",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pth",
    )
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        help="Path to Dehazing_SOTS_Outdoor_Restormer.yml",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id (default: 0)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=720,
        help="Max length of image longer side during inference "
             "(set <=0 to disable downscaling)",
    )
    args = parser.parse_args()

    inference_folder(
        args.input_dir,
        args.output_dir,
        args.ckpt,
        args.opt,
        gpu_id=args.gpu,
        max_side=args.max_side,
    )


if __name__ == "__main__":
    main()
