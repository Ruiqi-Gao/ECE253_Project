"""
DCP-based dehazing implementation.

Usage example (inside other scripts):

from classical.dcp_dehaze import DCPDehazer
import cv2

dehazer = DCPDehazer()
img = cv2.imread("hazy.png")  # BGR uint8
J, t, A = dehazer.dehaze(img)
cv2.imwrite("hazy_dehazed.png", J)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class DCPConfig:
    """Configuration parameters for Dark Channel Prior dehazing."""
    patch_size: int = 15          # window size for min filter
    omega: float = 0.95           # amount of haze removal, [0, 1]
    t_min: float = 0.1            # lower bound for transmission
    top_percent: float = 0.001    # top p% brightest in dark channel to estimate A
    guided_radius: int = 40       # radius for guided filter
    guided_eps: float = 1e-3      # epsilon for guided filter
    use_guided_filter: bool = True


class DCPDehazer:
    """
    Dark Channel Prior (DCP) dehazing.

    Reference:
        K. He, J. Sun, X. Tang,
        "Single Image Haze Removal Using Dark Channel Prior", CVPR 2010.
    """

    def __init__(self, config: DCPConfig | None = None) -> None:
        self.cfg = config or DCPConfig()

    # ---------- Public API ---------- #

    def dehaze(
        self,
        bgr_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply DCP dehazing to an input BGR image.

        Args:
            bgr_img: HxWx3 uint8 BGR image (OpenCV format).

        Returns:
            J: dehazed BGR image (uint8)
            t_refined: refined transmission map (float32 in [0,1])
            A: estimated atmospheric light, shape (3,) in [0,1] RGB order
        """
        if bgr_img is None or bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
            raise ValueError("Input must be an HxWx3 BGR image")

        # Convert to float [0,1], and BGR->RGB for consistency with paper
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        dark_channel = self._get_dark_channel(rgb, self.cfg.patch_size)
        A = self._estimate_atmospheric_light(rgb, dark_channel, self.cfg.top_percent)
        t = self._estimate_transmission(rgb, A, self.cfg.omega, self.cfg.patch_size)

        if self.cfg.use_guided_filter:
            t_refined = self._guided_filter(rgb, t, self.cfg.guided_radius, self.cfg.guided_eps)
        else:
            t_refined = t

        t_refined = np.clip(t_refined, self.cfg.t_min, 1.0)
        J = self._recover_scene_radiance(rgb, A, t_refined)

        # Convert back to BGR uint8 for OpenCV-friendly output
        J_bgr = cv2.cvtColor((J * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return J_bgr, t_refined.astype(np.float32), A

    # ---------- Core steps ---------- #

    @staticmethod
    def _get_dark_channel(rgb: np.ndarray, patch_size: int) -> np.ndarray:
        """
        Compute dark channel of RGB image.

        dark(x) = min_{y in patch(x)} ( min_c I_c(y) )
        """
        # min over color channels
        min_per_channel = np.min(rgb, axis=2)
        # min filter via erode
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark = cv2.erode(min_per_channel, kernel)
        return dark

    @staticmethod
    def _estimate_atmospheric_light(
        rgb: np.ndarray,
        dark_channel: np.ndarray,
        top_percent: float,
    ) -> np.ndarray:
        """
        Estimate atmospheric light A from the brightest pixels in the dark channel.

        Args:
            rgb: HxWx3 RGB image in [0,1].
            dark_channel: HxW dark channel.
            top_percent: fraction of pixels to consider (e.g., 0.001 for top 0.1%).
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        num_top = max(1, int(num_pixels * top_percent))

        # Flatten and get indices of largest dark channel values
        dark_flat = dark_channel.reshape(-1)
        indices = np.argpartition(dark_flat, -num_top)[-num_top:]
        # Among these positions, pick the pixels in RGB with highest intensity
        rgb_flat = rgb.reshape(-1, 3)
        brightest = rgb_flat[indices]
        # A is the mean or max over these brightest pixels
        A = np.max(brightest, axis=0)  # shape (3,)
        return A

    @staticmethod
    def _estimate_transmission(
        rgb: np.ndarray,
        A: np.ndarray,
        omega: float,
        patch_size: int,
    ) -> np.ndarray:
        """
        Estimate transmission t(x) = 1 - omega * dark_channel( I / A ).
        """
        # Normalize by atmospheric light
        norm = rgb / (A[None, None, :] + 1e-8)
        dark_norm = DCPDehazer._get_dark_channel(norm, patch_size)
        t = 1.0 - omega * dark_norm
        return np.clip(t, 0.0, 1.0)

    @staticmethod
    def _recover_scene_radiance(
        rgb: np.ndarray,
        A: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Recover scene radiance:
            J(x) = (I(x) - A) / t(x) + A
        """
        # Expand t and A to image shape
        t_expanded = t[..., None]
        A_expanded = A[None, None, :]
        J = (rgb - A_expanded) / (t_expanded + 1e-8) + A_expanded
        J = np.clip(J, 0.0, 1.0)
        return J

    # ---------- Guided filter (edge-preserving refinement) ---------- #

    @staticmethod
    def _guided_filter(
        guidance_rgb: np.ndarray,
        t: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """
        Edge-preserving smoothing of the transmission map using a fast guided filter.

        Args:
            guidance_rgb: HxWx3 RGB image in [0,1], used as guidance.
            t: HxW coarse transmission map in [0,1].
            radius: local window radius.
            eps: regularization term.

        Returns:
            t_refined: HxW refined transmission map.
        """
        # Use gray guidance for simplicity
        guidance_gray = cv2.cvtColor(
            (guidance_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0

        return DCPDehazer._fast_guided_filter_gray(guidance_gray, t, radius, eps)

    @staticmethod
    def _box_filter(src: np.ndarray, r: int) -> np.ndarray:
        """Fast box filter using integral image."""
        kernel_size = 2 * r + 1
        return cv2.blur(src, (kernel_size, kernel_size))

    @staticmethod
    def _fast_guided_filter_gray(
        I: np.ndarray,
        p: np.ndarray,
        r: int,
        eps: float,
    ) -> np.ndarray:
        """
        Fast guided filter for gray guidance image.

        Not the most optimized implementation, but sufficient for this project.
        Reference: K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010.
        """
        I = I.astype(np.float32)
        p = p.astype(np.float32)

        mean_I = DCPDehazer._box_filter(I, r)
        mean_p = DCPDehazer._box_filter(p, r)
        mean_Ip = DCPDehazer._box_filter(I * p, r)

        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = DCPDehazer._box_filter(I * I, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = DCPDehazer._box_filter(a, r)
        mean_b = DCPDehazer._box_filter(b, r)

        q = mean_a * I + mean_b
        return np.clip(q, 0.0, 1.0)
