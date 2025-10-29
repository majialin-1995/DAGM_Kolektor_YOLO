"""SRTS preprocessing callback compatible with Ultralytics training hooks."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy import ndimage


def _tensor_to_numpy(imgs: torch.Tensor) -> tuple[np.ndarray, bool, torch.device, torch.dtype]:
    """Convert BCHW tensor to uint8 numpy array in HWC order, tracking scale state."""

    device = imgs.device
    dtype = imgs.dtype
    imgs_cpu = imgs.detach().cpu()
    if torch.is_floating_point(imgs_cpu):
        imgs_np = imgs_cpu.numpy().astype(np.float32)
        imgs_np = np.clip(imgs_np, 0.0, 1.0)
        imgs_np = (imgs_np * 255.0).astype(np.uint8)
        scaled = True
    else:
        imgs_np = imgs_cpu.numpy()
        if imgs_np.dtype != np.uint8:
            imgs_np = imgs_np.astype(np.uint8)
        scaled = False
    imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
    return imgs_np, scaled, device, dtype


def _numpy_to_tensor(imgs_np: np.ndarray, scaled: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert augmented HWC numpy array back to BCHW tensor respecting original scale."""

    if scaled:
        imgs_np = imgs_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(imgs_np, (0, 3, 1, 2))).to(dtype=torch.float32)
    else:
        tensor = torch.from_numpy(np.transpose(imgs_np.astype(np.uint8), (0, 3, 1, 2))).to(dtype=dtype)
    return tensor.to(device)


class SRTSPreproc:
    """Saliency-Refined Texture Strengthening preprocessor."""

    def __init__(self, prob: float = 0.5, sigma: float = 2.5, alpha: float = 0.65) -> None:
        self.prob = prob
        self.sigma = sigma
        self.alpha = alpha

    def saliency_sr(self, gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32) / 255.0
        F = np.fft.fft2(gray)
        A = np.abs(F)
        L = np.log(A + 1e-6)
        R = L - ndimage.gaussian_filter(L, sigma=self.sigma)
        S = np.abs(np.fft.ifft2(np.exp(R + 1j * np.angle(F)))) ** 2
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)
        return S

    def __call__(self, im: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.prob:
            return im
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sal = self.saliency_sr(g)
        sal = cv2.GaussianBlur(sal, (0, 0), 2.0)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        sal3 = np.repeat(sal[..., None], 3, axis=2)
        return (im.astype(np.float32) * (0.5 + self.alpha * sal3)).clip(0, 255).astype(np.uint8)


class UltralyticsSRTSCallback:
    """Bridge SRTS preprocessing into Ultralytics' callback system."""

    def __init__(self, prob: float = 0.5, sigma: float = 2.5, alpha: float = 0.65) -> None:
        self.t = SRTSPreproc(prob=prob, sigma=sigma, alpha=alpha)

    def _apply(self, batch: dict) -> dict:
        imgs = batch.get("img")
        if imgs is None:
            return batch
        imgs_np, scaled, device, dtype = _tensor_to_numpy(imgs)
        for i in range(len(imgs_np)):
            imgs_np[i] = self.t(imgs_np[i])
        batch["img"] = _numpy_to_tensor(imgs_np, scaled, device, dtype)
        return batch

    def on_pretrain_routine_end(self, trainer) -> None:  # noqa: ANN001 - Ultralytics callback signature
        if getattr(trainer, "_srts_preproc_wrapped", False):
            return

        original_preprocess = trainer.preprocess_batch

        def wrapped(batch, *, _orig=original_preprocess):
            batch = self._apply(batch)
            return _orig(batch)

        trainer.preprocess_batch = wrapped
        setattr(trainer, "_srts_preproc_wrapped", True)
