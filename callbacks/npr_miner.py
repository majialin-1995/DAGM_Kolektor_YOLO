"""Negative Patch Replay augmentation bridged into Ultralytics callbacks."""

from __future__ import annotations

import random
from typing import Dict

import cv2
import numpy as np

from .preproc_srts import _numpy_to_tensor, _tensor_to_numpy


class NegativePatchReplay:
    """Randomly copy negative patches within an image."""

    def __init__(self, prob: float = 0.5, patch_size: int = 128) -> None:
        self.prob = prob
        self.ps = patch_size

    def __call__(self, im: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return im
        h, w = im.shape[:2]
        if h < self.ps or w < self.ps:
            return im
        for _ in range(4):
            x = random.randint(0, w - self.ps)
            y = random.randint(0, h - self.ps)
            X = random.randint(0, w - self.ps)
            Y = random.randint(0, h - self.ps)
            patch = im[y : y + self.ps, x : x + self.ps].copy()
            im[Y : Y + self.ps, X : X + self.ps] = cv2.addWeighted(
                im[Y : Y + self.ps, X : X + self.ps], 0.5, patch, 0.5, 0
            )
        return im


class UltralyticsNPRCallback:
    """Install NPR image augmentation inside Ultralytics' preprocessing step."""

    def __init__(self, prob: float = 0.5, patch_size: int = 128) -> None:
        self.npr = NegativePatchReplay(prob=prob, patch_size=patch_size)

    def _apply(self, batch: Dict) -> Dict:
        imgs = batch.get("img")
        if imgs is None:
            return batch
        imgs_np, scaled, device, dtype = _tensor_to_numpy(imgs)
        for i in range(len(imgs_np)):
            imgs_np[i] = self.npr(imgs_np[i])
        batch["img"] = _numpy_to_tensor(imgs_np, scaled, device, dtype)
        return batch

    def on_pretrain_routine_end(self, trainer) -> None:  # noqa: ANN001 - Ultralytics callback signature
        if getattr(trainer, "_npr_preproc_wrapped", False):
            return

        original_preprocess = trainer.preprocess_batch

        def wrapped(batch, *, _orig=original_preprocess):
            batch = self._apply(batch)
            return _orig(batch)

        trainer.preprocess_batch = wrapped
        setattr(trainer, "_npr_preproc_wrapped", True)
