"""Deep-learning track (DINOv2 primary, ResNet50 fallback).

DINOv2 is self-supervised on visual similarity — the current go-to for
place recognition without task-specific training. With only ~100 images
we have no room to fine-tune anyway; frozen pretrained features are
the right call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.features.base import l2_normalize

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pick_device(prefer: str | None = None) -> str:
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    # NOTE: MPS segfaults when co-loaded with faiss on macOS. Stick to CPU
    # until we need the speed — DINOv2 ViT-S/14 is fast enough on CPU for
    # our ~100-image dataset.
    return "cpu"


class DinoV2Embedder:
    """DINOv2 ViT-S/14 global embedding (384-dim, L2-normalized).

    Uses the CLS token from the final layer, which is what the DINOv2
    authors recommend for image-level similarity.
    """

    name = "dinov2_vits14"
    dim = 384

    def __init__(
        self,
        device: str | None = None,
        model_id: str = "vit_small_patch14_dinov2.lvd142m",
        image_size: int = 518,
    ):
        import timm

        self.device = _pick_device(device)
        self.image_size = image_size
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)

    def fit(self, image_paths: list[Path]) -> None:
        return  # frozen pretrained model — nothing to fit

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        w, h = pil.size
        scale = self.image_size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        pil = pil.crop((left, top, left + self.image_size, top + self.image_size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)

    @torch.inference_mode()
    def embed(self, image: np.ndarray) -> np.ndarray:
        x = self._preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model(x)  # (1, 384)
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        feats = self.model(batch)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()


class Resnet50Embedder:
    """ImageNet-pretrained ResNet50, pooled features (2048-dim)."""

    name = "resnet50_pool"
    dim = 2048

    def __init__(self, device: str | None = None):
        import timm

        self.device = _pick_device(device)
        self.model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        self.image_size = 224

    def fit(self, image_paths: list[Path]) -> None:
        return

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        scale = 256 / min(pil.size)
        new_w, new_h = int(round(pil.width * scale)), int(round(pil.height * scale))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        pil = pil.crop((left, top, left + self.image_size, top + self.image_size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - np.array(_IMAGENET_MEAN, dtype=np.float32)) / np.array(
            _IMAGENET_STD, dtype=np.float32
        )
        return torch.from_numpy(arr.transpose(2, 0, 1))

    @torch.inference_mode()
    def embed(self, image: np.ndarray) -> np.ndarray:
        x = self._preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model(x)
        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        feats = self.model(batch)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()
