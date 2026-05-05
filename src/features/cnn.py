"""Small CNN baseline used for the retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import DATA_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pick_device(prefer: str | None = None) -> str:
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _label_from_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).parts[0]
    except Exception:  # noqa: BLE001
        return path.parent.name


@dataclass(frozen=True)
class _Example:
    path: Path
    label_idx: int


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[_Example], transform: transforms.Compose):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        img = Image.open(ex.path).convert("RGB")
        return self.transform(img), ex.label_idx


class _TinyPlaceCNN(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        self.features = nn.Sequential(*layers)
        self.project = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x).flatten(1)
        embedding = self.project(features)
        logits = self.classifier(F.relu(embedding))
        return embedding, logits


class SimpleCNNEmbedder:
    name = "simple_cnn"
    dim = 128

    def __init__(
        self,
        device: str | None = None,
        image_size: int = 160,
        epochs: int = 12,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.device = _pick_device(device)
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.class_names: list[str] = []
        self.model: _TinyPlaceCNN | None = None
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self._eval_tf = transforms.Compose(
            [
                transforms.Resize(int(round(image_size * 1.15))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def _build_model(self, num_classes: int) -> _TinyPlaceCNN:
        model = _TinyPlaceCNN(embedding_dim=self.dim, num_classes=num_classes)
        return model.to(self.device)

    def fit(self, image_paths: list[Path]) -> None:
        if self.model is not None:
            return
        if not image_paths:
            raise ValueError("SimpleCNNEmbedder.fit() needs at least one image")

        label_names = sorted({_label_from_path(p) for p in image_paths})
        if len(label_names) < 2:
            raise ValueError(
                "SimpleCNNEmbedder needs at least 2 classes for supervised training"
            )

        label_to_idx = {label: i for i, label in enumerate(label_names)}
        examples = [
            _Example(path=path, label_idx=label_to_idx[_label_from_path(path)])
            for path in image_paths
        ]
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            _ImageDataset(examples, self._train_tf),
            batch_size=min(self.batch_size, len(examples)),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.class_names = label_names
        self.model = self._build_model(num_classes=len(label_names))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                _, logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def save(self, prefix: Path) -> None:
        if self.model is None:
            return
        prefix.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "image_size": self.image_size,
            "dim": self.dim,
        }
        torch.save(payload, prefix.with_suffix(".pt"))

    def load(self, prefix: Path) -> bool:
        path = prefix.with_suffix(".pt")
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device)
        self.class_names = list(payload["class_names"])
        self.model = self._build_model(num_classes=len(self.class_names))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return True

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        return self._eval_tf(pil)

    @torch.inference_mode()
    def embed(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SimpleCNNEmbedder must be fit() or load()'ed first")
        x = self._preprocess(image).unsqueeze(0).to(self.device)
        emb, _ = self.model(x)
        emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SimpleCNNEmbedder must be fit() or load()'ed first")
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        emb, _ = self.model(batch)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)
