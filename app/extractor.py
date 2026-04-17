"""
extractor.py — Загрузка DINOv2 и извлечение 1536-dim эмбеддингов.

Загрузка: torch.hub.load с trust_repo=True (без трансформеров).

Стратегия эмбеддинга:
  embedding = L2_norm( concat(CLS_token, mean(patch_tokens)) )
  CLS (768) + mean_patches (768) = 1536 dim total
"""

from __future__ import annotations

# Блокируем MPS до любых импортов torch (предотвращает зависание на macOS)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback на CPU для несупортируемых MPS ops

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

from app.config import IMAGE_SIZE, EMBEDDING_DIM

# ─── Глобальный синглтон ─────────────────────────────────────────────────────
_model: torch.nn.Module | None = None
_transform: T.Compose | None = None
_device: torch.device = torch.device("cpu")


def _build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_device() -> torch.device:
    """Определяет лучшее доступное устройство: MPS > CPU.
    Установи DINOV2_FORCE_CPU=1 чтобы принудительно использовать CPU.
    """
    if os.environ.get("DINOV2_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> tuple[torch.nn.Module, T.Compose, torch.device]:
    """Загружает DINOv2 один раз и кэширует. MPS для Apple Silicon."""
    global _model, _transform, _device

    if _model is not None:
        return _model, _transform, _device

    _device = get_device()
    print(f"[DINOv2] Loading dinov2_vitb14 via torch.hub (device={_device}) ...", flush=True)

    _model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        pretrained=True,
        verbose=False,
        trust_repo=True,
    )
    _model.eval()
    _model.to(_device)
    _transform = _build_transform()

    print(f"[DINOv2] Model ready. Embedding dim: {EMBEDDING_DIM}", flush=True)
    return _model, _transform, _device


def extract_embedding(image: "Image.Image | str | os.PathLike") -> np.ndarray:
    """
    Извлечь 1536-dim L2-нормализованный эмбеддинг.

    Returns:
        np.ndarray shape [1536], dtype=float32
    """
    model, transform, device = load_model()

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.forward_features(tensor)

    cls_token    = features["x_norm_clstoken"]     # [1, 768]
    patch_tokens = features["x_norm_patchtokens"]  # [1, N, 768]
    mean_patches = patch_tokens.mean(dim=1)         # [1, 768]

    combined = torch.cat([cls_token, mean_patches], dim=1)  # [1, 1536]
    combined = F.normalize(combined, p=2, dim=1)

    return combined.squeeze(0).cpu().float().numpy()
