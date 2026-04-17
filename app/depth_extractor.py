"""
depth_extractor.py — Извлечение геометрических признаков через Depth Anything V2.

Модель: depth-anything/Depth-Anything-V2-Small-hf (~100MB)
Выход:  L2-нормализованный вектор [2048]

Признаки (2048 dim):
  [0:1024]    — raw depth map 32x32, нормализованный
  [1024:2048] — gradient magnitude 32x32 (Sobel), нормализованный

Почему градиент?
  Пустые комнаты имеют похожие raw depth maps (пол близко, стены дальше).
  Градиент глубины кодирует СТРУКТУРУ: где стены переходят в пол/потолок,
  где окна (резкие перепады глубины от тёмных рам к внешнему пространству),
  где углы комнаты. Это даёт реальную геометрическую дискриминацию.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image

# ─── Конфигурация ─────────────────────────────────────────────────────────────
DEPTH_MODEL_NAME    = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MAP_SIZE      = 48          # 48×48 — лучше детализирует рёбра поверхностей
DEPTH_EMBEDDING_DIM = DEPTH_MAP_SIZE * DEPTH_MAP_SIZE * 2  # 2304 raw + 2304 grad = 4608

# ─── Глобальный синглтон ──────────────────────────────────────────────────────
_depth_pipe = None


def load_depth_model():
    """Загружает Depth Anything V2 один раз и кэширует."""
    global _depth_pipe

    if _depth_pipe is not None:
        return _depth_pipe

    print(f"[DepthV2] Loading {DEPTH_MODEL_NAME} ...", flush=True)

    from transformers import pipeline as hf_pipeline

    _depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_NAME,
        device="cpu",
    )

    print(f"[DepthV2] Model ready. Embedding dim: {DEPTH_EMBEDDING_DIM}", flush=True)
    return _depth_pipe


def _sobel_gradient(arr: np.ndarray) -> np.ndarray:
    """
    Вычислить magnitude градиента Sobel для 2D массива.
    Возвращает массив той же формы.
    """
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    from scipy.ndimage import convolve
    gx = convolve(arr, kx)
    gy = convolve(arr, ky)
    return np.sqrt(gx**2 + gy**2)


def extract_depth_embedding(image: "Image.Image | str | os.PathLike") -> np.ndarray:
    """
    Извлечь 2048-dim L2-нормализованный геометрический эмбеддинг.

    Состав:
      [0:1024]    raw depth 32x32 — общая 3D структура
      [1024:2048] Sobel gradient 32x32 — границы поверхностей (стены/пол/окна)

    Returns:
        np.ndarray shape [2048], dtype=float32
    """
    pipe = load_depth_model()

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Получить карту глубины
    result = pipe(image)
    depth_pil = result["depth"]  # PIL Image (grayscale)

    # Ресайз до 32×32
    depth_small = depth_pil.resize((DEPTH_MAP_SIZE, DEPTH_MAP_SIZE), Image.BILINEAR)
    depth_arr = np.array(depth_small, dtype=np.float32)

    # ── Raw depth: нормализация в [0, 1] ──────────────────────────────────────
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max - d_min > 1e-6:
        raw = (depth_arr - d_min) / (d_max - d_min)
    else:
        raw = np.zeros_like(depth_arr)

    # ── Gradient magnitude (Sobel) ─────────────────────────────────────────────
    try:
        grad = _sobel_gradient(raw)
    except ImportError:
        # Если scipy недоступен — простой численный градиент
        gy, gx = np.gradient(raw)
        grad = np.sqrt(gx**2 + gy**2)

    # Нормализуем градиент
    g_max = grad.max()
    if g_max > 1e-6:
        grad = grad / g_max

    # ── Concat + L2 normalize ──────────────────────────────────────────────────
    combined = np.concatenate([raw.flatten(), grad.flatten()])  # [2048]
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm

    return combined.astype(np.float32)


# ── Визуализация (debug) ───────────────────────────────────────────────────────

def _apply_turbo_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Применить Turbo colormap к нормализованному depth-массиву.
    Входные данные: float32 [0..1]. Выход: uint8 RGB (H, W, 3).
    Холодный (синий) = близко, горячий (красный) = далеко.
    """
    TURBO = np.array([
        [48,  18,  59],   # 0.00 — тёмно-синий
        [65,  99, 201],   # 0.14 — синий
        [27, 189, 212],   # 0.29 — голубой
        [98, 231, 109],   # 0.43 — зелёный
        [249, 220,  34],  # 0.57 — жёлтый
        [251, 115,  18],  # 0.71 — оранжевый
        [220,  31,  23],  # 0.86 — красный
        [122,   4,   3],  # 1.00 — тёмно-красный
    ], dtype=np.float32)

    N = len(TURBO) - 1
    seg = np.clip(gray, 0.0, 1.0) * N
    idx_lo = np.clip(seg.astype(np.int32), 0, N - 1)
    idx_hi = (idx_lo + 1).clip(0, N)
    t = (seg - idx_lo)[..., np.newaxis]
    rgb = (TURBO[idx_lo] * (1.0 - t) + TURBO[idx_hi] * t).astype(np.uint8)
    return rgb


def get_depth_visualization(image: "Image.Image") -> "Image.Image":
    """
    Вернуть цветную карту глубины (Turbo colormap) в том же размере что входное фото.

    Используется для debug-визуализации в UI.
    """
    pipe = load_depth_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    w, h = image.size
    result = pipe(image)
    depth_pil = result["depth"]  # PIL grayscale (L mode)

    # Ресайз до исходного размера
    depth_resized = depth_pil.resize((w, h), Image.BILINEAR)
    depth_arr = np.array(depth_resized, dtype=np.float32) / 255.0

    # Нормализуем
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_norm = depth_arr

    rgb = _apply_turbo_colormap(depth_norm)
    return Image.fromarray(rgb)


def get_depth_gray(
    image: "Image.Image",
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Вернуть карту глубины как grayscale uint8 (H, W) для запуска LSD.

    Используется вместо оригинала RGB при MLSD-на-depth подходе:
    - Нет текстур паркета / обоев
    - Нет теней
    - Только геометрические перепады глубины

    Args:
        image:       PIL.Image источник
        target_size: (w, h) — если нужен конкретный размер, иначе оригинальный

    Returns:
        uint8 numpy массив (H, W), значения 0-255
    """
    import cv2

    pipe = load_depth_model()

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    w_orig, h_orig = image.size

    # Ограничиваем размер входа для depth модели (предотвращает segfault на больших фото)
    MAX_SIDE = 518
    if max(w_orig, h_orig) > MAX_SIDE:
        scale = MAX_SIDE / max(w_orig, h_orig)
        img_for_depth = image.resize(
            (int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS
        )
    else:
        img_for_depth = image

    result    = pipe(img_for_depth)
    depth_pil = result["depth"]      # PIL grayscale

    # Resize к нужному выходному размеру
    out_size      = target_size if target_size else (w_orig, h_orig)
    depth_resized = depth_pil.resize(out_size, Image.BILINEAR)
    depth_arr     = np.array(depth_resized, dtype=np.float32)

    # Нормализуем в [0 .. 255]
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_norm = depth_arr / 255.0

    depth_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)

    # Лёгкое размытие — убирает артефакты depth-модели
    depth_u8 = cv2.GaussianBlur(depth_u8, (3, 3), 0)

    return depth_u8


