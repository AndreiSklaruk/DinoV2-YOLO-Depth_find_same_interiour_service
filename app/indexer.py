"""
indexer.py — Построение и загрузка FAISS-индекса из папки database/before/.

Индекс хранит эмбеддинги только "before" изображений (bc_NNN_before.jpg).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ВАЖНО: faiss импортируется ЛЕНИВО (внутри функций), иначе конфликт C++ runtime с torch → segfault

from app.config import (
    DATABASE_DIR,
    EMBEDDING_DIM,
    INDEX_DIR,
    SUPPORTED_EXTENSIONS,
)
from app.extractor import extract_embedding

BEFORE_DIR = DATABASE_DIR / "before"
AFTER_DIR  = DATABASE_DIR / "after"

# ─── Пути файлов индекса ──────────────────────────────────────────────────────
FAISS_INDEX_PATH    = INDEX_DIR / "faiss.index"
EMBEDDINGS_NPY_PATH = INDEX_DIR / "embeddings.npy"
METADATA_JSON_PATH  = INDEX_DIR / "metadata.json"


def _get_before_images() -> list[Path]:
    """
    Вернуть список "before" изображений из DATABASE_DIR/before/.
    Паттерн: bc_NNN_before.jpg
    Сортируем по номеру для воспроизводимости.
    """
    BEFORE_DIR.mkdir(parents=True, exist_ok=True)
    images = [
        p for p in BEFORE_DIR.iterdir()
        if p.suffix in SUPPORTED_EXTENSIONS and "_before" in p.stem
    ]
    # Сортировка по числовому номеру (bc_001_before → 1)
    def sort_key(p: Path) -> int:
        parts = p.stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    return sorted(images, key=sort_key)


def build_index() -> dict:
    """
    Просканировать все "before" изображения, извлечь эмбеддинги и сохранить FAISS индекс.

    Returns:
        Словарь со статистикой: {"count": int, "elapsed_sec": float}
    """
    import faiss  # Lazy import — после torch, иначе segfault на macOS
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    images = _get_before_images()
    if not images:
        raise FileNotFoundError(f"Не найдено изображений в {BEFORE_DIR}")

    print(f"[Indexer] Найдено {len(images)} пустых комнат для индексации")
    print(f"[Indexer] Сохранение индекса в {INDEX_DIR}")

    embeddings_list: list[np.ndarray] = []
    metadata: dict[str, str] = {}  # str(int_id) → filename

    start = time.time()

    total = len(images)
    for idx, img_path in enumerate(images):
        try:
            embedding = extract_embedding(img_path)  # [1536] float32 L2-norm
            embeddings_list.append(embedding)
            metadata[str(idx)] = img_path.name
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                elapsed_so_far = time.time() - start
                print(f"[Indexer] {idx+1}/{total} | {img_path.name} | {elapsed_so_far:.1f}s")
        except Exception as e:
            print(f"[Indexer] WARN: Пропускаем {img_path.name}: {e}")

    elapsed = time.time() - start

    if not embeddings_list:
        raise RuntimeError("Не удалось извлечь ни одного эмбеддинга")

    # Собрать матрицу [N, 1536]
    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)

    # Создать FAISS индекс (Inner Product = cosine similarity, т.к. L2-нормализованы)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Сохранить
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(str(EMBEDDINGS_NPY_PATH), embeddings)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    stats = {
        "count": len(embeddings_list),
        "elapsed_sec": round(elapsed, 1),
        "index_path": str(FAISS_INDEX_PATH),
    }
    print(f"[Indexer] ✅ Готово: {stats['count']} комнат за {stats['elapsed_sec']}с")
    return stats


def load_index() -> tuple["faiss.Index", dict[int, str]]:
    """
    Загрузить готовый FAISS-индекс и метаданные с диска.

    Returns:
        (faiss.Index, {int_id: filename})

    Raises:
        FileNotFoundError: если индекс не построен.
    """
    import faiss  # Lazy import
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Индекс не найден: {FAISS_INDEX_PATH}\n"
            "Запустите: python build_index.py"
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        raw_meta = json.load(f)

    # Конвертировать ключи из str в int
    metadata = {int(k): v for k, v in raw_meta.items()}

    print(f"[Indexer] Индекс загружен: {index.ntotal} векторов")
    return index, metadata


def is_index_built() -> bool:
    """Проверить, построен ли индекс."""
    return FAISS_INDEX_PATH.exists() and METADATA_JSON_PATH.exists()
