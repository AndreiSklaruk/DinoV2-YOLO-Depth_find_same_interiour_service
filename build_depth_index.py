"""
build_depth_index.py — Построение FAISS-индекса карт глубины.

Запуск:
    python build_depth_index.py

Создаёт:
    index/depth_faiss.index
    index/depth_embeddings.npy

ВАЖНО: Запускать ПОСЛЕ build_index.py.
Порядок изображений должен совпадать с основным индексом.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import DATABASE_DIR, INDEX_DIR, SUPPORTED_EXTENSIONS

DEPTH_FAISS_PATH      = INDEX_DIR / "depth_faiss.index"
DEPTH_EMBEDDINGS_PATH = INDEX_DIR / "depth_embeddings.npy"
METADATA_JSON_PATH    = INDEX_DIR / "metadata.json"


def get_before_images_ordered() -> list[Path]:
    """Возвращает before-изображения в том же порядке что и основной индекс."""
    if not METADATA_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Основной индекс не найден: {METADATA_JSON_PATH}\n"
            "Сначала запустите: python build_index.py"
        )

    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    BEFORE_DIR = DATABASE_DIR / "before"
    # Восстанавливаем порядок из metadata (ключи — int индексы)
    ordered = [
        BEFORE_DIR / metadata[str(i)]
        for i in range(len(metadata))
    ]
    return ordered


if __name__ == "__main__":
    print("=" * 60)
    print("  DINOv2 Room Retrieval — Depth Index Builder")
    print("=" * 60)

    images = get_before_images_ordered()
    total  = len(images)
    print(f"  Images  : {total}")
    print(f"  Index   : {INDEX_DIR}")
    print("=" * 60)
    print("\n🔄 Запуск извлечения карт глубины...\n")

    # Lazy imports после os.environ
    import numpy as np
    from app.depth_extractor import extract_depth_embedding, DEPTH_EMBEDDING_DIM

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_list = []
    start = time.time()

    for idx, img_path in enumerate(images):
        try:
            emb = extract_depth_embedding(img_path)
            embeddings_list.append(emb)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                elapsed = time.time() - start
                print(f"[Depth] {idx+1}/{total} | {img_path.name} | {elapsed:.1f}s")
        except Exception as e:
            print(f"[Depth] WARN: Пропускаем {img_path.name}: {e}")
            # Добавляем нулевой вектор чтобы не сбить индексы
            embeddings_list.append(np.zeros(DEPTH_EMBEDDING_DIM, dtype=np.float32))

    elapsed = time.time() - start

    if not embeddings_list:
        print("❌ Не удалось извлечь ни одного эмбеддинга")
        sys.exit(1)

    import faiss
    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)

    index = faiss.IndexFlatIP(DEPTH_EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(DEPTH_FAISS_PATH))
    np.save(str(DEPTH_EMBEDDINGS_PATH), embeddings)

    print("\n" + "=" * 60)
    print("  ✅ Depth индекс успешно построен!")
    print(f"  📊 Обработано комнат : {len(embeddings_list)}")
    print(f"  ⏱  Время             : {round(elapsed, 1)} сек")
    print(f"  📁 Индекс            : {DEPTH_FAISS_PATH}")
    print("=" * 60)
    print("\nТеперь перезапустите сервер:\n  uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n")
