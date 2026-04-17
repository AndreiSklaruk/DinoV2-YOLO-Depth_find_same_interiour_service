"""
build_index.py — Standalone скрипт для построения FAISS-индекса.

Запуск:
    python build_index.py

Нужен интернет при первом запуске для загрузки модели (~330 MB).
"""

# ── MPS включён (ARM Python 3.11 + torch 2.2.2 — стабильно на Apple Silicon) ──
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback на CPU для несовместимых ops
# ──────────────────────────────────────────────────────────────────────────────

import sys
import time
from pathlib import Path

# Добавить корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import DATABASE_DIR, INDEX_DIR, MODEL_NAME
from app.indexer import build_index, is_index_built

if __name__ == "__main__":
    print("=" * 60)
    print("  DINOv2 Room Retrieval — Index Builder")
    print("=" * 60)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Database: {DATABASE_DIR}")
    print(f"  Index   : {INDEX_DIR}")
    print("=" * 60)

    if is_index_built():
        ans = input("\n⚠️  Индекс уже существует. Пересобрать? [y/N]: ").strip().lower()
        if ans != "y":
            print("Отменено.")
            sys.exit(0)

    print("\n🔄 Запуск индексации...\n")
    start = time.time()

    try:
        stats = build_index()
        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("  ✅ Индекс успешно построен!")
        print(f"  📊 Проиндексировано комнат : {stats['count']}")
        print(f"  ⏱  Время                  : {stats['elapsed_sec']} сек")
        print(f"  📁 Путь к индексу         : {stats['index_path']}")
        print("=" * 60)
        print("\nТеперь можно запускать сервер:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
