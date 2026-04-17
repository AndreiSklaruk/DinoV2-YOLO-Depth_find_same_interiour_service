"""
build_yolo_index.py — Скрипт массового сканирования базы изображений
Формирует файл index/yolo_metadata.json с секторами окон и дверей.
"""

import json
import time
from pathlib import Path
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from app.config import DATABASE_DIR, INDEX_DIR, SUPPORTED_EXTENSIONS
from app.yolo_extractor import extract_yolo_grid

BEFORE_DIR = DATABASE_DIR / "before"
YOLO_METADATA_PATH = INDEX_DIR / "yolo_metadata.json"

def build_yolo_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    images = [
        p for p in BEFORE_DIR.iterdir()
        if p.suffix in SUPPORTED_EXTENSIONS and "_before" in p.stem
    ]
    
    # Сортировка для предсказуемости
    def sort_key(p: Path) -> int:
        parts = p.stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
        
    images = sorted(images, key=sort_key)
    
    if not images:
        print("[YOLO Indexer] Ошибка: нет изображений в database/before")
        return
        
    print(f"[YOLO Indexer] Сканируем {len(images)} комнат (режим: 3x3 сетка)...")
    
    metadata = {}
    
    start = time.time()
    
    for img_path in tqdm(images, desc="YOLO Analysis", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")
            grid_data = extract_yolo_grid(image)
            metadata[img_path.name] = grid_data
        except Exception as e:
            print(f"WARN: Ошибка при обработке {img_path.name}: {e}")
            
    elapsed = time.time() - start
    
    with open(YOLO_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"[YOLO Indexer] ✅ Готово: {len(metadata)} файлов за {elapsed:.1f}с")
    print(f"[YOLO Indexer] Сохранено в {YOLO_METADATA_PATH}")

if __name__ == "__main__":
    build_yolo_index()
