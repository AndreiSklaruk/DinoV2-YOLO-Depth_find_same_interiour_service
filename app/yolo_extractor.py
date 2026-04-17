"""
yolo_extractor.py — Модуль для работы с пространственной детекцией объектов.
Использует ультрабыструю модель YOLOv8 (OpenImages V7) для поиска окон и дверей.
Вычисляет положение объектов по сетке 3x3.
"""

from __future__ import annotations

import logging
from typing import Dict, List
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    # Обертка для graceful fallback
    YOLO = None

# Константы классов YOLOv8 OIV7
OIV7_WINDOW = 587
OIV7_DOOR = 164

_model = None

def get_yolo_model():
    """Ленивая загрузка модели."""
    global _model
    if YOLO is None:
        raise ImportError("Ultralytics не установлен. Выполните pip install ultralytics")
    if _model is None:
        logging.info("[YOLO] Загрузка yolov8m-oiv7.pt ...")
        _model = YOLO("yolov8m-oiv7.pt", verbose=False)
    return _model

def _get_sector(x_center: float, y_center: float, width: int, height: int) -> str:
    """Вычисляет 3x3 позицию по координатам центра."""
    # X axis -> left, center, right
    if x_center < width / 3:
        col = "left"
    elif x_center < (2 * width) / 3:
        col = "center"
    else:
        col = "right"
    
    # Y axis -> top, mid, bottom
    if y_center < height / 3:
        row = "top"
    elif y_center < (2 * height) / 3:
        row = "mid"
    else:
        row = "bottom"
        
    return f"{row}-{col}"

def extract_yolo_grid(image: Image.Image) -> Dict[str, List[str]]:
    """
    Детектирует окна и двери и возвращает сектора.
    {"windows": ["top-left", "mid-right"], "doors": ["bottom-center"]}
    """
    model = get_yolo_model()
    
    # DINOv2 и Depth обычно работают с RGB PIL напрямую. YOLOv8 принимает PIL Image или numpy.
    results = model(image, verbose=False)[0]
    
    width, height = image.size
    
    features = {
        "windows": set(),
        "doors": set()
    }
    
    # Итерируемся по боксам результата
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        # Минимальный порог уверенности, чтобы отсечь шум
        if conf < 0.20:
            continue
            
        if cls_id == OIV7_WINDOW or cls_id == OIV7_DOOR:
            # Получаем координаты центра [xc, yc, w, h]
            xywh = box.xywh[0].tolist()
            xc, yc = xywh[0], xywh[1]
            
            sector = _get_sector(xc, yc, width, height)
            
            if cls_id == OIV7_WINDOW:
                features["windows"].add(sector)
            else:
                features["doors"].add(sector)
                
    return {
        "windows": list(features["windows"]),
        "doors": list(features["doors"])
    }

def get_yolo_visualization(image: Image.Image) -> Image.Image:
    """
    Возвращает изображение с отрисованными боксами для окон и дверей (и сеткой 3x3).
    """
    model = get_yolo_model()
    
    # YOLO inference
    results = model(image, verbose=False)[0]
    
    # Создаем копию для отрисовки
    img_pil = image.copy().convert("RGBA")
    
    from PIL import ImageDraw, ImageFont
    # Итерируемся по боксам результата
    
    # Создаем прозрачный слой для прямоугольников и заливок
    overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        # Минимальный порог уверенности
        if conf < 0.20:
            continue
            
        if cls_id == OIV7_WINDOW or cls_id == OIV7_DOOR:
            # Получаем координаты [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            
            color = (236, 72, 153, 160) if cls_id == OIV7_WINDOW else (59, 130, 246, 160)
            label = f"Window {conf:.2f}" if cls_id == OIV7_WINDOW else f"Door {conf:.2f}"
            
            # Рисуем прямоугольник
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Рисуем подложку для текста
            draw.rectangle([x1, y1 - 20, x1 + 80, y1], fill=color)
            draw.text((x1 + 4, y1 - 18), label, fill=(255, 255, 255, 255))
            
    width, height = img_pil.size
    
    # Вертикальные линии сетки
    draw.line([(width/3, 0), (width/3, height)], fill=(255, 255, 255, 128), width=2)
    draw.line([(2*width/3, 0), (2*width/3, height)], fill=(255, 255, 255, 128), width=2)
    
    # Горизонтальные линии сетки
    draw.line([(0, height/3), (width, height/3)], fill=(255, 255, 255, 128), width=2)
    draw.line([(0, 2*height/3), (width, 2*height/3)], fill=(255, 255, 255, 128), width=2)
    
    out = Image.alpha_composite(img_pil, overlay)
    return out.convert("RGB")
