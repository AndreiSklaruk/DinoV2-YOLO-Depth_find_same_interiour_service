from pathlib import Path

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent   # /Users/pro/git/DINOv2
DATABASE_DIR = BASE_DIR / "database"
INDEX_DIR    = BASE_DIR / "index"
UPLOADS_DIR  = BASE_DIR / "uploads"
STATIC_DIR   = BASE_DIR / "static"

# ─── Model settings ───────────────────────────────────────────────────────────
MODEL_NAME    = "facebook/dinov2-base"  # ViT-B/14 — 768-dim
IMAGE_SIZE    = 518                      # кратно patch_size=14 (37×14=518)
EMBEDDING_DIM = 1536                     # CLS (768) + mean patches (768)

# ─── Search settings ──────────────────────────────────────────────────────────
TOP_K = 10

# ─── ImageNet normalization constants ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Supported image extensions ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
