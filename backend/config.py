import os

# Project root is one level above this file (which lives in backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── HuggingFace model ──────────────────────────────────────────────────────
HF_REPO_ID     = "Sauravyadav7721/Brain-Breast-Tumor"
MODEL_FILENAME = "trig_model.keras"

# ── Image params ───────────────────────────────────────────────────────────
IMG_SIZE = 128

# ── Directory paths (all relative to project root) ─────────────────────────
LOGS_DIR     = os.path.join(BASE_DIR, "backend", "logs")
MODELS_DIR   = os.path.join(BASE_DIR, "backend", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
