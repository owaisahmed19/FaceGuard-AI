import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Dataset path (folders inside represent individuals)
DATASET_DIR = DATA_DIR / "people_dataset"
DATASET_DIR.mkdir(exist_ok=True)

# File to store face encodings so we do not rebuild them constantly
EMBEDDINGS_FILE = MODELS_DIR / "face_embeddings.pkl"

# Face recognition settings using InsightFace
# InsightFace downloads models automatically to ~/.insightface/models/buffalo_l

FACE_MATCH_THRESHOLD = 0.65  # Cosine similarity threshold — tuned for averaged embeddings (0.65 = ~85%+ accuracy)
FACE_MIN_QUALITY     = 0.70  # Minimum InsightFace detection confidence score to accept a face
FACE_MIN_SIZE        = 60    # Minimum face bounding-box side length (pixels) to accept

# Database settings
DB_URL = os.getenv("DB_URL", f"sqlite:///{LOGS_DIR}/faceguard.db")
