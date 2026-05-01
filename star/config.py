from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGE_DIR = PROCESSED_DATA_DIR / "images"
LABELS_FILE = PROCESSED_DATA_DIR / "labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "star_cnn.pt"

CLASS_NAMES = ["elliptical", "spiral"]
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_STATE = 42