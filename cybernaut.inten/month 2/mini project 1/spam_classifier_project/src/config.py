"""Central configuration: all paths and constants live here."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "spam.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "spam_classifier_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
