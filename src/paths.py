import os
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__))) / ".."

DATA_PATH = ROOT / "data"
FIGURES_PATH = ROOT / "reports" / "figures"
MODELS_PATH = ROOT / "models"
EXPERIMENTS_PATH = ROOT / "experiments"
TESTS_PATH = ROOT / "tests"
