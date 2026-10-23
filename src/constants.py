from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_cell_growth_4factor.csv"

TARGET_COLUMN = "Specific Growth Rate [d^-1]"

FEATURE_COLUMNS = [
    "Temperature [C]",
    "pH",
    "Glucose [mM]",
    "Lactate [mM]",
]