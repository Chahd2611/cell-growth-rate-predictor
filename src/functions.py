import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.constants import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN ,DATASET_PATH, FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN



def load_dataset():
    """Load the growth-rate dataset and check that it has the required columns."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    return data.dropna(subset=required_columns).copy()


def compare_models(data):
    """Compare regression models using cross-validation and a held-out test set."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear regression": LinearRegression(),
        "Polynomial regression (degree 2)": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        ),
        "K-nearest neighbors (k=5)": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5, weights="distance"),
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "CV RMSE mean": -scores["test_rmse"].mean(),
            "CV RMSE SD": scores["test_rmse"].std(),
            "CV R2 mean": scores["test_r2"].mean(),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "Test R2": r2_score(y_test, predictions),
        })

    return pd.DataFrame(results).sort_values("CV RMSE mean")

def calculate_feature_importance(data):
    """Measure how much each feature affects random-forest prediction error."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )

    results = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance mean": importance.importances_mean,
        "Importance SD": importance.importances_std,
    })

    return results.sort_values("Importance mean", ascending=False)

def plot_partial_dependence(data):
    """Save plots showing each feature's average relationship with growth rate."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X, y)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=FEATURE_COLUMNS,
        ax=axes.ravel(),
        grid_resolution=30,
    )

    figure.suptitle("Partial dependence of predicted growth rate")
    figure.tight_layout()

    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / "partial_dependence.png"

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path