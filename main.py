from src.constants import PROJECT_ROOT
from src.functions import (
    calculate_feature_importance,
    compare_models,
    load_dataset,
    plot_partial_dependence,
)


def main():
    data = load_dataset()
    results = compare_models(data)
    importance_results = calculate_feature_importance(data)
    figure_path = plot_partial_dependence(data)

    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(exist_ok=True)

    results.to_csv(tables_dir / "model_comparison.csv", index=False)
    importance_results.to_csv(
        tables_dir / "feature_importance.csv",
        index=False,
    )

    print("Model comparison:")
    print(results.to_string(index=False))

    print("\nPermutation feature importance:")
    print(importance_results.to_string(index=False))

    print(f"\nSaved partial dependence plot: {figure_path}")


if __name__ == "__main__":
    main()