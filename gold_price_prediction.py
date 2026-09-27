"""
Gold Price Prediction using Random Forest Regressor
====================================================
Full pipeline: load → EDA → train → evaluate → plots → feature importance → sample prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_dataset():
    for path in [
        "gld_price_data.csv",
        "gold price dataset.csv",
        "/content/gld_price_data.csv",
        "/content/gold price dataset.csv",
    ]:
        if os.path.exists(path):
            print(f"Loading: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("Dataset not found. Place gld_price_data.csv in the project folder.")


def main():
    print("=" * 60)
    print("  Gold Price Prediction | Random Forest Regressor")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading dataset...")
    gold_data = load_dataset()
    print(f"Shape: {gold_data.shape}")
    print("\nFirst 5 rows:\n", gold_data.head())
    print("\nLast 5 rows:\n", gold_data.tail())
    print("\nInfo:")
    print(gold_data.info())
    print("\nMissing values:\n", gold_data.isnull().sum())
    print("\nStatistical summary:\n", gold_data.describe())

    # 2. GLD distribution
    print("\n[2/7] Generating EDA plots...")
    plt.figure(figsize=(8, 5))
    sns.histplot(gold_data["GLD"], kde=True, color="goldenrod")
    plt.title("GLD (Gold Price) Distribution")
    plt.xlabel("GLD Price")
    plt.tight_layout()
    plt.savefig("gld_distribution.png", dpi=150)
    print("Saved: gld_distribution.png")
    plt.close()

    # Correlation heatmap (numeric only)
    numeric = gold_data.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        square=True,
        cbar=True,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("Saved: correlation_heatmap.png")
    plt.close()

    # Pairplot of key features (sample for speed if large)
    plot_df = gold_data[["SPX", "GLD", "USO", "SLV", "EUR/USD"]].copy()
    g = sns.pairplot(plot_df, diag_kind="kde", corner=True, plot_kws={"alpha": 0.4, "s": 12})
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    plt.savefig("pairplot.png", dpi=120, bbox_inches="tight")
    print("Saved: pairplot.png")
    plt.close()

    # SPX vs GLD
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="SPX", y="GLD", data=gold_data, alpha=0.5, color="darkgoldenrod")
    plt.title("SPX vs GLD")
    plt.tight_layout()
    plt.savefig("spx_vs_gld.png", dpi=150)
    print("Saved: spx_vs_gld.png")
    plt.close()

    # 3. Split
    print("\n[3/7] Train / Test split (80/20)...")
    X = gold_data.drop(["Date", "GLD"], axis=1)
    Y = gold_data["GLD"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 4. Train
    print("\n[4/7] Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    print("Model trained successfully!")

    # 5. Evaluate
    print("\n[5/7] Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    r2_train = metrics.r2_score(Y_train, train_pred)
    r2_test = metrics.r2_score(Y_test, test_pred)
    mae_train = metrics.mean_absolute_error(Y_train, train_pred)
    mae_test = metrics.mean_absolute_error(Y_test, test_pred)
    mse_test = metrics.mean_squared_error(Y_test, test_pred)
    rmse_test = np.sqrt(mse_test)

    print(f"\n--- Training ---")
    print(f"R² Score : {r2_train:.4f}")
    print(f"MAE      : {mae_train:.4f}")
    print(f"\n--- Test ---")
    print(f"R² Score : {r2_test:.4f}")
    print(f"MAE      : {mae_test:.4f}")
    print(f"MSE      : {mse_test:.4f}")
    print(f"RMSE     : {rmse_test:.4f}")

    # 6. Plots: Actual vs Predicted
    print("\n[6/7] Generating prediction & importance plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(Y_train, train_pred, alpha=0.5, color="steelblue", s=15)
    axes[0].plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual GLD")
    axes[0].set_ylabel("Predicted GLD")
    axes[0].set_title(f"Train – Actual vs Predicted (R²={r2_train:.3f})")

    axes[1].scatter(Y_test, test_pred, alpha=0.5, color="darkorange", s=15)
    axes[1].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--", lw=2)
    axes[1].set_xlabel("Actual GLD")
    axes[1].set_ylabel("Predicted GLD")
    axes[1].set_title(f"Test – Actual vs Predicted (R²={r2_test:.3f})")

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    print("Saved: actual_vs_predicted.png")
    plt.close()

    # Residuals
    residuals = Y_test - test_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="slateblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Test Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig("residuals.png", dpi=150)
    print("Saved: residuals.png")
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="goldenrod")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved: feature_importance.png")
    plt.close()

    print("\nFeature Importance Ranking:")
    print(importances.sort_values(ascending=False).to_string())

    # 7. Sample prediction
    print("\n[7/7] Sample prediction...")
    sample = X_test.iloc[[0]]
    actual = Y_test.iloc[0]
    pred = model.predict(sample)[0]
    print(f"Features:\n{sample.to_string(index=False)}")
    print(f"Actual GLD     : {actual:.4f}")
    print(f"Predicted GLD  : {pred:.4f}")
    print(f"Absolute Error : {abs(actual - pred):.4f}")

    print("\n" + "=" * 60)
    print("Done! Generated plots:")
    print("  - gld_distribution.png")
    print("  - correlation_heatmap.png")
    print("  - pairplot.png")
    print("  - spx_vs_gld.png")
    print("  - actual_vs_predicted.png")
    print("  - residuals.png")
    print("  - feature_importance.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
