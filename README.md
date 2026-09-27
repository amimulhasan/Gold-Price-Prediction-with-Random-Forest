# 🥇 Gold Price Prediction using Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete machine learning project that predicts **gold prices (GLD)** using a **Random Forest Regressor**, with full EDA, visualizations, metrics, and feature importance analysis.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Algorithm** | Random Forest Regressor |
| **Dataset** | Gold Price Dataset (~2290 samples) |
| **Target** | `GLD` – Gold ETF price |
| **Train/Test Split** | 80% / 20% |
| **Typical Test R²** | ~0.98+ |

### Features
| Feature | Description |
|---------|-------------|
| `SPX` | S&P 500 index |
| `USO` | United States Oil Fund |
| `SLV` | Silver Trust (iShares) |
| `EUR/USD` | Euro to US Dollar exchange rate |

**Target:** `GLD` – SPDR Gold Shares price

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/gold-price-prediction.git
cd gold-price-prediction

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# Notebook
jupyter notebook Project_6_Gold_Price_Prediction.ipynb

# Full script (all figures + metrics)
python gold_price_prediction.py
```

---

## 📈 What This Project Includes (A–Z)

### Data & Preprocessing
- Load `gld_price_data.csv`
- Head / tail / shape / info / missing values
- Statistical summary

### Visualizations
- GLD price distribution
- Correlation heatmap
- Pairwise relationships (key features)
- Actual vs Predicted scatter (train & test)
- Feature importance bar chart
- Residual distribution

### Model & Metrics
- Random Forest Regressor
- R² Score (train & test)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE) / RMSE
- Feature importance ranking

### Predictive System
- Sample prediction pipeline

---

## 📁 Project Structure

```
gold-price-prediction/
├── Project_6_Gold_Price_Prediction.ipynb
├── gold_price_prediction.py
├── gld_price_data.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Technologies

- Python 3.8+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (RandomForestRegressor, metrics)

---

## 📝 License

MIT License – free to use and modify.

---

**Made with ❤️ for learning Machine Learning**
