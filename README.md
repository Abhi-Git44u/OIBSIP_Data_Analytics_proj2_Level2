# 🍷 Wine Quality Prediction
### Data Analytics Project — Level 2

> Predicting wine quality from chemical characteristics using three machine learning classifiers: **Random Forest**, **Stochastic Gradient Descent (SGD)**, and **Support Vector Classifier (SVC)**.

---

## 📌 Project Overview

This project applies supervised machine learning to predict the quality score of red wine (scale 3–8) based on 11 chemical features such as acidity, sulphates, alcohol content, and density. It demonstrates end-to-end data analysis — from EDA and visualization to model training, evaluation, and comparison.

---

## 📁 Project Structure

```
wine-quality-prediction/
│
├── wine_quality_prediction.py     # Main script (EDA + Models + Evaluation)
├── WineQT.csv                     # Dataset (1,143 wine samples)
├── model_summary.csv              # Final accuracy & metrics table
│
├── plot_01_quality_distribution.png
├── plot_02_correlation_heatmap.png
├── plot_03_features_vs_quality.png
├── plot_04_feature_distributions.png
├── plot_05_feature_importance.png
├── plot_06_confusion_matrices.png
├── plot_07_model_comparison.png
│
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | WineQT.csv (Red Wine Dataset) |
| **Samples** | 1,143 rows |
| **Features** | 11 chemical attributes |
| **Target** | `quality` (integer score: 3 to 8) |
| **Missing Values** | None |

### Features

| Feature | Description |
|---|---|
| `fixed acidity` | Tartaric acid concentration |
| `volatile acidity` | Acetic acid — high values lead to vinegar taste |
| `citric acid` | Adds freshness and flavor |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt concentration |
| `free sulfur dioxide` | Free SO₂ — prevents microbial growth |
| `total sulfur dioxide` | Total SO₂ (free + bound) |
| `density` | Density of wine (close to water) |
| `pH` | Acidity level (0–14 scale) |
| `sulphates` | Additive contributing to SO₂ levels |
| `alcohol` | Alcohol percentage by volume |

### Quality Distribution

| Quality Score | Count |
|---|---|
| 3 | 6 |
| 4 | 33 |
| 5 | 483 |
| 6 | 462 |
| 7 | 143 |
| 8 | 16 |

> **Note:** The dataset is imbalanced — most wines fall in the 5–6 quality range.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Language** | Python 3.x |

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project

```bash
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the project

```bash
python wine_quality_prediction.py
```

All plots and the summary CSV will be generated in the same directory.

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)
- Checked for null values (none found)
- Analyzed class distribution of `quality`
- Computed descriptive statistics
- Generated correlation heatmap to identify multicollinearity

### Step 2 — Visualizations
- **Quality Distribution** — bar chart of wine quality scores
- **Correlation Heatmap** — feature-to-feature and feature-to-target correlations
- **Box Plots** — key features vs quality to spot trends
- **Histograms** — individual distribution of every feature

### Step 3 — Preprocessing
- Dropped the `Id` column (non-informative)
- Split data: **80% train / 20% test** with stratification
- Applied `StandardScaler` for SGD and SVC models
- Random Forest used raw (unscaled) features

### Step 4 — Model Training

| Model | Key Parameters |
|---|---|
| Random Forest | `n_estimators=200`, `random_state=42` |
| SGD Classifier | `max_iter=1000`, `tol=1e-3` |
| SVC | `kernel=rbf`, `C=10`, `gamma=scale` |

### Step 5 — Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1 per class)
- Macro-averaged metrics
- Confusion Matrix for each model
- Feature Importance (Random Forest)

---

## 📈 Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Random Forest** | **71.62%** | 35.53% | 34.44% | 34.58% |
| Support Vector Classifier | 64.19% | 31.82% | 31.08% | 31.17% |
| Stochastic Gradient Descent | 57.64% | 26.43% | 24.88% | 24.58% |

> ✅ **Random Forest is the best-performing model** with 71.62% accuracy on the test set.

---

## 🔍 Key Insights

- **`alcohol`** is the most important predictor of wine quality — higher alcohol generally correlates with higher quality scores.
- **`volatile acidity`** has a strong negative correlation with quality — wines with high volatile acidity tend to score lower.
- **`sulphates`** and **`citric acid`** are also significant positive contributors.
- The dataset is class-imbalanced (majority of wines are scored 5 or 6), which limits macro-averaged performance across minority classes (3, 4, 8).
- Random Forest outperforms the other two models because it handles non-linear relationships and feature interactions naturally.

---

## 📉 Visualizations Produced

| File | Description |
|---|---|
| `plot_01_quality_distribution.png` | Bar chart of wine quality score counts |
| `plot_02_correlation_heatmap.png` | Heatmap of feature correlations |
| `plot_03_features_vs_quality.png` | Box plots — top 6 features vs quality |
| `plot_04_feature_distributions.png` | Histograms of all 11 features |
| `plot_05_feature_importance.png` | Random Forest feature importance ranking |
| `plot_06_confusion_matrices.png` | Confusion matrices for all 3 models |
| `plot_07_model_comparison.png` | Accuracy bar chart — model comparison |

---

## 🚀 Possible Improvements

- **Handle class imbalance** using SMOTE or class weighting to improve minority class prediction
- **Hyperparameter tuning** with GridSearchCV or RandomizedSearchCV
- **Feature engineering** — create composite features like acidity ratio
- **Ensemble stacking** — combine predictions of all three models
- **Binary classification** — simplify to Good (≥7) vs Not Good (<7) for better accuracy
- **Deep Learning** — try a simple neural network with PyTorch or TensorFlow

---

## 👤 Author

**Abhishek**
Data Analytics — Project 2 | Level 2

---

## 📄 License

This project is for academic and educational purposes.
