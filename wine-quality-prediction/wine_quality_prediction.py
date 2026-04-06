

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)

# ── Plotting style ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#FFD700",
    "axes.labelcolor":  "#FFD700",
    "xtick.color":      "#FFD700",
    "ytick.color":      "#FFD700",
    "text.color":       "#FFD700",
    "grid.color":       "#333333",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})
GOLD   = "#FFD700"
DARK   = "#0d0d0d"
ACCENT = "#e0a800"


df = pd.read_csv("/mnt/user-data/uploads/WineQT.csv")
df.drop(columns=["Id"], inplace=True, errors="ignore")

print("=" * 55)
print("  WINE QUALITY PREDICTION — PROJECT REPORT")
print("=" * 55)
print(f"\n[1] Dataset Shape : {df.shape}")
print(f"    Columns       : {list(df.columns)}")
print(f"\n[2] Missing Values:\n{df.isnull().sum()}")
print(f"\n[3] Quality Distribution:\n{df['quality'].value_counts().sort_index()}")
print(f"\n[4] Descriptive Statistics:\n{df.describe().round(3)}")


fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK)
counts = df["quality"].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color=GOLD, edgecolor=DARK, width=0.6)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(int(bar.get_height())), ha="center", va="bottom",
            color=GOLD, fontsize=10, fontweight="bold")
ax.set_title("Wine Quality Distribution", fontsize=14, fontweight="bold", color=GOLD, pad=12)
ax.set_xlabel("Quality Score", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("/home/claude/plot_01_quality_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Saved: plot_01_quality_distribution.png")


fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(DARK)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(10, 50, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            linewidths=0.5, linecolor="#333", ax=ax,
            annot_kws={"size": 8, "color": "white"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", color=GOLD, pad=12)
plt.tight_layout()
plt.savefig("/home/claude/plot_02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Saved: plot_02_correlation_heatmap.png")

# ── 2c. Key Features vs Quality (Box plots) ─────────────────
features_to_plot = ["alcohol", "volatile acidity", "sulphates", "citric acid",
                    "density", "fixed acidity"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle("Key Chemical Features vs Wine Quality", fontsize=14, fontweight="bold",
             color=GOLD, y=1.01)
palette = sns.color_palette("YlOrBr", n_colors=df["quality"].nunique())
for ax, feat in zip(axes.flat, features_to_plot):
    sns.boxplot(x="quality", y=feat, data=df, palette=palette, ax=ax,
                flierprops=dict(marker="o", markerfacecolor=GOLD, markersize=3))
    ax.set_title(feat.title(), fontsize=10, color=GOLD)
    ax.set_xlabel("Quality", fontsize=9)
    ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/plot_03_features_vs_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Saved: plot_03_features_vs_quality.png")

# ── 2d. Feature Distributions (Histograms) ──────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", color=GOLD)
for ax, col in zip(axes.flat, df.columns):
    ax.hist(df[col], bins=30, color=GOLD, edgecolor=DARK, alpha=0.85)
    ax.set_title(col, fontsize=9, color=GOLD)
    ax.grid(axis="y", alpha=0.3)
for ax in axes.flat[len(df.columns):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig("/home/claude/plot_04_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Saved: plot_04_feature_distributions.png")


X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[5] Train samples : {X_train.shape[0]}")
print(f"    Test  samples : {X_test.shape[0]}")


models = {
    "Random Forest":              RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Stochastic Gradient Descent": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
    "Support Vector Classifier":  SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
}

results    = {}
cm_objects = {}

for name, model in models.items():
    # SVC and SGD use scaled data; RF can use raw
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    results[name]    = {"accuracy": acc, "report": report, "y_pred": y_pred}
    cm_objects[name] = cm

    print(f"\n{'─'*55}")
    print(f"  MODEL : {name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))


rf_model = models["Random Forest"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(DARK)
colors = [GOLD if v == feat_imp.max() else ACCENT for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax, color=colors, edgecolor=DARK)
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold",
             color=GOLD, pad=12)
ax.set_xlabel("Importance Score", fontsize=10)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("/home/claude/plot_05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Saved: plot_05_feature_importance.png")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", color=GOLD)
cmap_cm = plt.cm.YlOrBr
for ax, (name, cm) in zip(axes, cm_objects.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, cmap=cmap_cm)
    ax.set_title(name, fontsize=10, color=GOLD, pad=8)
    ax.set_facecolor("#1a1a1a")
    for text in ax.texts:
        text.set_color("white")
plt.tight_layout()
plt.savefig("/home/claude/plot_06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Saved: plot_06_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK)
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
short = ["Random\nForest", "SGD", "SVC"]
bars  = ax.bar(short, accs, color=[GOLD, "#c9a800", "#8c7200"], edgecolor=DARK, width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{acc:.2f}%", ha="center", va="bottom",
            color=GOLD, fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", color=GOLD, pad=12)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("/home/claude/plot_07_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Saved: plot_07_model_comparison.png")


metrics_summary = []
for name, res in results.items():
    rpt = res["report"]
    metrics_summary.append({
        "Model":     name,
        "Accuracy":  round(res["accuracy"] * 100, 2),
        "Macro Precision": round(rpt["macro avg"]["precision"] * 100, 2),
        "Macro Recall":    round(rpt["macro avg"]["recall"] * 100, 2),
        "Macro F1":        round(rpt["macro avg"]["f1-score"] * 100, 2),
    })

summary_df = pd.DataFrame(metrics_summary).set_index("Model")
print("\n\n[6] Final Summary Table:")
print(summary_df.to_string())

# ── Save summary CSV ─────────────────────────────────────────
summary_df.to_csv("/home/claude/model_summary.csv")
print("\n[✓] Saved: model_summary.csv")
print("\n[✓] All tasks completed successfully!\n")

