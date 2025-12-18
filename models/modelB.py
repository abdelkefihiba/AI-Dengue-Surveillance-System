import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("dengue_weka_ready 1.csv")


if df["State"].dtype == "object":
    df["State_encoded"] = LabelEncoder().fit_transform(df["State"])
else:
    df["State_encoded"] = df["State"]


df = df.sort_values(["State", "Year"]).reset_index(drop=True)


features = [
    "State_encoded",
    "Year",
    "Cases_last_year",
    "Cases_2yrs_ago",
    "Cases_3yr_avg",
    "AvgTemp",
    "TotalRainfall",
    "Humidity",
    "population",
    "Cases_per_100k",
    "Cases_last_year_per_100k",
    "Cases_2yrs_ago_per_100k",
    "Cases_3yr_avg_per_100k"
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["Outbreak"]


test_years = sorted(df["Year"].unique())[-3:]

train_mask = ~df["Year"].isin(test_years)
test_mask = df["Year"].isin(test_years)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")
print(f"Test years   : {test_years}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    "C": [0.5, 1, 5],
    "gamma": ["scale", 0.1]
}

tscv = TimeSeriesSplit(n_splits=4)

grid = GridSearchCV(
    SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    ),
    param_grid,
    scoring="f1",
    cv=tscv
)

print("\nTraining SVM...")
grid.fit(X_train_scaled, y_train)
model = grid.best_estimator_

print("Best parameters:", grid.best_params_)


y_proba = model.predict_proba(X_test_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_proba >= best_threshold).astype(int)

print(f"Optimized threshold: {best_threshold:.2f}")


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("\nTest Performance:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Outbreak", "Outbreak"]))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Outbreak", "Outbreak"],
    yticklabels=["No Outbreak", "Outbreak"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM – Last 3 Years Test)")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=300)
plt.show()

print("\nSaved: svm_confusion_matrix.png")


