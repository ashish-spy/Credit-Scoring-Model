import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/german_credit_data.csv")

print("\nDataset Loaded Successfully!\n")

print(df.head())

# ==============================
# Features and Target
# ==============================

X = df.drop("kredit", axis=1)
y = df["kredit"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Models
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
}

best_model = None
best_accuracy = 0

# Store model results
results = []

print("\n==============================")
print("Model Evaluation")
print("==============================\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    roc = roc_auc_score(y_test, probability)

    print("=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(roc, 4))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, prediction))

print()

# Store model results
results.append({
    "Model": name,
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1 Score": round(f1, 4),
    "ROC-AUC": round(roc, 4)
})

# Select best model
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model = model

# ==============================
# Model Comparison Table
# ==============================

results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("=" * 70)
print(results_df)

# Save comparison table
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ==============================
# Save Best Model
# ==============================

joblib.dump(best_model, "models/credit_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("=" * 50)
print("Best Model Saved Successfully!")
print("Accuracy:", round(best_accuracy, 4))
print("=" * 50)

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("outputs/confusion_matrix.png")
plt.close()


# ==============================
# ROC Curve
# ==============================

RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.savefig("outputs/roc_curve.png")
plt.close()


# ==============================
# Feature Importance
# ==============================

if hasattr(best_model, "feature_importances_"):

    feature_names = X.columns

    importance = best_model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.bar(feature_names, importance)

    plt.xticks(rotation=90)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")

    plt.close()

print("\nGraphs saved successfully!")