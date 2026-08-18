from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

DATA_FILE = DATA_DIR / "breast_cancer_dataset.csv"
TEST_FILE = ROOT / "test_data.csv"
RESULT_FILE = MODEL_DIR / "evaluation_results.csv"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

if "target" not in df.columns:
    raise ValueError("Dataset must contain a 'target' column.")

X = df.drop(columns=["target"])
y = df["target"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# SAVE HELD-OUT TEST DATA
# ============================================================

test_data = X_test.copy()
test_data["target"] = y_test.values

test_data.to_csv(
    TEST_FILE,
    index=False,
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),

    "kNN": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=7,
                ),
            ),
        ]
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    result = {
        "ML Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            y_test,
            predictions,
        ),
    }

    results.append(result)

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / filename,
    )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")
print()
print(f"Training records : {len(X_train)}")
print(f"Test records     : {len(X_test)}")
print(f"Features         : {X.shape[1]}")
print()
print("Generated:")
print(f"  - {TEST_FILE.name}")
print(f"  - {RESULT_FILE.name}")
print("  - 5 trained model files")
print()
print(results_df.to_string(index=False))