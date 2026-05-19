import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("disease_data.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

# =========================
# REMOVE MISSING VALUES
# =========================
df = df.dropna()

# =========================
# TARGET COLUMN
# =========================
target_column = "Outcome"

# =========================
# FEATURES & TARGET
# =========================
X = df.drop(target_column, axis=1)
y = df[target_column]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # TRAIN
    model.fit(X_train, y_train)

    # PREDICT
    predictions = model.predict(X_test)

    # METRICS
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# RESULTS
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
