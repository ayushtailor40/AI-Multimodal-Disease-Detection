import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv(
    "dataset/clinical/clinical_data.csv"
)

print("Clinical Dataset:")
print(data)


# ============================================
# 2. SELECT FEATURES
# ============================================

features = [
    "age",
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "cough",
    "breathing_difficulty"
]

X = data[features]
y = data["label"]


# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# 4. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================
# 6. PREDICTION
# ============================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================
# 7. EVALUATION
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nClinical ML Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================
# 8. SAVE MODEL
# ============================================

with open(
    "models/clinical_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


# Save scaler
with open(
    "models/clinical_scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print("\nClinical model saved successfully!")
