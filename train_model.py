
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.metrics import precision_recall_curve, f1_score


# Load datasets
train_data = pd.read_csv("trainC.csv")
val_data = pd.read_csv("valC.csv")
test_data = pd.read_csv("testC.csv")

# Original selected features
selected_features = [
    "V17", "V14", "V12", "V10", "V16",
    "V7", "V3", "V11", "V4", "V18"
]

# Separate features and target
X_train = train_data[selected_features]
y_train = train_data["Class"]

X_val = val_data[selected_features]
y_val = val_data["Class"]

X_test = test_data[selected_features]
y_test = test_data["Class"]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

print("\nFraud distribution:")
print("Train:", y_train.value_counts().to_dict())
print("Validation:", y_val.value_counts().to_dict())
print("Test:", y_test.value_counts().to_dict())




# Oversampling
over = SMOTE(
    sampling_strategy=0.05,
    random_state=42
)

X_res, y_res = over.fit_resample(X_train, y_train)

# Undersampling
under = RandomUnderSampler(
    sampling_strategy=0.2,
    random_state=42
)

X_res, y_res = under.fit_resample(X_res, y_res)

print("\nBefore resampling:", Counter(y_train))
print("After resampling:", Counter(y_res))




scaler = MinMaxScaler()

# Fit on original training data only
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Scale resampled training data using the same scaler
X_res_scaled = scaler.transform(X_res)

print("\nScaled shapes:")
print("Train:", X_train_scaled.shape)
print("Resampled Train:", X_res_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)




# Define models
cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

lr = LogisticRegression(
    solver="liblinear",
    random_state=42,
    max_iter=1000
)

# Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("xgb", xgb),
        ("cat", cat)
    ],
    voting="soft"
)

# Train on resampled and scaled training data
voting_clf.fit(X_res_scaled, y_res)

print("Ensemble training completed successfully!")




# Validation probabilities
y_val_proba = voting_clf.predict_proba(X_val_scaled)[:, 1]

# Default threshold = 0.5
y_val_pred = (y_val_proba >= 0.5).astype(int)

print("\n--- Validation Results (Threshold = 0.5) ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

print("ROC-AUC:", roc_auc_score(y_val, y_val_proba))
print("PR-AUC:", average_precision_score(y_val, y_val_proba))



# Calculate precision-recall curve on validation
precision, recall, thresholds = precision_recall_curve(
    y_val,
    y_val_proba
)

# Calculate F1 for each threshold
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

# Find best threshold
best_idx = np.argmax(f1_scores)

best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

print("\n--- Threshold Optimization ---")
print("Best Threshold:", best_threshold)
print("Best Validation F1:", best_f1)


from sklearn.metrics import classification_report, confusion_matrix

y_val_pred_optimized = (
    y_val_proba >= best_threshold
).astype(int)

print("\n--- Optimized Validation Results ---")

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_optimized))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_val_pred_optimized,
    zero_division=0
))


# Final evaluation on untouched test set
y_test_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    zero_division=0
))

print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))
print("PR-AUC:", average_precision_score(y_test, y_test_proba))

import pickle

artifacts = {
    "model": voting_clf,
    "scaler": scaler,
    "best_threshold": best_threshold,
    "selected_features": selected_features
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nModel artifacts saved successfully!")