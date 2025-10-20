from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score

def train_model(X_train, y_train):
    """
    Trains a soft voting ensemble classifier: Logistic Regression, XGBoost, CatBoost
    """
# ==== 1. Define models ====
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=559,
        verbose=0,
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=559,  # 170579 / 305
        eval_metric='logloss',
        random_state=42
    )



    lr = LogisticRegression(
        class_weight={0:1, 1:15},
        solver='liblinear',
        random_state=42
    )
    # ==== 2. Combine in a soft-voting ensemble ====
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('xgb', xgb), ('cat', cat)],
        voting='soft'
    )


    voting_clf.fit(X_train, y_train)
    return voting_clf

def evaluate_model(model, X_val, y_val, X_test, y_test, threshold):
    """
    Evaluates the model with a fixed threshold on validation and test sets
    """
    y_proba_val = model.predict_proba(X_val)[:,1]
    y_proba_test = model.predict_proba(X_test)[:,1]

    y_pred_val = (y_proba_val >= threshold).astype(int)
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print("\n--- Validation Set ---")
    print(classification_report(y_val, y_pred_val))
    print("Validation ROC-AUC:", roc_auc_score(y_val, y_proba_val))

    print("\n--- Test Set ---")
    print(classification_report(y_test, y_pred_test))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_proba_test))

    return {
        "val_roc_auc": roc_auc_score(y_val, y_proba_val),
        "test_roc_auc": roc_auc_score(y_test, y_proba_test),
        "best_threshold": threshold
    }
