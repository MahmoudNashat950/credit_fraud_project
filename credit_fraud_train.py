import argparse
import pickle
from credit_fraud_utils_data import load_data, preprocess_data, balance_data, scale_features
from credit_fraud_utils_eval import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Credit Fraud Detection Training")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output_model", type=str, default="model.pkl", help="Path to save trained model")
    args = parser.parse_args()

    print("Loading data...")
    train_data, val_data, test_data = load_data(args.train_path, args.val_path, args.test_path)

    # Select features based on notebook analysis
    selected_features =['V17', 'V14', 'V12', 'V10', 'V16', 'V7', 'V3', 'V11', 'V4', 'V18', 'Class']

    print("Preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        train_data, val_data, test_data, selected_features
    )

    print("Balancing data...")
    X_res, y_res = balance_data(X_train, y_train)

    print("Scaling features...")
    X_res_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_res, X_val, X_test
    )

    print("Training model...")
    model = train_model(X_res_scaled, y_res)

    # Fixed threshold from notebook
    best_threshold = 0.7445532313170999

    print(f"Evaluating model with fixed threshold {best_threshold}...")
    results = evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test, best_threshold)

    # Save model and scaler
    model_dict = {
        "model": model,
        "scaler": scaler,
        "best_threshold": best_threshold
    }

    with open(args.output_model, "wb") as f:
        pickle.dump(model_dict, f)

    print(f"Model saved to {args.output_model}")
    print(f"Validation ROC-AUC: {results['val_roc_auc']:.3f}, Test ROC-AUC: {results['test_roc_auc']:.3f}")

if __name__ == "__main__":
    main()
