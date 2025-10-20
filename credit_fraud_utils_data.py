import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def load_data(train_path, val_path, test_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    return train_data, val_data, test_data

def preprocess_data(train_data, val_data, test_data, selected_features=None):
    # Select top features if provided
    if selected_features:
        train_data = train_data[selected_features]
        val_data = val_data[selected_features]
        test_data = test_data[selected_features]

    X_train = train_data.drop("Class", axis=1)
    y_train = train_data["Class"]

    X_val = val_data.drop("Class", axis=1)
    y_val = val_data["Class"]

    X_test = test_data.drop("Class", axis=1)
    y_test = test_data["Class"]

    return X_train, y_train, X_val, y_val, X_test, y_test

def balance_data(X_train, y_train):
    # Hybrid: SMOTE + Undersampling
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    over = SMOTE(sampling_strategy=0.05, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
    X_res, y_res = over.fit_resample(X_train, y_train)
    X_res, y_res = under.fit_resample(X_res, y_res)

    print("After resampling:", Counter(y_res))
    return X_res, y_res

def scale_features(X_train, X_res, X_val, X_test):
    scaler = MinMaxScaler()
    # Fit only on original X_train
    scaler.fit(X_train)

    X_res_scaled = scaler.transform(X_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_res_scaled, X_val_scaled, X_test_scaled, scaler
