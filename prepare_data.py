
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("creditcard.csv")

df["source_id"] = df.index

# Separate features and target
X = df.drop(["Class", "source_id"], axis=1)
y = df["Class"]
# 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# Save splits
train_data = X_train.copy()
train_data["Class"] = y_train

val_data = X_val.copy()
val_data["Class"] = y_val

test_data = X_test.copy()
test_data["Class"] = y_test

train_data["source_id"] = X_train.index
val_data["source_id"] = X_val.index
test_data["source_id"] = X_test.index

train_data.to_csv("trainC.csv", index=False)
val_data.to_csv("valC.csv", index=False)
test_data.to_csv("testC.csv", index=False)

print("Train:", train_data.shape, "| Fraud:", y_train.sum())
print("Validation:", val_data.shape, "| Fraud:", y_val.sum())
print("Test:", test_data.shape, "| Fraud:", y_test.sum())
