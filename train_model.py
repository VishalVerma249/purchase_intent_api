import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- 1. Environment & Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "online_shoppers_intention.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")

# --- 2. Data Loading ---
df = pd.read_csv(CSV_PATH)

# --- 3. Preprocessing ---
df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)

le_month = LabelEncoder()
df["Month"] = le_month.fit_transform(df["Month"])

le_visitor = LabelEncoder()
df["VisitorType"] = le_visitor.fit_transform(df["VisitorType"])

# --- 4. Data Splitting ---
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Model Training ---
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=5,
)
model.fit(X_train, y_train)

# --- 6. Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# --- 7. Model Saving ---
model_dict = {
    "model": model,
    "le_month": le_month,
    "le_visitor": le_visitor,
    "feature_columns": X.columns.tolist(),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_dict, f)

print(f"Model successfully saved to: {MODEL_PATH}")
