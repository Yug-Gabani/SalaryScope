import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

from xgboost import XGBClassifier, XGBRegressor
from preprocess import preprocess_data

# Create models directory
os.makedirs("ml/models", exist_ok=True)

# Load & preprocess data
X, y, label_encoder = preprocess_data(
    "college_student_placement_dataset.csv",
    is_training=True
)

# Save feature names
joblib.dump(list(X.columns), "ml/models/feature_names.pkl")


# 1️⃣ Train Placement Model (Classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

placement_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

placement_model.fit(X_train, y_train)

# Save placement model
joblib.dump(placement_model, "ml/models/placement_model.pkl")
joblib.dump(label_encoder, "ml/models/label_encoder.pkl")

# Evaluate placement model
y_pred = placement_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Placement Model Accuracy: {accuracy * 100:.2f}%")

# 2️⃣ Train Salary Model (Regression)
# We only train salary model on placed students
salary_data = X.copy()
salary_data["Placement"] = y

# Keep only placed students
placed_data = salary_data[salary_data["Placement"] == 1]

X_salary = placed_data.drop(columns=["Salary", "Placement"])
y_salary = placed_data["Salary"]

X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

salary_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

salary_model.fit(X_train_sal, y_train_sal)

# Save salary model
joblib.dump(salary_model, "ml/models/salary_model.pkl")

# Evaluate salary model
y_pred_sal = salary_model.predict(X_test_sal)
mae = mean_absolute_error(y_test_sal, y_pred_sal)

print(f"✅ Salary Model MAE: ₹{mae:,.2f}")

print("\n🎉 Training complete. Models saved in ml/models/")
