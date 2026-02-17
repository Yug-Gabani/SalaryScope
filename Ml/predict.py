import joblib
import pandas as pd
from preprocess import preprocess_data

# Load Models
placement_model = joblib.load("ml/models/placement_model.pkl")
salary_model = joblib.load("ml/models/salary_model.pkl")
label_encoder = joblib.load("ml/models/label_encoder.pkl")
feature_names = joblib.load("ml/models/feature_names.pkl")

def get_valid_input(prompt, min_val=None, max_val=None, is_float=False):
    while True:
        try:
            value = float(input(prompt)) if is_float else int(input(prompt))

            if min_val is not None and value < min_val:
                print(f"❌ Value must be >= {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"❌ Value must be <= {max_val}")
                continue

            return value

        except ValueError:
            print("❌ Invalid input. Try again.")



print("\n🎓 Enter student details for prediction\n")

IQ = get_valid_input("🧠 IQ (80-160): ", 80, 160)
CGPA = get_valid_input("📊 CGPA (0-10): ", 0, 10, is_float=True)
Prev_Sem_Result = get_valid_input("📘 Previous Semester % (0-100): ", 0, 100)
Academic_Performance = get_valid_input("📚 Academic Performance (1-10): ", 1, 10)
Communication_Skills = get_valid_input("🗣 Communication Skills (1-10): ", 1, 10)
Extra_Curricular_Score = get_valid_input("🏆 Extra Curricular (1-10): ", 1, 10)
Internship_Experience = input("💼 Internship Experience (Yes/No): ").strip().lower()
while Internship_Experience not in ["yes", "no"]:
    print("❌ Enter only Yes or No")
    Internship_Experience = input("💼 Internship Experience (Yes/No): ").strip().lower()

Internship_Experience = 1 if Internship_Experience == "yes" else 0

# Create Input DataFrame
data = {
    "College_ID": [1],
    "IQ": [IQ],
    "CGPA": [CGPA],
    "Prev_Sem_Result": [Prev_Sem_Result],
    "Academic_Performance": [Academic_Performance],
    "Communication_Skills": [Communication_Skills],
    "Extra_Curricular_Score": [Extra_Curricular_Score],
    "Internship_Experience": [Internship_Experience]
}

df = pd.DataFrame(data)

df.to_csv("input.csv", index=False)

# Preprocess
X_input = preprocess_data("input.csv", is_training=False)

# Ensure Feature Order Matches
for col in feature_names:
    if col not in X_input.columns:
        X_input[col] = 0

X_input = X_input[feature_names]

# 1️⃣ Placement Prediction
placement_pred = placement_model.predict(X_input)
placement_result = label_encoder.inverse_transform(placement_pred)

print("\n🎯 Placement Prediction:", placement_result[0])

# 2️⃣ Salary Prediction (if placed)
if placement_pred[0] == 1:

    # Salary model does NOT need Salary column
    salary_input = X_input.copy()

    if "Salary" in salary_input.columns:
        salary_input = salary_input.drop(columns=["Salary"])

    predicted_salary = salary_model.predict(salary_input)[0]

    print(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")

else:
    print("💰 Predicted Salary: ₹0 (Not Placed)")
