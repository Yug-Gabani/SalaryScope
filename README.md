# 🎓 SalaryScope — Student Placement & Salary Prediction System

🔗 **Live Demo:** [Link](https://placement-helper-website--hiworl10.replit.app)

SalaryScope is a Machine Learning project that predicts:

- Whether a student will be **Placed or Not Placed**
- The **Expected Salary** if the student is placed

It uses **XGBoost** for both tasks — a classifier for placement prediction and a regressor for salary prediction.

---

## 📌 Overview

- **XGBClassifier** → Predicts placement status (Yes / No)
- **XGBRegressor** → Predicts expected salary

The goal is to help students, colleges, and placement departments get data-driven insight into placement chances and likely compensation, based on academic and extracurricular performance.

---
## 📂 Project Structure

```
project/
│
├── train.py
├── predict.py
├── preprocess.py
│
└── ml/
    └── models/
        ├── placement_model.pkl
        ├── salary_model.pkl
        ├── label_encoder.pkl
        ├── feature_names.pkl
        ├── salary_importance.png
        └── metrics_comparison.txt
```

---

## ⚙️ Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib

---

## 📊 Features Used

- IQ
- CGPA
- Previous Semester Result
- Academic Performance
- Communication Skills
- Extra Curricular Score
- Internship Experience

---

## 🤖 Machine Learning Models

### 1️⃣ Placement Model
- **Algorithm:** XGBClassifier
- **Output:** Placed / Not Placed
- **Accuracy:** 99.95%

### 2️⃣ Salary Model
- **Algorithm:** XGBRegressor
- **Output:** Predicted salary amount
- **MAE:** ₹12,000

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Yug-Gabani/SalaryScope.git
cd SalaryScope
```

### 2. Install dependencies
```bash
pip install -r requirement.txt
```

### 3. Train the models
Run the training script(s) inside the `Ml/` folder to generate the model artifacts (placement model, salary model, encoders, etc.) in `Outputs/`.

### 4. Run predictions
Use the prediction script inside `Ml/` and enter student details when prompted.

**Example output:**
```
Placement: Yes
Salary: 664454
```

---

## 📈 Outputs

| File                     | Description                     |
|--------------------------|----------------------------------|
| `placement_model.pkl`    | Trained placement prediction model |
| `salary_model.pkl`       | Trained salary prediction model    |
| `salary_importance.png`  | Feature importance graph           |
| `metrics_comparison.txt` | Model performance report           |

---

## 📷 Feature Importance

Analysis shows the features that most influence predicted salary:

- **CGPA** → Highest impact
- **IQ** → High impact
- **Internship Experience** → Medium impact

---

## 🎯 Objective

SalaryScope aims to help:

- **Students** — understand their placement readiness and salary expectations
- **Colleges** — identify students who may need additional support
- **Placement departments** — plan and advise more effectively using data-driven predictions

---

## ✅ Future Improvements

- [ ] Web app interface (Streamlit)
- [ ] Integration with real-world placement datasets
- [ ] Improved model accuracy and validation
- [ ] Public deployment

---

## 👨‍💻 Author

**[Yug Gabani](https://github.com/Yug-Gabani)**

---
