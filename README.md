# 🎓 Student Placement and Salary Prediction System

## 📌 Project Overview

This project uses **Machine Learning (XGBoost)** to:

* Predict whether a student will be **Placed or Not Placed**
* Predict the **Expected Salary** if the student is placed

It uses:

* **XGBClassifier** → Placement Prediction
* **XGBRegressor** → Salary Prediction

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

* Python 3.11
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Joblib
* Matplotlib

---

## 📊 Features Used

* IQ
* CGPA
* Previous Semester Result
* Academic Performance
* Communication Skills
* Extra Curricular Score
* Internship Experience

---

## 🤖 Machine Learning Models

### 1️⃣ Placement Model

* Algorithm: XGBClassifier
* Output: Yes / No
* Accuracy: **99.95%**

---

### 2️⃣ Salary Model

* Algorithm: XGBRegressor
* Output: Salary Amount
* MAE: **₹12,000**

---

## 🚀 How to Run Project

### Step 1 Install Libraries

```
pip install pandas numpy scikit-learn xgboost joblib matplotlib
```

---

### Step 2 Train Model

```
python train.py
```

This creates:

```
placement_model.pkl
salary_model.pkl
```

---

### Step 3 Run Prediction

```
python predict.py
```

Enter student details.

Example Output:

```
Placement: Yes
Salary: 664454
```

---

## 📈 Output Files

| File                   | Description                |
| ---------------------- | -------------------------- |
| placement_model.pkl    | Placement prediction model |
| salary_model.pkl       | Salary prediction model    |
| salary_importance.png  | Feature importance graph   |
| metrics_comparison.txt | Model performance report   |

---

## 📷 Feature Importance

Shows which features affect salary most:

* CGPA → Highest
* IQ → High
* Internship → Medium

---

## 🎯 Project Objective

Help colleges predict:

* Student placement chances
* Expected salary

Useful for:

* Students
* Colleges
* Placement departments

---

## 👨‍💻 Author

**Name:** Yug Gabani

**Course:** B.Tech / BE (AI / ML / IT / CSE)

**Year:** 2026

---

## ✅ Future Improvements

* Add Web App (Streamlit)
* Add Real Dataset
* Improve Accuracy to 95%
* Deploy Online

---

## ⭐ Conclusion

This project successfully predicts:

* Placement Status
* Salary

Using Machine Learning.

---
