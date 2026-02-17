import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def add_salary_column(df):

    if "Salary" not in df.columns:

        cgpa_col = "cgpa" if "cgpa" in df.columns else "CGPA"
        iq_col = "iq" if "iq" in df.columns else "IQ"

        df["Salary"] = (
            df[cgpa_col] * 50000
            + df[iq_col] * 2000
            + np.random.normal(0, 10000, len(df))
        )

        if "Placement" in df.columns:
            df.loc[df["Placement"] == 0, "Salary"] = 0

    return df


def preprocess_data(csv_path, is_training=True):

    df = pd.read_csv(csv_path)

    df["Internship_Experience"] = df["Internship_Experience"].map({
        "Yes": 1,
        "No": 0,
        1: 1,
        0: 0
    })

    if is_training:

        label_encoder = LabelEncoder()

        df["Placement"] = label_encoder.fit_transform(df["Placement"])

        df = add_salary_column(df)

        X = df.drop(columns=["Placement", "College_ID"])

        y = df["Placement"]

        return X, y, label_encoder

    else:

        df = add_salary_column(df)

        X = df.drop(columns=["College_ID"])

        return X
