"""Bước 5: GaussianNB và hồi quy logistic trên cùng tập sinh viên."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[["gio_on", "diem_giua_ky"]], df.qua_mon,
        test_size=0.25, random_state=17, stratify=df.qua_mon)
    gnb = GaussianNB()
    for ten, model in [("GaussianNB", gnb), ("LogisticRegression", LogisticRegression())]:
        model.fit(X_train, y_train)
        print(f"{ten}: accuracy = {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print("\nThong so hoc duoc (gio_on, diem_giua_ky):")
    for i, nhan in enumerate(gnb.classes_):
        print(f"Lop {nhan}: trung binh = {gnb.theta_[i].round(4)}")
        print(f"       do lech chuan = {(gnb.var_[i] ** 0.5).round(4)}")
    print("Tien nghiem:", gnb.class_prior_.round(4))
    moi = pd.DataFrame({"gio_on": [5, 13, 20, 28], "diem_giua_ky": [4., 6., 7., 9.]})
    moi["P(qua_mon)"] = gnb.predict_proba(moi)[:, 1]
    print("\nDu doan bon sinh vien moi:")
    print(moi.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
