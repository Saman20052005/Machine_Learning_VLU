# -*- coding: utf-8 -*-
"""Bài 4: Dò lưới mịn quanh cặp tốt nhất (C=10, gamma=1) của mục 5."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
    tim = GridSearchCV(SVC(kernel="rbf"), luoi, cv=5)
    tim.fit(X_tr, y_tr)

    bang = pd.DataFrame(tim.cv_results_["params"])
    bang["diem"] = tim.cv_results_["mean_test_score"]
    print("Diem kiem dinh cheo luoi min (hang C, cot gamma):")
    print(bang.pivot(index="C", columns="gamma", values="diem").round(4).to_string())
    print()
    print("Cap tot nhat:", tim.best_params_)
    print(f"Diem CV tot nhat : {tim.best_score_:.4f} (luoi thua muc 5: 0.8711)")
    print(f"Accuracy tap test: {tim.score(X_te, y_te):.4f} (luoi thua muc 5: 0.9200)")
    print()
    print("Nhan xet:")
    print("- Do luoi min giup tim duoc diem toi uu sat hon trong vung tot, cai thien diem CV;")
    print("  tuy nhien tren tap kiem tra nho (75 mau), accuracy co the giu nguyen hoac thay doi it.")


if __name__ == "__main__":
    main()
