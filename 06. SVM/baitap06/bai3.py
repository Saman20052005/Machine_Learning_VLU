# -*- coding: utf-8 -*-
"""Bài 3: Kernel đa thức bậc 2, 3, 4 trên bộ hai vầng trăng."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print("Bac   C tot nhat   CV accuracy   Test accuracy")
    ket_qua = []
    for d in [2, 3, 4]:
        pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=d))
        tim = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10, 100]}, cv=5)
        tim.fit(X_tr, y_tr)
        acc_te = tim.score(X_te, y_te)
        c_tot = tim.best_params_["svc__C"]
        ket_qua.append((d, c_tot, tim.best_score_, acc_te))
        print(f" {d}      {c_tot:6}       {tim.best_score_:.4f}        {acc_te:.4f}")
    print()

    tot = max(ket_qua, key=lambda x: x[2])
    print(f"Bac tot nhat theo CV: bac {tot[0]} (C={tot[1]}), CV={tot[2]:.4f}, test={tot[3]:.4f}")
    print("So voi kernel RBF o muc 5 (CV=0.8711, test=0.9200):")
    print("- Kernel da thuc khong vuot duoc kernel RBF tren bo hai vang trang.")


if __name__ == "__main__":
    main()
