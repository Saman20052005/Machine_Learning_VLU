# -*- coding: utf-8 -*-
"""Bài 3: Chọn ngưỡng lớn nhất trên tập học để đạt recall >= 0.80."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = make_pipeline(StandardScaler(), LogisticRegression())
    xs_hoc = cross_val_predict(lr, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]

    cac_nguong = np.round(np.arange(0.01, 1.00, 0.01), 2)
    dat_yeu_cau = []
    for t in cac_nguong:
        r_cv = recall_score(y_tr, (xs_hoc >= t).astype(int))
        if r_cv >= 0.80:
            dat_yeu_cau.append((t, r_cv))

    t_tot, r_cv_tot = dat_yeu_cau[-1]  # Ngưỡng lớn nhất đạt recall >= 0.80
    print(f"Nguong lon nhat tren tap hoc dat CV recall >= 0.80: {t_tot:.2f} (CV recall = {r_cv_tot:.4f})")

    lr.fit(X_tr, y_tr)
    xs_te = lr.predict_proba(X_te)[:, 1]
    du_te = (xs_te >= t_tot).astype(int)
    r_te = recall_score(y_te, du_te)
    p_te = precision_score(y_te, du_te)
    so_bao = int(du_te.sum())

    print(f"Ket qua tren tap kiem tra (chi mo 1 lan):")
    print(f"  Recall    = {r_te:.4f} ({'dat >= 0.80' if r_te >= 0.80 else 'chua dat 0.80'})")
    print(f"  Precision = {p_te:.4f}")
    print(f"  So ban bi bao = {so_bao} / {len(y_te)}")


if __name__ == "__main__":
    main()
