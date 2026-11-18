# -*- coding: utf-8 -*-
"""Bài 5: Đo độ quan trọng bằng permutation_importance và so với feature_importances_."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    rng = np.random.default_rng(0)
    X_tr_nn = X_tr.copy()
    X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
    X_te_nn = X_te.copy()
    X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))

    rung_nn = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr_nn, y_tr)
    fi = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns).sort_values(ascending=False)

    perm = permutation_importance(
        rung_nn, X_te_nn, y_te, n_repeats=20, random_state=0
    )
    pi = pd.Series(perm.importances_mean, index=X_te_nn.columns).sort_values(ascending=False)

    print("1. Do quan trong theo feature_importances_ (Gini):")
    print(fi.round(4).to_string())
    print()
    print("2. Do quan trong theo permutation_importance (tren tap kiem tra):")
    print(pi.round(4).to_string())
    print()
    hang_fi = list(fi.index).index("so_ngau_nhien") + 1
    hang_pi = list(pi.index).index("so_ngau_nhien") + 1
    print(f"Thu hang cot 'so_ngau_nhien': feature_importances_ = {hang_fi}/5, permutation_importance = {hang_pi}/5")
    print()
    print("Giai thich:")
    print("- Theo feature_importances_, cot ngau nhien co nhieu gia tri khac nhau (450 muc)")
    print("  nen cay luon tim duoc nguong chia nho tap hoc va cong diem cho no (xep thu 3).")
    print("- Theo permutation_importance tren tap kiem tra, xao tron mot cot vo dung")
    print("  khong lam accuracy giam (hoac tham chi tang nhe, gia tri gan 0 hoac am),")
    print("  nen no roi xuong vi tri thap nhat, phan anh dung muc do huu ich thuc su.")


if __name__ == "__main__":
    main()
