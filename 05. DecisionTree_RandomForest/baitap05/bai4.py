# -*- coding: utf-8 -*-
"""Bài 4: Cắt tỉa bằng min_samples_leaf và so với max_depth=5."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print("min_samples_leaf   tap hoc   kiem dinh cheo")
    diem = {}
    for m in [1, 2, 5, 10, 20, 40]:
        mh = DecisionTreeClassifier(min_samples_leaf=m, random_state=0)
        diem[m] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        hoc = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))
        print(f"  {m:14d}   {hoc:.4f}       {diem[m]:.4f}")
    m_tot = max(diem, key=diem.get)
    print()

    cay_leaf = DecisionTreeClassifier(min_samples_leaf=m_tot, random_state=0).fit(X_tr, y_tr)
    acc_leaf = accuracy_score(y_te, cay_leaf.predict(X_te))
    print(f"min_samples_leaf tot nhat = {m_tot}, do sau = {cay_leaf.get_depth()}, so la = {cay_leaf.get_n_leaves()}")
    print(f"Accuracy test (min_samples_leaf={m_tot}): {acc_leaf:.4f}")

    cay_sau5 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr, y_tr)
    acc_sau5 = accuracy_score(y_te, cay_sau5.predict(X_te))
    print(f"Accuracy test (max_depth=5)        : {acc_sau5:.4f}")
    print()
    print("Nhan xet:")
    if acc_leaf > acc_sau5:
        print(f"- Cat tia bang min_samples_leaf={m_tot} cho accuracy cao hon ({acc_leaf:.4f} > {acc_sau5:.4f}).")
    elif acc_leaf < acc_sau5:
        print(f"- Cat tia bang max_depth=5 cho accuracy cao hon ({acc_sau5:.4f} > {acc_leaf:.4f}).")
    else:
        print(f"- Ca hai cach cat tia cho cung accuracy tren tap kiem tra ({acc_leaf:.4f}).")


if __name__ == "__main__":
    main()
