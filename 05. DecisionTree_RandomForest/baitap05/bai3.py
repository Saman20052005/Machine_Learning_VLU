# -*- coding: utf-8 -*-
"""Bài 3: So sánh criterion='gini' và criterion='entropy'."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def chon_do_sau(X_tr, y_tr, X_te, y_te, criterion):
    diem = {}
    for d in range(1, 16):
        mh = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=0)
        diem[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    d_tot = max(diem, key=diem.get)
    tot = DecisionTreeClassifier(criterion=criterion, max_depth=d_tot, random_state=0).fit(X_tr, y_tr)
    acc_te = accuracy_score(y_te, tot.predict(X_te))
    return d_tot, diem[d_tot], tot.get_n_leaves(), acc_te, diem


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    d_g, cv_g, la_g, te_g, bang_g = chon_do_sau(X_tr, y_tr, X_te, y_te, "gini")
    d_e, cv_e, la_e, te_e, bang_e = chon_do_sau(X_tr, y_tr, X_te, y_te, "entropy")

    print("Do sau   CV gini   CV entropy")
    for d in range(1, 16):
        print(f"  {d:2d}     {bang_g[d]:.4f}     {bang_e[d]:.4f}")
    print()
    print(f"Gini   : do sau tot nhat = {d_g}, so la = {la_g}, CV = {cv_g:.4f}, test = {te_g:.4f}")
    print(f"Entropy: do sau tot nhat = {d_e}, so la = {la_e}, CV = {cv_e:.4f}, test = {te_e:.4f}")
    print()
    print("Nhan xet:")
    print("- Doi thuoc do tu gini sang entropy khong lam ket qua khac di nhieu;")
    print("  ca hai thuong cho do sau toi uu va accuracy tren tap kiem tra rat gan nhau.")


if __name__ == "__main__":
    main()
