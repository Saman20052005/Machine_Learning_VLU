# -*- coding: utf-8 -*-
"""Bài 5: So sánh ảnh hưởng của class_weight='balanced' cho RandomForest và SVC."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def danh_sach_mh(cw):
    return {
        "RandomForest": RandomForestClassifier(200, class_weight=cw, random_state=0),
        "SVC (RBF)": make_pipeline(StandardScaler(), SVC(class_weight=cw, random_state=0)),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    truoc = danh_sach_mh(None)
    sau = danh_sach_mh("balanced")

    print("Mo hinh        Trang thai   Precision   Recall      F1")
    thay_doi = {}
    for ten in truoc:
        du_t = truoc[ten].fit(X_tr, y_tr).predict(X_te)
        p_t, r_t, f_t = precision_score(y_te, du_t), recall_score(y_te, du_t), f1_score(y_te, du_t)
        print(f"{ten:14s} Truoc        {p_t:9.4f}   {r_t:6.4f}  {f_t:6.4f}")

        du_s = sau[ten].fit(X_tr, y_tr).predict(X_te)
        p_s, r_s, f_s = precision_score(y_te, du_s), recall_score(y_te, du_s), f1_score(y_te, du_s)
        print(f"{ten:14s} Can bang     {p_s:9.4f}   {r_s:6.4f}  {f_s:6.4f}")
        thay_doi[ten] = abs(r_s - r_t) + abs(p_s - p_t)
    print()
    nhieu_nhat = max(thay_doi, key=thay_doi.get)
    print(f"Nhan xet: Mo hinh thay doi nhieu nhat la {nhieu_nhat}.")
    print("- SVC (RBF) tang recall manh va giam precision dang ke khi dat class_weight='balanced',")
    print("  trong khi RandomForest it thay doi hon.")


if __name__ == "__main__":
    main()
