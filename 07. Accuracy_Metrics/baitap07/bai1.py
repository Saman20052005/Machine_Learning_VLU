# -*- coding: utf-8 -*-
"""Bài 1: Tự viết hàm danh_gia(y_that, y_doan) bằng numpy và xử lý chia cho 0."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Tinh accuracy, precision, recall, f1 chi dung numpy; tra ve 0.0 neu mau so bang 0."""
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)
    tp = int(np.sum((that == 1) & (doan == 1)))
    fp = int(np.sum((that == 0) & (doan == 1)))
    fn = int(np.sum((that == 1) & (doan == 0)))
    tn = int(np.sum((that == 0) & (doan == 0)))

    acc = (tp + tn) / len(that) if len(that) > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return acc, p, r, f1


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)

    for ten, mh in [("Hoi quy logistic", lr), ("Mo hinh luoi", luoi)]:
        du = mh.predict(X_te)
        acc, p, r, f1 = danh_gia(y_te, du)
        acc_sk = accuracy_score(y_te, du)
        p_sk = precision_score(y_te, du, zero_division=0)
        r_sk = recall_score(y_te, du, zero_division=0)
        f1_sk = f1_score(y_te, du, zero_division=0)
        print(f"{ten}:")
        print(f"  Tu viet     : acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        print(f"  Scikit-learn: acc={acc_sk:.4f}, P={p_sk:.4f}, R={r_sk:.4f}, F1={f1_sk:.4f}")
    print()
    print("Giai thich:")
    print("- Voi mo hinh luoi, mo hinh khong bao ai bo hoc (TP = 0, FP = 0), nen mau so")
    print("  cua precision (TP + FP) bang 0 va khong tinh duoc theo phep chia thong thuong.")
    print("- Ham danh_gia kiem tra mau so > 0 truoc khi chia; neu bang 0 thi tra ve 0.0")
    print("  (tuong duong zero_division=0 trong scikit-learn) de tranh loi chia cho 0.")


if __name__ == "__main__":
    main()
