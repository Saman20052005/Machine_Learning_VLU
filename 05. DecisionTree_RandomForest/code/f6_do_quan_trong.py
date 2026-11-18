# -*- coding: utf-8 -*-
"""Bước 6: đọc độ quan trọng của đặc trưng, và thấy giới hạn của nó."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    rung = RandomForestClassifier(n_estimators=200, random_state=0)
    rung.fit(X_tr, y_tr)
    qt = pd.Series(rung.feature_importances_, index=X.columns)
    print("Do quan trong, bon cot that")
    print(qt.sort_values(ascending=False).round(3).to_string())
    print("Tong:", round(qt.sum(), 3))
    print()

    # Thêm một cột toàn số ngẫu nhiên, chẳng liên quan gì tới việc trả nợ
    rng = np.random.default_rng(0)
    X_tr_nn = X_tr.copy()
    X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
    rung_nn = RandomForestClassifier(n_estimators=200, random_state=0)
    rung_nn.fit(X_tr_nn, y_tr)
    qt_nn = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
    print("Do quan trong, them mot cot ngau nhien")
    print(qt_nn.sort_values(ascending=False).round(3).to_string())

    # Biểu đồ độ quan trọng
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)
    qt_sap = qt.sort_values(ascending=True)
    plt.figure(figsize=(7, 4))
    plt.barh(qt_sap.index, qt_sap.values)
    plt.xlabel("Do quan trong, tong bang 1")
    plt.title("Rung cho biet cot nao duoc dung nhieu nhat")
    plt.tight_layout()
    plt.savefig(bieu_do / "f6_do_quan_trong.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
