# -*- coding: utf-8 -*-
"""Bài 3: So sánh các hàm kích hoạt logistic, tanh, relu trên hai vầng trăng."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    print("activation   tap hoc   tap kiem tra")
    ket_qua = {}
    for act in ["logistic", "tanh", "relu"]:
        mh = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1.0,
                          activation=act, solver="lbfgs",
                          max_iter=20000, random_state=0),
        ).fit(X_tr, y_tr)
        a_tr = accuracy_score(y_tr, mh.predict(X_tr))
        a_te = accuracy_score(y_te, mh.predict(X_te))
        ket_qua[act] = a_te
        print(f"  {act:10s}  {a_tr:.4f}      {a_te:.4f}")
    print()
    tot = max(ket_qua, key=ket_qua.get)
    print(f"Nhan xet: Ham kich hoat cho accuracy tap kiem tra cao nhat la '{tot}' ({ket_qua[tot]:.4f}).")


if __name__ == "__main__":
    main()
