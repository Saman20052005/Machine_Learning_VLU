# -*- coding: utf-8 -*-
"""Bài 1: Tự tính lan truyền tiến bằng numpy từ trọng số đã học của MLP trên XOR."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/xor.csv")
    X = df[["x1", "x2"]].to_numpy()
    y = df["nhan"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
                        max_iter=5000, random_state=0).fit(X_tr, y_tr)

    W1, W2 = mlp.coefs_
    b1, b2 = mlp.intercepts_

    X5 = X_te[:5]
    H = relu(X5 @ W1 + b1)
    xs_tu_tinh = sigmoid(H @ W2 + b2).ravel()
    xs_sklearn = mlp.predict_proba(X5)[:, 1]

    np.testing.assert_allclose(xs_tu_tinh, xs_sklearn, rtol=1e-10, atol=1e-10)
    print("Diem   tu tinh (numpy)   mlp.predict_proba   chenh lech")
    for i in range(5):
        diff = abs(xs_tu_tinh[i] - xs_sklearn[i])
        print(f"  {i + 1}    {xs_tu_tinh[i]:15.10f}   {xs_sklearn[i]:17.10f}   {diff:.2e}")
    print("\nHai ket qua khop hoan toan toi sai so dau phay dong.")


if __name__ == "__main__":
    main()
