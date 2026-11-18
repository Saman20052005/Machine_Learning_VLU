# -*- coding: utf-8 -*-
"""Bài 4: Ảnh hưởng của tốc độ học learning_rate_init trên bộ chữ số."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)

    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    plt.figure(figsize=(7.5, 4.8))
    print("learning_rate_init   n_iter_   mat mat cuoi   accuracy test")
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        mh = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=lr,
                          max_iter=1000, random_state=0),
        ).fit(X_tr, y_tr)
        mlp = mh[-1]
        acc = accuracy_score(y_te, mh.predict(X_te))
        mm_cuoi = mlp.loss_curve_[-1]
        print(f"  {lr:<18} {mlp.n_iter_:7d}   {mm_cuoi:12.4f}   {acc:13.4f}")
        plt.plot(mlp.loss_curve_, label=f"lr={lr}")

    plt.xlabel("Vong hoc")
    plt.ylabel("Mat mat (loss)")
    plt.title("Duong mat mat theo toc do hoc (learning_rate_init)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = bieu_do / "toc_do_hoc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nDa luu bieu do: {out_path.relative_to(root).as_posix()}")
    print("Nhan xet:")
    print("- Toc do hoc qua nho (0.0001): mat mat giam rat cham, can nhieu vong hoc (cham toi da max_iter).")
    print("- Toc do hoc qua lon (0.1): buoc nhay qua manh, duong mat mat dao dong hoac dung som voi")
    print("  accuracy thap hon muc chuan (0.001).")


if __name__ == "__main__":
    main()
