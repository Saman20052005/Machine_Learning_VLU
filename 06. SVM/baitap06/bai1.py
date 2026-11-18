# -*- coding: utf-8 -*-
"""Bài 1: Tính tay f(x) và khoảng cách tới ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def main():
    X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                      cluster_std=0.9, random_state=0)
    y = 1 - y
    svm = SVC(kernel="linear", C=1000).fit(X, y)

    w1, w2 = svm.coef_[0]
    b = svm.intercept_[0]
    do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)
    do_rong_le = 2.0 / do_dai_w

    print(f"w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}")
    print(f"||w|| = {do_dai_w:.6f}, do rong le = {do_rong_le:.6f}, nua le = {do_rong_le / 2:.6f}")
    print()

    print("Tinh tay cho hai diem moi:")
    for x1, x2 in [(3.0, 3.0), (5.0, 4.5)]:
        f_tay = w1 * x1 + w2 * x2 + b
        kc = abs(f_tay) / do_dai_w
        print(f"  Diem ({x1}, {x2}): f(x) = {f_tay:+.4f}, khoang cach = {kc:.4f}")
    print()

    print("Khoang cach cua ba vecto ho tro:")
    for idx in svm.support_:
        x1, x2 = X[idx]
        f_tay = w1 * x1 + w2 * x2 + b
        kc = abs(f_tay) / do_dai_w
        print(f"  Diem {idx:2d} ({x1:.2f}, {x2:.2f}): f(x) = {f_tay:+.4f}, khoang cach = {kc:.4f}")
    print()
    print("Giai thich:")
    print("- Ba vecto ho tro nam dung tren hai mep le f(x) = +1 va f(x) = -1, nen |f(x)| = 1.")
    print("- Do do khoang cach tu chung toi ranh gioi la |f(x)| / ||w|| = 1 / ||w||,")
    print("  bang dung mot nua do rong le (2 / ||w||).")


if __name__ == "__main__":
    main()
