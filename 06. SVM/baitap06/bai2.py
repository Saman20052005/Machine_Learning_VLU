# -*- coding: utf-8 -*-
"""Bài 2: Thêm một điểm mới và quan sát sự thay đổi của đường ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def in_ket_qua(ten, X, y):
    svm = SVC(kernel="linear", C=1000).fit(X, y)
    w1, w2 = svm.coef_[0]
    b = svm.intercept_[0]
    print(f"{ten}:")
    print(f"  w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}, so vecto HT = {len(svm.support_)}")
    return w1, w2, b


def main():
    X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                      cluster_std=0.9, random_state=0)
    y = 1 - y

    in_ket_qua("1. Du lieu goc (24 diem)", X, y)

    X_xa = np.vstack([X, [-1.0, -1.0]])
    y_xa = np.append(y, 1)
    in_ket_qua("2. Them diem lop 1 tai (-1, -1)", X_xa, y_xa)

    X_gan = np.vstack([X, [4.5, 4.0]])
    y_gan = np.append(y, 1)
    in_ket_qua("3. Thay bang diem lop 1 tai (4.5, 4)", X_gan, y_gan)
    print()
    print("Giai thich:")
    print("- Khi them (-1, -1) o xa phia sau cum lop 1, diem nay nam ngoai le va khong vi pham")
    print("  mep le nao, nen khong tro thanh vecto ho tro; duong ranh gioi giu nguyen.")
    print("- Khi them (4.5, 4) nam sat ranh gioi (lot vao vung le cu), diem nay tro thanh")
    print("  vecto ho tro moi va day duong ranh gioi dich chuyen theo.")


if __name__ == "__main__":
    main()
