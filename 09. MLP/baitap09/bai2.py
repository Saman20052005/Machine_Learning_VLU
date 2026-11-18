# -*- coding: utf-8 -*-
"""Bài 2: Kiểm chứng một mạng XOR khác bằng numpy."""

import numpy as np


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    x1 = X[:, 0]
    x2 = X[:, 1]

    h1 = np.maximum(0.0, x1 - x2)
    h2 = np.maximum(0.0, x2 - x1)
    y = h1 + h2

    print("x1  x2   h1   h2    y")
    for a, b, u, v, out in zip(x1, x2, h1, h2, y):
        print(f" {a:.0f}   {b:.0f}    {u:.0f}    {v:.0f}    {out:.0f}")
    print()
    print("Giai thich:")
    print("- Khi x1 = x2 ((0,0) hoac (1,1)): x1 - x2 = 0 va x2 - x1 = 0,")
    print("  nen h1 = 0, h2 = 0, suy ra y = 0.")
    print("- Khi x1 != x2 ((0,1) hoac (1,0)): mot hieu bang +1 va hieu con lai bang -1")
    print("  (bi ReLU cat ve 0), nen h1 + h2 = 1 + 0 = 1.")
    print("- Hai no-ron an tao thanh phep |x1 - x2|, tinh dung XOR cho 4 dau vao.")


if __name__ == "__main__":
    main()
