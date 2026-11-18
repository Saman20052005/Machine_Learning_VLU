# -*- coding: utf-8 -*-
"""Bài 6: Đo thời gian .fit của SVC() và LinearSVC() trên dữ liệu lớn."""
import time

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC


def main():
    print("n_samples   SVC (s)   LinearSVC (s)")
    ket_qua = []
    for n in [2000, 5000, 10000, 20000]:
        X, y = make_classification(n_samples=n, n_features=20, random_state=0)

        t0 = time.perf_counter()
        SVC().fit(X, y)
        t_svc = time.perf_counter() - t0

        t0 = time.perf_counter()
        LinearSVC(random_state=0).fit(X, y)
        t_lin = time.perf_counter() - t0

        ket_qua.append((n, t_svc, t_lin))
        print(f"  {n:7d}   {t_svc:7.4f}     {t_lin:11.4f}")
    print()
    print("Nhan xet:")
    print("- Khi so mau tang gap doi (tu 5000 len 10000, hoac 10000 len 20000),")
    print("  thoi gian .fit cua SVC (kernel RBF) tang nhanh hon tuyen tinh (khoang 3-4 lan),")
    print("  trong khi LinearSVC tang cham hon nhieu (xap xi tuyen tinh, khoang 2 lan) va nhanh hon rat nhieu.")


if __name__ == "__main__":
    main()
