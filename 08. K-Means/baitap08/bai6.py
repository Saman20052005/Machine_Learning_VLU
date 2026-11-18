# -*- coding: utf-8 -*-
"""Bài 6: Phân cụm K-Means (K=10) trên bộ chữ số viết tay load_digits."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


def main():
    X, nhan_that = load_digits(return_X_y=True)
    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
    cum = km.labels_

    bang = pd.crosstab(nhan_that, cum)
    bang.index.name = "chu_so"
    bang.columns.name = "cum"
    print("Bang cheo chu_so that (hang) vs cum K-Means (cot):")
    print(bang.to_string())
    print()
    print("Nhan xet:")
    print("- Cac chu so co net dac trung ro rang nhu 0, 4, 6, 7 duoc gom rat gon vao mot cum.")
    print("- Cac chu so co net giong nhau nhu 1 va 8 (hoac 2, 3, 9) hay bi gom chung hoac chia ra.")


if __name__ == "__main__":
    main()
