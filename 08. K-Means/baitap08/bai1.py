# -*- coding: utf-8 -*-
"""Bài 1: Tự tính chỉ số silhouette bằng numpy và tìm khách có s nhỏ nhất."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Tinh a, b, s cho diem thu i theo dung dinh nghia silhouette."""
    cum_i = nhan[i]
    kc = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    cung_cum = (nhan == cum_i)
    cung_cum[i] = False
    if cung_cum.sum() == 0:
        return 0.0, 0.0, 0.0
    a = float(kc[cung_cum].mean())

    cac_cum_khac = [c for c in np.unique(nhan) if c != cum_i]
    b = float(min(kc[nhan == c].mean() for c in cac_cum_khac))
    s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return a, b, float(s)


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_hang.csv")
    X = df[["thu_nhap", "diem_chi_tieu"]]
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
    nhan = km.labels_

    ds_s = np.array([silhouette_mot_diem(Xs, nhan, i)[2] for i in range(len(Xs))])
    s_tb = float(ds_s.mean())
    s_sk = float(silhouette_score(Xs, nhan))

    print(f"Silhouette trung binh tu tinh   : {s_tb:.6f}")
    print(f"Silhouette trung binh scikit-learn: {s_sk:.6f}")
    print()

    idx_min = int(np.argmin(ds_s))
    a_min, b_min, s_min = silhouette_mot_diem(Xs, nhan, idx_min)
    khach = df.iloc[idx_min]
    print(f"Khach co s nho nhat: chi so {idx_min} (s = {s_min:.4f}, a = {a_min:.4f}, b = {b_min:.4f})")
    print(f"  tuoi = {int(khach['tuoi'])}, thu_nhap = {khach['thu_nhap']:.1f}, "
          f"diem_chi_tieu = {int(khach['diem_chi_tieu'])}, cum = {nhan[idx_min]}")
    print()
    print("Nhan xet:")
    print("- Khach nay nam o vung giap ranh giua hai cum (xa tam cum cua minh va gan cum lan can),")
    print("  nen khoang cach trung binh toi cum minh (a) gan bang toi cum gan nhat khac (b).")


if __name__ == "__main__":
    main()
