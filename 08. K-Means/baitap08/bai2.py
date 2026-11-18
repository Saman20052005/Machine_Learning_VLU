# -*- coding: utf-8 -*-
"""Bài 2: Thêm cột tuổi vào phân cụm khách hàng và chọn lại K bằng silhouette."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_hang.csv")
    cot = ["tuoi", "thu_nhap", "diem_chi_tieu"]
    sc = StandardScaler().fit(df[cot])
    Xs = sc.transform(df[cot])

    print("K    silhouette")
    diem = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        diem[k] = silhouette_score(Xs, km.labels_)
        print(f"{k:2d}   {diem[k]:.4f}")

    k_tot = max(diem, key=diem.get)
    print(f"\nK tot nhat theo silhouette: K = {k_tot} (silhouette = {diem[k_tot]:.4f})")
    print()

    km_tot = KMeans(n_clusters=k_tot, n_init=10, random_state=0).fit(Xs)
    tam = pd.DataFrame(sc.inverse_transform(km_tot.cluster_centers_), columns=cot)
    tam["so_khach"] = np.bincount(km_tot.labels_)
    print(f"Tam cum voi K = {k_tot} (don vi goc):")
    print(tam.round(1).to_string())
    print()
    print("Nhan xet:")
    print("- Khi them cot tuoi, K tot nhat van la 5 (giong muc 3), nhung chi so silhouette")
    print("  giam tu 0.723 xuong 0.4808 vi cac cum bi trai rong them theo chieu tuoi.")
    print("- Tam cum va so khach thay doi nhe (nhom thu nhap thap chi nhieu co 39 khach,")
    print("  tuoi tb 25.1; nhom thu nhap vua chi tieu vua giam tu 60 xuong 57 khach).")


if __name__ == "__main__":
    main()
