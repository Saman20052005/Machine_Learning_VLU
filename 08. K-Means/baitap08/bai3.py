# -*- coding: utf-8 -*-
"""Bài 3: So sánh init='random' và init='k-means++' qua 50 lần chạy."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_hang.csv")
    Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

    for init_method in ["random", "k-means++"]:
        ds_inertia = []
        dem_tot = 0
        for rs in range(50):
            km = KMeans(n_clusters=5, init=init_method, n_init=1, random_state=rs).fit(Xs)
            val = round(km.inertia_, 1)
            ds_inertia.append(val)
            if val == 23.0:
                dem_tot += 1
        print(f"init = '{init_method}':")
        print(f"  So lan dat inertia 23.0: {dem_tot} / 50 ({dem_tot / 50:.0%})")
        print(f"  Inertia nho nhat / lon nhat: {min(ds_inertia)} / {max(ds_inertia)}")
        print()

    print("Nhan xet:")
    print("- k-means++ chon cac tam ban dau nam xa nhau, nen ti le dat nghiem tot (inertia 23.0)")
    print("  cao hon han so voi random va it roi vao cuc tieu dia phuong xau. Do do k-means++ dang tin hon.")


if __name__ == "__main__":
    main()
