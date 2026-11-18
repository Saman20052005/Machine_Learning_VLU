# -*- coding: utf-8 -*-
"""Bước 3: chọn số cụm K bằng khuỷu tay và chỉ số silhouette."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_hang.csv")
    Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

    print("   K   inertia   giam so voi K-1   silhouette")
    truoc = None
    inertias = []
    sils = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        inertias.append(km.inertia_)
        giam = "" if truoc is None else f"{truoc - km.inertia_:8.1f}"
        # Silhouette cần ít nhất hai cụm
        if k == 1:
            sil = ""
        else:
            s = silhouette_score(Xs, km.labels_)
            sils.append(s)
            sil = f"{s:.3f}"
        print(f"  {k:2d}   {km.inertia_:7.1f}   {giam:>15s}   {sil:>10s}")
        truoc = km.inertia_

    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, 11), inertias, "o-")
    axes[0].set_title("Phuong phap khuyu tay")
    axes[0].set_xlabel("So cum K")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(2, 11), sils, "o-")
    axes[1].set_title("Chi so silhouette")
    axes[1].set_xlabel("So cum K")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(bieu_do / "k3_chon_k.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
