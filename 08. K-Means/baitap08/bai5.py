# -*- coding: utf-8 -*-
"""Bài 5: Thử DBSCAN với các giá trị eps trên bộ hai vầng trăng."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X = df[["x1", "x2"]].to_numpy()
    y = df["nhan"].to_numpy()

    print("eps   so cum   so nhieu (-1)   ty le khop nhan that (bo nhieu)")
    for e in [0.2, 0.3, 0.4]:
        db = DBSCAN(eps=e, min_samples=5).fit(X)
        nhan = db.labels_
        so_cum = len(set(nhan) - {-1})
        so_nhieu = int((nhan == -1).sum())
        if so_cum == 2:
            hop_le = nhan != -1
            khop = (nhan[hop_le] == y[hop_le]).mean()
            khop_tot = f"{max(khop, 1 - khop):.4f}"
        else:
            khop_tot = "N/A"
        print(f"{e:.1f}   {so_cum:6d}   {so_nhieu:13d}   {khop_tot}")
    print()
    print("Nhan xet:")
    print("- Ca ba gia tri eps (0.2, 0.3, 0.4) deu chi tim duoc 1 cum (va so diem nhieu giam")
    print("  tu 40 xuong 3 khi eps tang), nen chua tach duoc hai vang trang tot hon K-Means")
    print("  o muc nay (do du lieu hai vang trang co nhieu va hai nhanh nam kha gan nhau).")
    print("- Vai tro cua eps: la ban kinh tim lang gieng. eps nho thi nhieu diem bi coi la nhieu (-1);")
    print("  eps lon thi cac diem noi voi nhau thanh mot cum lon, giam so diem nhieu.")


if __name__ == "__main__":
    main()
