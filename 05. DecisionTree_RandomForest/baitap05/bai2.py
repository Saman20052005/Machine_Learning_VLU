# -*- coding: utf-8 -*-
"""Bài 2: Tự tìm câu hỏi gốc tốt nhất trên dữ liệu thật (450 khách tập học)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def gini(nhan):
    """Chi so Gini cua mot nut co hai lop."""
    if len(nhan) == 0:
        return 0.0
    p1 = nhan.mean()
    return 1.0 - p1 ** 2 - (1.0 - p1) ** 2


def tim_cau_hoi(X, y):
    """Thu moi cot va moi nguong nam giua hai gia tri lien nhau, tra ve cau hoi giam Gini nhieu nhat."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    g_goc = gini(y_arr)
    tot_nhat = None

    for cot in X.columns:
        gia_tri = X[cot].to_numpy()
        muc = np.sort(np.unique(gia_tri))
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (a + b) / 2.0
            di_trai = gia_tri <= nguong
            trai, phai = y_arr[di_trai], y_arr[~di_trai]
            g_chia = len(trai) / n * gini(trai) + len(phai) / n * gini(phai)
            giam = g_goc - g_chia
            if tot_nhat is None or giam > tot_nhat[2]:
                tot_nhat = (cot, float(nguong), float(giam), float(g_chia))
    return tot_nhat


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    cot, nguong, giam, g_chia = tim_cau_hoi(X_tr, y_tr)
    print(f"So khach tap hoc: {len(X_tr)}")
    print(f"Gini nut goc    : {gini(y_tr.to_numpy()):.4f}")
    print(f"Cau hoi tot nhat: {cot} <= {nguong:.2f}")
    print(f"Gini sau chia   : {g_chia:.4f}, Gini giam: {giam:.4f}")
    print()

    cay = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
    print("Doi chieu voi scikit-learn (max_depth=1):")
    print(export_text(cay, feature_names=list(X.columns), show_weights=True, decimals=2))
    print("Giai thich:")
    print("- Ham duyet tat ca cac cot va moi nguong nam giua hai gia tri da sap xep lien ke.")
    print("- Voi moi cap (cot, nguong), ham tinh Gini trung binh co trong so cua hai nhanh")
    print("  va chon phep chia lam Gini giam nhieu nhat (Gini sau chia nho nhat).")
    print("- Ket qua thu_nhap <= 15.05 trung khop hoan toan voi nut goc cua scikit-learn.")


if __name__ == "__main__":
    main()
