"""Bước 1: tự tính khoảng cách, chuẩn hóa và tìm láng giềng bằng numpy."""
from pathlib import Path

import numpy as np
import pandas as pd


def kc(u, v):
    return np.sqrt(np.sum((u - v) ** 2))


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien_mo_rong.csv")
    ba_ban = {"A": np.array([5., 3., 4000.]), "B": np.array([25., 9., 4100.]),
              "C": np.array([6., 3.5, 7000.])}
    print("KHOANG CACH EUCLID: d(u,v) = sqrt(sum((u-v)^2))")
    print(f"Vi du (10,5) va (13,9): {kc(np.array([10, 5]), np.array([13, 9])):.1f}")
    for ten in ["B", "C"]:
        print(f"Chua chuan hoa: d(A,{ten}) = {kc(ba_ban['A'], ba_ban[ten]):.1f}")
    X = df[["gio_on", "diem_giua_ky", "chi_tieu"]].to_numpy(dtype=float)
    tb, dl = X.mean(axis=0), X.std(axis=0)
    print("Do lech chuan (gio_on, diem_giua_ky, chi_tieu):", dl.round(2))
    chuan = {ten: (v - tb) / dl for ten, v in ba_ban.items()}
    for ten in ["B", "C"]:
        print(f"Sau chuan hoa: d(A,{ten}) = {kc(chuan['A'], chuan[ten]):.2f}")
    # Đây là minh họa tìm hàng xóm trên kho đầy đủ, không phải đánh giá test.
    X2 = X[:, :2]
    moi = np.array([12., 6.5])
    khoang = np.linalg.norm((X2 - moi) / X2.std(axis=0), axis=1)
    gan = np.argsort(khoang, kind="stable")[:5]
    print("\nBan moi: on 12 gio, diem giua ky 6.5. Nam lang gieng:")
    for i in gan:
        print(f"Dong {i:3d}: gio={X2[i, 0]:5.1f}; diem={X2[i, 1]:4.1f}; "
              f"d={khoang[i]:.3f}; {'qua' if df.qua_mon.iloc[i] else 'rot'}")
    so_qua = int(df.qua_mon.iloc[gan].sum())
    print(f"Bo phieu: {so_qua} qua, {5-so_qua} rot")
    print("Du doan:", "qua mon" if so_qua >= 3 else "rot mon")
    print("Chi_tieu co thang do lon nen chi phoi khoang cach khi chua chuan hoa.")
    print("Khi danh gia mo hinh, chi tinh thong so chuan hoa tren tap train.")


if __name__ == "__main__":
    main()
