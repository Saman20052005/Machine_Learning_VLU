# -*- coding: utf-8 -*-
"""Bài 6: Rừng ngẫu nhiên cho bài toán hồi quy trên dữ liệu giá nhà."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)
    cay3 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
    duong = LinearRegression().fit(X_tr, y_tr)

    mae_rung = mean_absolute_error(y_te, rung.predict(X_te))
    mae_cay3 = mean_absolute_error(y_te, cay3.predict(X_te))
    mae_duong = mean_absolute_error(y_te, duong.predict(X_te))

    print("MAE tren tap kiem tra (ty dong):")
    print(f"  Rung 200 cay : {mae_rung:.4f}")
    print(f"  Cay sau 3    : {mae_cay3:.4f}")
    print(f"  Duong thang  : {mae_duong:.4f}")
    print()

    can_250 = pd.DataFrame({"dien_tich": [250.0]})
    p_rung = rung.predict(can_250)[0]
    p_cay3 = cay3.predict(can_250)[0]
    p_duong = duong.predict(can_250)[0]
    print("Du doan gia can 250 m2 (ty dong):")
    print(f"  Rung 200 cay : {p_rung:.2f}")
    print(f"  Cay sau 3    : {p_cay3:.2f}")
    print(f"  Duong thang  : {p_duong:.2f}")
    print()
    print("Giai thich:")
    print("- Rung KHONG ngoai suy duoc vuot ra ngoai khoang gia tri da thay trong tap hoc.")
    print("- Moi cay trong rung du doan bang trung binh gia cua cac mau tap hoc roi vao la do.")
    print("- Can 250 m2 lon hon moi can trong tap hoc nen o moi cay no deu roi vao la ngoai cung")
    print("  ben phai; trung binh cua 200 la do van bi chan boi muc gia cao nhat trong tap hoc.")


if __name__ == "__main__":
    main()
