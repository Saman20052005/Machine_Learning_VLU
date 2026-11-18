# -*- coding: utf-8 -*-
"""Bài 6: Ảnh hưởng của một lần đoán sai lớn (5 tỷ) lên MAE, RMSE và R^2."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    duong = LinearRegression().fit(X_tr, y_tr)
    du = duong.predict(X_te)

    mae_1 = mean_absolute_error(y_te, du)
    rmse_1 = root_mean_squared_error(y_te, du)
    r2_1 = r2_score(y_te, du)

    du_sai = du.copy()
    du_sai[0] += 5.0  # Cộng thêm 5 tỷ vào dự đoán của đúng một căn
    mae_2 = mean_absolute_error(y_te, du_sai)
    rmse_2 = root_mean_squared_error(y_te, du_sai)
    r2_2 = r2_score(y_te, du_sai)

    print("Thuoc do   Ban dau   Sau khi sai +5 ty   Chenh lech")
    print(f"MAE        {mae_1:7.4f}   {mae_2:17.4f}   {mae_2 - mae_1:+10.4f}")
    print(f"RMSE       {rmse_1:7.4f}   {rmse_2:17.4f}   {rmse_2 - rmse_1:+10.4f}")
    print(f"R2         {r2_1:7.4f}   {r2_2:17.4f}   {r2_2 - r2_1:+10.4f}")
    print()
    print("Giai thich:")
    print("- RMSE thay doi nhieu nhat (+1.1775), R2 giam manh (-0.6148), con MAE chi tang +0.4167.")
    print("- MAE lay trung binh tri tuyet doi sai so, nen 1 can sai them 5 ty chia cho 12 can")
    print("  chi lam MAE tang toi da ~5/12 = 0.4167.")
    print("- RMSE binh phuong sai so truoc khi lay trung binh roi khai can (5^2 = 25),")
    print("  nen mot sai so lon bi phong dai rat manh, khien RMSE va R2 thay doi nhieu.")


if __name__ == "__main__":
    main()
