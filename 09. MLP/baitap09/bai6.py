# -*- coding: utf-8 -*-
"""Bài 6: MLPRegressor trên dữ liệu giá nhà và kiểm tra ngoại suy."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
    ).fit(X_tr, y_tr)
    duong = LinearRegression().fit(X_tr, y_tr)

    mae_mlp = mean_absolute_error(y_te, mlp.predict(X_te))
    mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
    print("MAE tren tap kiem tra (ty dong):")
    print(f"  MLPRegressor     : {mae_mlp:.4f}")
    print(f"  LinearRegression : {mae_duong:.4f}")
    print()

    can_250 = pd.DataFrame({"dien_tich": [250.0]})
    p_mlp = mlp.predict(can_250)[0]
    p_duong = duong.predict(can_250)[0]
    print("Du doan gia can 250 m2 (ty dong):")
    print(f"  MLPRegressor     : {p_mlp:.2f}")
    print(f"  LinearRegression : {p_duong:.2f}")
    print()
    print("Nhan xet:")
    print("- Voi ham kich hoat ReLU, MLPRegressor keo dai tuyen tinh ngoai vung du lieu hoc")
    print("  nen ngoai suy tot hon nhieu so voi cay quyet dinh hay KNN (khong bi dung yen o gia max),")
    print("  tuy nhien doan gia van co the lech chut it so voi duong thang hoi quy tuyen tinh.")


if __name__ == "__main__":
    main()
