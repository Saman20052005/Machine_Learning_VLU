"""Bước 5: hồi quy giá nhà và giới hạn ngoại suy của KNN."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/gia_nha.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df[["dien_tich"]], df.gia, test_size=.2, random_state=42)
    knn = KNeighborsRegressor(5).fit(X_tr, y_tr)
    duong = LinearRegression().fit(X_tr, y_tr)
    print("MAE TREN TAP TEST (TY DONG)")
    print(f"KNN k=5       : {mean_absolute_error(y_te, knn.predict(X_te)):.4f}")
    print(f"LinearRegression: {mean_absolute_error(y_te, duong.predict(X_te)):.4f}")
    moi = pd.DataFrame({"dien_tich": [60., 100., 150., 250.]})
    print("\nDien tich    KNN    Duong thang")
    for dt, p, q in zip(moi.dien_tich, knn.predict(moi), duong.predict(moi)):
        print(f"{dt:9.0f} {p:7.2f} {q:13.2f}")
    gan = knn.kneighbors(pd.DataFrame({"dien_tich": [250.]}), return_distance=False)[0]
    print("Nam lang gieng can 250 m2:", X_tr.iloc[gan].dien_tich.tolist())
    print("KNN uniform lay trung binh gia hang xom: ngoai vung du lieu, tap hang xom")
    print("o dau bien khong doi nen du doan phang. Duong thang van keo dai xu huong.")
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = pd.DataFrame({"dien_tich": np.linspace(25, 260, 700)})
    ax.axvspan(X_tr.dien_tich.min(), X_tr.dien_tich.max(), color="#d9efe3", alpha=.6, label="Vung dien tich train")
    ax.scatter(X_tr.dien_tich, y_tr, label="Train", s=22)
    ax.scatter(X_te.dien_tich, y_te, label="Test", marker="x", s=35)
    ax.plot(grid.dien_tich, knn.predict(grid), label="KNN k=5")
    ax.plot(grid.dien_tich, duong.predict(grid), label="LinearRegression")
    ax.set(xlabel="Dien tich (m2)", ylabel="Gia (ty dong)", title="KNN va hoi quy tuyen tinh: noi suy / ngoai suy")
    ax.legend()
    fig.tight_layout()
    (root / "bieu_do").mkdir(exist_ok=True)
    fig.savefig(root / "bieu_do/e5_hoi_quy_gia_nha.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
