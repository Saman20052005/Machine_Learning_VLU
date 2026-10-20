"""Bài 5: chọn k hồi quy bằng CV MAE trên train, rồi mới đánh giá test."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/gia_nha.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df[["dien_tich"]], df.gia, test_size=.2, random_state=42)
    print("BAI 5: CHON K HOI QUY BANG CV 5 PHAN TREN TRAIN")
    print("Chi dung dien_tich nhu muc 6; MAE tinh bang ty dong.")
    diem = {}
    print("  k    CV MAE (duong)")
    for k in range(1, 21):
        diem[k] = cross_val_score(KNeighborsRegressor(k), X_tr, y_tr, cv=5,
                                  scoring="neg_mean_absolute_error").mean()
        print(f"{k:3d}   {-diem[k]:.6f}")
    # sklearn tối đa hóa score âm, tương đương tối thiểu hóa MAE dương.
    k_tot = max(diem, key=diem.get)
    knn = KNeighborsRegressor(k_tot).fit(X_tr, y_tr)
    duong = LinearRegression().fit(X_tr, y_tr)
    mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
    mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
    print(f"\nk tot nhat = {k_tot}; CV MAE = {-diem[k_tot]:.6f}")
    print(f"Test MAE KNN = {mae_knn:.6f}")
    print(f"Test MAE LinearRegression = {mae_duong:.6f}")
    print("KNN co MAE thap hon duong thang?", "Co" if mae_knn < mae_duong else "Khong")
    print("MAE cang nho cang tot; neg_MAE cang lon (gan 0) cang tot.")
    print("Chi chon k bang train CV; khong dieu chinh k sau khi xem test.")
    print("KNN lay trung binh cuc bo, con duong thang tan dung xu huong tuyen tinh;")
    print("chon k khong bao dam vuot mo hinh phu hop voi cau truc that cua du lieu.")


if __name__ == "__main__":
    main()
