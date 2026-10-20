"""Bài 3: đổi đơn vị chi tiêu và kiểm chứng tính bất biến sau chuẩn hóa."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien_mo_rong.csv")
    X = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df.qua_mon, test_size=.25, random_state=17, stratify=df.qua_mon)
    print("BAI 3: DOI CHI_TIEU TU NGHIN DONG SANG TRIEU DONG, k=7")
    print("Cung dong train/test cho moi cau hinh.")
    print("Don vi         Chuan hoa   Accuracy")
    du_doan_chuan = []
    da_chuan = []
    for don_vi, he_so in [("Nghin dong", 1.), ("Trieu dong", .001)]:
        train, test = X_tr.copy(), X_te.copy()
        train["chi_tieu"] *= he_so
        test["chi_tieu"] *= he_so
        for chuan in [False, True]:
            model = (make_pipeline(StandardScaler(), KNeighborsClassifier(7))
                     if chuan else KNeighborsClassifier(7))
            model.fit(train, y_tr)
            pred = model.predict(test)
            print(f"{don_vi:14s} {str(chuan):9s} {accuracy_score(y_te, pred):.4f}")
            if chuan:
                du_doan_chuan.append(pred)
                da_chuan.append(model.named_steps["standardscaler"].transform(test))
    np.testing.assert_allclose(da_chuan[0], da_chuan[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(*du_doan_chuan)
    print("\nDa kiem tra: du lieu test sau chuan hoa tuong duong, nhan du doan trung nhau.")
    print("Chua chuan hoa: d^2 = delta_gio^2 + delta_diem^2 + delta_chi_tieu^2.")
    print("Chia chi_tieu cho 1000 lam thanh phan binh phuong cua no giam 1 trieu lan,")
    print("doi thu tu lang gieng du thong tin thuc te khong doi.")
    print("Sau chuan hoa, voi a=1/1000: (a*x-a*mu)/(a*sigma)=(x-mu)/sigma.")
    print("Do a duong, doi don vi khong doi khoang cach chuan hoa (tru sai so so thuc).")
    print("Pipeline fit scaler rieng tren train; khong chuan hoa truoc khi chia.")


if __name__ == "__main__":
    main()
