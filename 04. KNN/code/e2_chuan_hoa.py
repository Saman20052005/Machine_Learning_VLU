"""Bước 2: so sánh ảnh hưởng của chuẩn hóa và đặc trưng không liên quan."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien_mo_rong.csv")
    cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cot], df.qua_mon, test_size=0.25, random_state=17, stratify=df.qua_mon)
    print("ANH HUONG CUA CHUAN HOA (k=7)")
    print(f"Luon doan qua mon: {y_te.mean():.4f}")
    cau_hinh = [
        ("Ba cot, khong chuan hoa", cot, KNeighborsClassifier(7)),
        ("Ba cot, co chuan hoa", cot, make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
        ("Bo chi_tieu, co chuan hoa", cot[:2], make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
    ]
    for ten, chon, model in cau_hinh:
        model.fit(X_tr[chon], y_tr)
        print(f"{ten:30s}: {accuracy_score(y_te, model.predict(X_te[chon])):.4f}")
    print("Pipeline chi fit StandardScaler tren train, sau do transform test.")
    print("Chuan hoa xu ly thang do; bo cot khong lien quan con giup giam nhieu.")


if __name__ == "__main__":
    main()
