"""Bài 5: so sánh biến thể sai và đúng trên dữ liệu liên tục."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/sinh_vien.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df[["gio_on", "diem_giua_ky"]], df.qua_mon,
        test_size=0.25, random_state=17, stratify=df.qua_mon)
    print("BAI 5: CHON SAI BIEN THE NAIVE BAYES")
    print(f"So sinh vien: {len(df)}; train: {len(X_train)}; test: {len(X_test)}")
    for ten, model in [("MultinomialNB", MultinomialNB()), ("GaussianNB", GaussianNB())]:
        model.fit(X_train, y_train)
        print(f"{ten:14s}: accuracy = {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print("\nChuong trinh khong bao loi vi ca hai dac trung deu khong am.")
    print("MultinomialNB chap nhan so thuc khong am, nhung gio on va diem")
    print("khong phai so dem cac loai su kien trong mo hinh multinomial.")
    print("No bien hai dai luong khac don vi thanh thanh phan dem, lam sai")
    print("y nghia xac suat; doi don vi gio sang phut cung co the doi ket qua.")
    print("GaussianNB mo ta mat do lien tuc bang trung binh va phuong sai")
    print("tung dac trung theo lop, phu hop hon ve loai du lieu.")
    print("Accuracy tren mot tap nho khong chung minh gia dinh phan phoi dung.")


if __name__ == "__main__":
    main()
