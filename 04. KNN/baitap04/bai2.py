"""Bài 2: chọn k riêng bằng CV cho Euclid và Manhattan."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[["x1", "x2"]], df.nhan, test_size=.25, random_state=17, stratify=df.nhan)
    print("BAI 2: KHOANG CACH EUCLID VA MANHATTAN")
    print("CV 5 phan tren train, k le tu 1 den 51; hoa thi lay k nho nhat.")
    print("Metric        k tot    CV acc   Test acc")
    ket_qua = {}
    for metric in ["euclidean", "manhattan"]:
        diem = {k: cross_val_score(KNeighborsClassifier(k, metric=metric), X_tr, y_tr, cv=5).mean()
                for k in range(1, 52, 2)}
        k_tot = max(diem, key=diem.get)
        pred = KNeighborsClassifier(k_tot, metric=metric).fit(X_tr, y_tr).predict(X_te)
        acc = accuracy_score(y_te, pred)
        ket_qua[metric] = acc
        print(f"{metric:12s} {k_tot:5d} {diem[k_tot]:9.4f} {acc:10.4f}")
    chenhlech = ket_qua["manhattan"] - ket_qua["euclidean"]
    print(f"\nManhattan - Euclid tren test: {chenhlech:+.4f} ({chenhlech*100:+.2f} diem %).")
    print("Euclid = sqrt(sum(delta^2)); Manhattan = sum(abs(delta)).")
    print("Doi metric doi thu tu hang xom va co the doi k duoc chon;")
    print("chenh lech tren mot lan chia nho khong chung minh metric nao luon tot hon.")


if __name__ == "__main__":
    main()
