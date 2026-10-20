"""Bài 4: so sánh bỏ phiếu đều và trọng số nghịch khoảng cách."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[["x1", "x2"]], df.nhan, test_size=.25, random_state=17, stratify=df.nhan)
    print("BAI 4: LANG GIENG CO TRONG SO, KHOANG CACH EUCLID")
    print("  k    Weights    Train acc    Test acc")
    ket_qua = {}
    for k in [1, 13, 101]:
        for weights in ["uniform", "distance"]:
            mh = KNeighborsClassifier(k, weights=weights).fit(X_tr, y_tr)
            acc = accuracy_score(y_te, mh.predict(X_te))
            ket_qua[k, weights] = acc
            print(f"{k:3d}   {weights:8s} {accuracy_score(y_tr, mh.predict(X_tr)):10.4f} {acc:11.4f}")
    delta = ket_qua[101, "distance"] - ket_qua[101, "uniform"]
    print(f"\nk=101: distance - uniform = {delta:+.4f} ({delta*100:+.2f} diem %).")
    print("Trong so w=1/d uu tien diem gan, giam anh huong diem xa khi k lon,")
    print("nen co the giu duoc cau truc cuc bo va bot lam tron qua muc.")
    print("Ket qua thuc te o bang tren; khong dam bao weights=distance luon tot hon.")
    print("k=1: chi mot phieu nen hai cach cho cung nhan.")
    print("Khoang cach 0: sklearn chi dung cac hang xom trung diem de tranh chia 0.")
    print("Voi distance, diem train trung chinh no nen train=1 khong chung minh mo hinh tot.")


if __name__ == "__main__":
    main()
