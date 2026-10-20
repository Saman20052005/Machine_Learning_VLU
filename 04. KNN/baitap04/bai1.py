"""Bài 1: KNN tự viết bằng numpy, sklearn chỉ dùng chia tập và đối chứng."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Tính Euclid, lấy k láng giềng và bỏ phiếu; hòa thì chọn nhãn nhỏ nhất."""
    X_hoc = np.asarray(X_hoc, dtype=float)
    y_hoc = np.asarray(y_hoc)
    x_moi = np.asarray(x_moi, dtype=float)
    if X_hoc.ndim != 2 or x_moi.shape != (X_hoc.shape[1],):
        raise ValueError("Sai kich thuoc dac trung")
    if y_hoc.shape != (len(X_hoc),) or not 1 <= k <= len(X_hoc):
        raise ValueError("Sai kich thuoc nhan hoac k ngoai khoang hop le")
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    gan = np.argsort(khoang_cach, kind="stable")[:k]
    nhan, phieu = np.unique(y_hoc[gan], return_counts=True)
    return nhan[np.argmax(phieu)]


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/hai_vang_trang.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[["x1", "x2"]].to_numpy(), df.nhan.to_numpy(),
        test_size=.25, random_state=17, stratify=df.nhan)
    pred_tay = np.array([knn_du_doan(X_tr, y_tr, x, 13) for x in X_te])
    model = KNeighborsClassifier(13).fit(X_tr, y_tr)
    pred_sk = model.predict(X_te)
    print("BAI 1: KNN TU VIET, k=13, EUCLID, KHONG CHUAN HOA NHU DE")
    print(f"Train: {len(X_tr)}; test: {len(X_te)}")
    print(f"Accuracy tu viet = {accuracy_score(y_te, pred_tay):.4f}")
    print(f"Accuracy sklearn = {accuracy_score(y_te, pred_sk):.4f}")
    print(f"So du doan khac nhau = {np.count_nonzero(pred_tay != pred_sk)}")
    np.testing.assert_array_equal(pred_tay, pred_sk)
    # Kiểm tra thêm quy ước hòa phiếu trên một ví dụ nhỏ có đáp án rõ ràng.
    assert knn_du_doan([[0], [2]], [1, 0], [1], 2) == 0
    print("\nGiai thich: tinh khoang cach toi tung mau train, sap tang dan bang argsort,")
    print("lay 13 nhan gan nhat, dem so phieu va chon lop nhieu phieu nhat.")
    print("k=13 le va chi co hai lop nen khong hoa phieu. Ham van xu ly hoa")
    print("bang cach chon nhan nho nhat, giong quy uoc bo phieu cua sklearn.")
    print("Cung du lieu, metric va k nen cung ket qua; accuracy bang nhau thoi chua du,")
    print("vi vay chuong trinh con kiem tra tung nhan du doan.")
    print("Neu hoa khoang cach o vi tri k, ham uu tien thu tu dong train;")
    print("backend sklearn co the chon khac. Bo nay khong gay lech du doan.")


if __name__ == "__main__":
    main()
