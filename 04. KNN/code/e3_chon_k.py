"""Bước 3: khóa k bằng CV trên train trước khi đánh giá test."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[["x1", "x2"]], df.nhan, test_size=0.25, random_state=17, stratify=df.nhan)
    # Hai trục có thang đo tương tự, giữ dữ liệu thô như đề.
    diem = {k: cross_val_score(KNeighborsClassifier(k), X_tr, y_tr, cv=5).mean()
            for k in range(1, 52, 2)}
    k_tot = max(diem, key=diem.get)  # Nếu hòa, chọn k nhỏ nhất đã duyệt.
    print("CHON K BANG KIEM DINH CHEO 5 PHAN TREN TAP TRAIN")
    for k, score in diem.items():
        print(f"k={k:2d}: CV accuracy={score:.4f}" + (" <- tot nhat" if k == k_tot else ""))
    model = KNeighborsClassifier(k_tot).fit(X_tr, y_tr)
    print(f"\nk tot nhat = {k_tot}; CV accuracy = {diem[k_tot]:.4f}")
    print(f"Accuracy test cua mo hinh da chon = {accuracy_score(y_te, model.predict(X_te)):.4f}")
    print("\nBang minh hoa hoc vet/cau tha sau khi da khoa k; khong dung de chon k:")
    print("  k    Train     Test")
    for k in [1, 5, 13, 31, 101]:
        mh = KNeighborsClassifier(k).fit(X_tr, y_tr)
        print(f"{k:3d}   {accuracy_score(y_tr, mh.predict(X_tr)):.4f}   "
              f"{accuracy_score(y_te, mh.predict(X_te)):.4f}")
    print("k=1: diem train la lang gieng cua chinh no; train=1 khong bao dam tong quat.")
    print("k rat lon lam tron bien qua muc; chon k bang CV, khong bang bang test.")
    out = root / "bieu_do"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(list(diem), list(diem.values()), "o-", label="5-fold CV tren train")
    ax.axvline(k_tot, color="tomato", linestyle="--", label=f"k da chon = {k_tot}")
    ax.set(xlabel="So lang gieng k", ylabel="Accuracy trung binh", title="Chon k bang CV, khong dung test")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "e3_chon_k.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
