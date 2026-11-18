# -*- coding: utf-8 -*-
"""Bài 5: Ma trận nhầm lẫn và lưu ảnh các chữ số bị SVM đoán sai."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    root = Path(__file__).resolve().parents[1]
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)

    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    svm = SVC(kernel="rbf").fit(X_tr, y_tr)
    du_doan = svm.predict(X_te)

    cm = confusion_matrix(y_te, du_doan)
    print("Ma tran nham lan 10x10 (hang = that, cot = doan):")
    print(cm)
    print()

    sai = np.flatnonzero(du_doan != y_te)
    print(f"Cac cap chu so bi nham ({len(sai)} anh tren {len(y_te)}):")
    for idx in sai:
        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(X_te[idx].reshape(8, 8), cmap="gray_r")
        plt.title(f"That {y_te[idx]}, doan {du_doan[idx]}")
        plt.axis("off")
        ten_anh = bieu_do / f"bai5_sai_anh_{idx}.png"
        plt.savefig(ten_anh, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  anh {idx:3d}: that {y_te[idx]} -> doan {du_doan[idx]} (da luu {ten_anh.name})")
    print()
    print("Nhan xet:")
    print("- Cac loi xay ra o nhung chu so co net viet tay gan giong nhau hoac bi mo/nghieng:")
    print("  9 bi nham voi 3, 8 bi nham voi 1, 4 bi nham voi 8, 7 bi nham voi 9.")


if __name__ == "__main__":
    main()
