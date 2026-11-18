# -*- coding: utf-8 -*-
"""Bài 2: Báo cáo phân loại cho 10 chữ số viết tay của SVM."""

from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    svm = SVC(kernel="rbf").fit(X_tr, y_tr)
    du = svm.predict(X_te)

    print(classification_report(y_te, du, digits=4))

    rep = classification_report(y_te, du, output_dict=True)
    lop = [str(i) for i in range(10)]
    min_r = min(lop, key=lambda k: rep[k]["recall"])
    min_p = min(lop, key=lambda k: rep[k]["precision"])
    print(f"Chu so co recall thap nhat   : {min_r} ({rep[min_r]['recall']:.4f})")
    print(f"Chu so co precision thap nhat: {min_p} ({rep[min_p]['precision']:.4f})")
    print()
    print("Giai thich:")
    print("- Macro avg va weighted avg gan nhu bang nhau vi so luong anh cua 10 chu so")
    print("  trong tap kiem tra gan nhu can bang (moi lop khoang 43-46 anh), nen trong so")
    print("  theo so mau khong khac nhieu so voi trung binh cong deu.")


if __name__ == "__main__":
    main()
