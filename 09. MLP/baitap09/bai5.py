# -*- coding: utf-8 -*-
"""Bài 5: GridSearchCV cho MLP trên bộ chữ số viết tay."""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(max_iter=1000, random_state=0),
    )
    luoi = {
        "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
        "mlpclassifier__alpha": [0.0001, 0.01, 1],
    }
    tim = GridSearchCV(pipe, luoi, cv=5, n_jobs=-1)
    tim.fit(X_tr, y_tr)

    acc_mlp = tim.score(X_te, y_te)
    acc_svm = accuracy_score(y_te, SVC().fit(X_tr, y_tr).predict(X_te))

    print("Cap tham so tot nhat:", tim.best_params_)
    print(f"Diem CV tot nhat    : {tim.best_score_:.4f}")
    print(f"Accuracy test (MLP) : {acc_mlp:.4f}")
    print(f"Accuracy test (SVM) : {acc_svm:.4f}")
    print()
    if acc_mlp > acc_svm:
        print("Nhan xet: MLP sau do luoi da vuot qua SVM tren tap kiem tra.")
    elif acc_mlp == acc_svm:
        print("Nhan xet: MLP sau do luoi dat accuracy ngang bang SVM.")
    else:
        print("Nhan xet: MLP sau do luoi van chua vuot duoc SVM (0.9911) tren bo du lieu nho nay.")


if __name__ == "__main__":
    main()
