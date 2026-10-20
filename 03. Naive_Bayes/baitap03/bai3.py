"""Bài 3: so sánh năm hệ số làm trơn trên cùng một tập kiểm tra."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    train, test, y_train, y_test = train_test_split(
        df.noi_dung, df.la_rac, test_size=0.25, random_state=17, stratify=df.la_rac)
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train, X_test = vec.fit_transform(train), vec.transform(test)
    print("BAI 3: DOI THAM SO LAM TRON")
    print(f"{'Alpha':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s}")
    for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
        pred = MultinomialNB(alpha=alpha).fit(X_train, y_train).predict(X_test)
        print(f"{alpha:8.3f} {accuracy_score(y_test, pred):10.4f} "
              f"{precision_score(y_test, pred, zero_division=0):10.4f} "
              f"{recall_score(y_test, pred):10.4f}")
    print("\nNhan xet: alpha qua lon keo P(tu|lop) ve 1/V, lam mat")
    print("suc phan biet cua tu; tien nghiem lop thuong dan chi phoi.")
    print("Trong gioi han alpha tien toi vo cung, mo hinh chon lop pho bien.")
    print("Tren tap nay, cac alpha co the cung ket qua; khong suy ra alpha vo hai.")
    print("Khong chon alpha toi uu bang tap test; can tap validation rieng.")


if __name__ == "__main__":
    main()
