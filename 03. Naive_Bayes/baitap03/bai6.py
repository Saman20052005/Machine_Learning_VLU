"""Bài 6: lớp rác hiếm và cái bẫy accuracy."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    # head(20) giữ đúng 20 tin rác đầu tiên theo thứ tự trong CSV.
    lech = pd.concat([df[df.la_rac == 0], df[df.la_rac == 1].head(20)], ignore_index=True)
    train, test, y_train, y_test = train_test_split(
        lech.noi_dung, lech.la_rac, test_size=0.25, random_state=17, stratify=lech.la_rac)
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train, X_test = vec.fit_transform(train), vec.transform(test)
    model = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    pred = model.predict(X_test)
    print("BAI 6: LOP HIEM VA XAC SUAT TIEN NGHIEM")
    print("Du lieu moi: 180 tin thuong, 20 tin rac, tong 200 tin.")
    print("Train (thuong, rac):", y_train.value_counts().sort_index().tolist())
    print("Test  (thuong, rac):", y_test.value_counts().sort_index().tolist())
    print("Tien nghiem (thuong, rac):", np.exp(model.class_log_prior_).round(4))
    print(f"Accuracy  = {accuracy_score(y_test, pred):.4f}")
    print(f"Precision = {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall rac (pos_label=1) = {recall_score(y_test, pred, pos_label=1):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    print(f"TN = {tn}; FP = {fp}; FN = {fn}; TP = {tp}")
    baseline = np.zeros(len(y_test), dtype=int)
    print(f"\nLuon doan thuong: accuracy = {accuracy_score(y_test, baseline):.4f}; "
          f"recall rac = {recall_score(y_test, baseline, pos_label=1):.4f}")
    print("Nhan xet: accuracy cao chua du ket luan bo loc tot vi 90% tin la thuong.")
    print("Ngay ca bo loc bo qua toan bo tin rac van dat accuracy 90%.")
    print("Prior 0.9/0.1 nghieng ve lop thuong: dau hieu tu phai bu lai loi the 9:1.")
    print("Chi 15 tin rac de hoc con lam uoc luong xac suat tu kem on dinh.")
    print("Can xem recall rac de biet bao nhieu rac bi lot, kem precision/FP")
    print("de tranh chan nham tin thuong. Test chi co 5 tin rac: moi tin doi recall 20%.")
    print("Khong suy rong chat luong tu mot lan chia nho; can them du lieu kiem tra.")


if __name__ == "__main__":
    main()
