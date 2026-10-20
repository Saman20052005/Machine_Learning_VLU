"""Bài 4: tự cài bộ lọc bằng thư viện chuẩn; sklearn chỉ dùng để đối chiếu."""
from collections import Counter
from math import log
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesTuViet:
    """Tự tách từ, đếm, làm trơn và tính tổng loga, không dùng bộ học sklearn."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha phai duong de tinh log an toan")
        self.alpha = alpha

    @staticmethod
    def tach_tu(cau):
        return re.findall(r"(?u)\S+", cau.lower())

    def fit(self, van_ban, nhan):
        van_ban, nhan = list(van_ban), list(nhan)
        self.lop = sorted(set(nhan))
        self.dem_tin = Counter(nhan)
        self.dem_tu = {lop: Counter() for lop in self.lop}
        self.tu_dien = set()
        for cau, lop in zip(van_ban, nhan):
            tu = self.tach_tu(cau)
            self.dem_tu[lop].update(tu)
            self.tu_dien.update(tu)
        self.log_prior = {lop: log(self.dem_tin[lop] / len(nhan)) for lop in self.lop}
        self.log_prob = {}
        for lop in self.lop:
            mau_so = sum(self.dem_tu[lop].values()) + self.alpha * len(self.tu_dien)
            self.log_prob[lop] = {
                tu: log((self.dem_tu[lop][tu] + self.alpha) / mau_so)
                for tu in self.tu_dien}
        return self

    def tinh_diem(self, cau):
        dem = Counter(self.tach_tu(cau))
        return [self.log_prior[lop] + sum(
            so_lan * self.log_prob[lop][tu]
            for tu, so_lan in dem.items() if tu in self.tu_dien)
            for lop in self.lop]

    def predict(self, van_ban):
        # Khi hòa điểm, chọn lớp nhỏ hơn, giống thứ tự classes_ của sklearn.
        return [self.lop[max(range(len(self.lop)), key=diem.__getitem__)]
                for diem in map(self.tinh_diem, van_ban)]


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    train, test, y_train, y_test = train_test_split(
        df.noi_dung, df.la_rac, test_size=0.25, random_state=17, stratify=df.la_rac)
    tu_viet = NaiveBayesTuViet(alpha=1.0).fit(train, y_train)
    pred_tay = tu_viet.predict(test)
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train, X_test = vec.fit_transform(train), vec.transform(test)
    thu_vien = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    pred_sk = thu_vien.predict(X_test)
    print("BAI 4: BO LOC TU VIET VA MULTINOMIALNB")
    print(f"Tap hoc: {len(train)}; tap test: {len(test)}; tu dien: {len(tu_viet.tu_dien)}")
    print(f"Accuracy tu viet      = {accuracy_score(y_test, pred_tay):.4f}")
    print(f"Accuracy MultinomialNB = {accuracy_score(y_test, pred_sk):.4f}")
    print(f"So du doan khac nhau  = {np.count_nonzero(pred_sk != pred_tay)}")
    # So cả điểm log hậu nghiệm chưa chuẩn hóa, không chỉ accuracy.
    diem_tay = np.array([tu_viet.tinh_diem(cau) for cau in test])
    diem_sk = X_test @ thu_vien.feature_log_prob_.T + thu_vien.class_log_prior_
    print(f"Sai so diem log lon nhat = {np.max(np.abs(diem_tay - diem_sk)):.3e}")
    np.testing.assert_allclose(diem_tay, diem_sk, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(pred_tay, pred_sk)
    # Tin chỉ có từ ngoài từ điển phải được quyết định bằng tiên nghiệm.
    assert tu_viet.predict(["zzzzzzunknown"])[0] == y_train.value_counts().idxmax()
    print("\nGiai thich: score(k) = log P(k) + sum count(tu)*log P(tu|k).")
    print("P(tu|k) = (dem(tu,k)+alpha)/(tong_tu(k)+alpha*V).")
    print("Laplace tranh log(0); cong log tranh tran duoi khi nhan nhieu so nho.")
    print("Hai bo loc cung tap chia, cach tach tu, tu dien, alpha va tien nghiem,")
    print("nen cho cung diem (sai so so thuc rat nho) va cung du doan.")
    print("Tu ngoai tu dien bi bo qua; tin khong co tu da biet chi dung prior.")
    print("Neu lech: kiem tra lower(), tu mot ky tu, so lan lap, mau so alpha*V,")
    print("va viec chi hoc tu dien tren tap train, khong hoc lai tren tap test.")


if __name__ == "__main__":
    main()
