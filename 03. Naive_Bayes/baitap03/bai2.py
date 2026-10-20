"""Bài 2: xếp hạng từ theo log P(từ|rác) - log P(từ|thường)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    model = MultinomialNB(alpha=1.0).fit(vec.fit_transform(df.noi_dung), df.la_rac)
    hieu = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    tu = vec.get_feature_names_out()
    print("BAI 2: TU NAO TO CAO MANH NHAT (HOC TOAN BO 300 TIN)")
    print("Diem = log P(tu|rac) - log P(tu|thuong)")
    for ten, chi_so in [("5 tu nghieng ve tin rac", np.argsort(-hieu, kind="stable")[:5]),
                        ("5 tu nghieng ve tin thuong", np.argsort(hieu, kind="stable")[:5])]:
        print(f"\n{ten}:")
        for i in chi_so:
            print(f"{tu[i]:12s} {hieu[i]:+.6f}")
    print("\nDiem duong ung ho lop rac; diem am ung ho lop thuong.")
    print("Xep hang theo ty so xac suat, khong chi theo tan suat tho.")


if __name__ == "__main__":
    main()
