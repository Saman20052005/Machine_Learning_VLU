"""Bước 2: thống kê dữ liệu và biểu diễn túi từ."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    print("Kich thuoc bang:", df.shape)
    print("So tin theo nhan (0: thuong, 1: rac):")
    print(df.la_rac.value_counts().sort_index().to_string())
    print(f"Ty le tin rac: {df.la_rac.mean():.4f}")
    print("\nBa tin nhan dau tien:")
    for row in df.head(3).itertuples():
        print(f"[{row.la_rac}] {row.noi_dung}")
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df.noi_dung)
    print("\nKich thuoc bang dem tu:", X.shape)
    print("So tu trong tu dien:", len(vec.vocabulary_))
    tong = pd.DataFrame({
        "thuong": X[df.la_rac.to_numpy() == 0].sum(axis=0).A1,
        "rac": X[df.la_rac.to_numpy() == 1].sum(axis=0).A1,
    }, index=vec.get_feature_names_out())
    for nhan in ["rac", "thuong"]:
        print(f"\nMuoi tu xuat hien nhieu nhat trong tin {nhan}:")
        print(tong.sort_values(nhan, ascending=False, kind="stable").head(10).to_string())


if __name__ == "__main__":
    main()
