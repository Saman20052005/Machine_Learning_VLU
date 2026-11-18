# -*- coding: utf-8 -*-
"""Bài 1: Tự vẽ cây năm lá và cây sâu 2 bằng plot_tree."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def main():
    root = Path(__file__).resolve().parents[1]
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)

    df = pd.read_csv(root / "data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    ten_cot = list(X.columns)
    ten_lop = ["tre han", "dung han"]

    # Cây năm lá như mục 3
    cay_5 = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
    plt.figure(figsize=(14, 8))
    plot_tree(cay_5, feature_names=ten_cot, class_names=ten_lop,
              filled=True, rounded=True, fontsize=10)
    plt.title("Cay quyet dinh nam la")
    plt.tight_layout()
    duong_dan_5 = bieu_do / "cay_nam_la.png"
    plt.savefig(duong_dan_5, dpi=150)
    plt.close()

    # Cây sâu 2
    cay_2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
    plt.figure(figsize=(12, 6))
    plot_tree(cay_2, feature_names=ten_cot, class_names=ten_lop,
              filled=True, rounded=True, fontsize=10)
    plt.title("Cay quyet dinh sau 2")
    plt.tight_layout()
    duong_dan_2 = bieu_do / "cay_sau_2.png"
    plt.savefig(duong_dan_2, dpi=150)
    plt.close()

    print("Da luu hinh:", duong_dan_5.relative_to(root).as_posix())
    print("Da luu hinh:", duong_dan_2.relative_to(root).as_posix())
    print()
    print("Cay sau 2 dang chu:")
    print(export_text(cay_2, feature_names=ten_cot, show_weights=True, decimals=3))
    print("Nhan xet:")
    print("- Hai la ben trai cua cay sau 2 deu co cung ket luan la class 0 (tre han).")
    print("- Cay van chia nut do vi phep chia lam giam chi so Gini trung binh co trong so:")
    print("  mot nhanh tro nen rat thuan khiet (114 tre han, 14 dung han, gini=0.195),")
    print("  con nhanh kia gan 50-50 (68 tre han, 53 dung han). Cay chi nhin muc giam Gini,")
    print("  khong quan tam hai la con co cung nhan du doan hay khong.")


if __name__ == "__main__":
    main()
