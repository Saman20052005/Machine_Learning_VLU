# -*- coding: utf-8 -*-
"""Bài tập 5: Dò ngưỡng tối ưu theo F1-score.

Duyệt ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.05.
Tính và in bảng F1-score theo từng ngưỡng.
Tìm ngưỡng cho F1 cao nhất và nhận xét đánh giá.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. Chuẩn bị dữ liệu và huấn luyện mô hình
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

# Lấy xác suất lớp 1 (qua môn)
p = mo_hinh.predict_proba(X_test)[:, 1]

print("=" * 55)
print("BAI TAP 5: DO NGUONG TOI UU THEO F1-SCORE")
print("=" * 55)
print(f"{'Nguong (t)':<12} | {'F1-Score':<10} | {'So ban doan qua'}")
print("-" * 55)

nguong_list = np.arange(0.05, 1.0, 0.05)
ket_qua = []

for nguong in nguong_list:
    y_pred = (p >= nguong).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    so_ban_qua = int(y_pred.sum())
    ket_qua.append((nguong, f1, so_ban_qua))
    print(f"{nguong:<12.2f} | {f1:<10.4f} | {so_ban_qua:3d}")

print("-" * 55)

# Tìm ngưỡng có F1 cao nhất
f1_max = max(item[1] for item in ket_qua)
cac_nguong_max = [item for item in ket_qua if np.isclose(item[1], f1_max)]

print(f"F1-score cao nhat dat duoc: {f1_max:.4f}")
print("Cac nguong cho F1 cao nhat:")
for ng, f1, so_ban in cac_nguong_max:
    print(f"  - Nguong t = {ng:.2f} (F1 = {f1:.4f}, so ban doan qua: {so_ban})")

print("-" * 55)
print("NHAN XET:")
co_05 = any(np.isclose(ng, 0.5) for ng, _, _ in cac_nguong_max)
if co_05:
    print("- Nguong 0.5 nam trong dai nguong cho F1 toi uu cao nhat (khoang tu 0.30 den 0.55).")
else:
    nguong_opt = cac_nguong_max[0][0]
    print(f"- Nguong toi uu theo F1 la {nguong_opt:.2f}, khong nhat thiet phai bang dung 0.5.")
print("- Giai thich nguyen nhan:")
print("  + Nguong 0.5 chi la mac dinh mang tinh truc giac doi xung.")
print("  + F1-score la trung binh dieu hoa giua Precision va Recall. Khi thay doi nguong,")
print("    Precision va Recall bien thien nguoc chieu nhau. Vung nguong toi uu la noi")
print("    hai chi so nay dung hoa tot nhat tren phan bo xac suat cua tap du lieu.")
print("=" * 55)
