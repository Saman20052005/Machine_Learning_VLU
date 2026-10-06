# -*- coding: utf-8 -*-
"""Bài tập 3: Dự đoán cho sinh viên cụ thể.

Khớp lại mô hình 1 biến gio_on trên toàn bộ tập dữ liệu.
Viết hàm du_doan(gio) tính z, xác suất p, và nhãn theo ngưỡng 0.5.
Gọi hàm với các mốc giờ [3, 8, 12.89, 18, 26].
Giải thích toán học vì sao mốc 12.89 giờ có xác suất xấp xỉ 0.5.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Đọc dữ liệu và khớp mô hình trên toàn bộ tập dữ liệu
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print("=" * 65)
print("BAI TAP 3: DU DOAN CHO SINH VIEN CU THE")
print("=" * 65)
print(f"He so goc w   = {w:.6f}")
print(f"He so chan b  = {b:.6f}")
print(f"Phuong trinh  : z = {w:.6f} * x + ({b:.6f})")
print("-" * 65)


def du_doan(gio):
    """Du doan ket qua hoc tap dua vao so gio on."""
    z = w * gio + b
    p = 1 / (1 + np.exp(-z))
    nhan = 1 if p >= 0.5 else 0
    return z, p, nhan


# 2. Chạy thử trên các mốc giờ yêu cầu
cac_moc_gio = [3, 8, 12.89, 18, 26]
print("Ket qua du doan cho cac moc gio on:")
print("  So gio on (x)      z = w*x + b     Xac suat p     Nhan du doan")
print("  " + "-" * 56)
for gio in cac_moc_gio:
    z, p, nhan = du_doan(gio)
    ket_qua_chu = "Qua mon (1)" if nhan == 1 else "Rot mon (0)"
    print(f"      {gio:5.2f}          {z:8.4f}         {p:.4f}         {ket_qua_chu}")

print("-" * 65)
print("GIAI THICH TOAN HOC CHO MOC 12.89 GIO:")
z_moc, p_moc, _ = du_doan(12.89)
print(f"1. Tinh z tai x = 12.89 gio:")
print(f"   z = w * x + b = {w:.6f} * 12.89 + ({b:.6f}) = {z_moc:.6f} ~= 0")
print(f"2. Tinh xac suat p = sigma(z):")
print(f"   sigma(z) = 1 / (1 + e^(-z))")
print(f"   Khi z = 0 => e^(-0) = e^0 = 1")
print(f"   => sigma(0) = 1 / (1 + 1) = 1 / 2 = 0.5 (xap xi {p_moc:.4f})")
print("3. Y nghia: Mien z = 0 tuong duong voi x = -b/w = 12.89 gio.")
print("   Day chinh la bien quyet dinh (decision boundary) cua mo hinh 1 bien,")
print("   noi mo hinh phan van dung 50/50 giua hai kha nang qua mon va rot mon.")
print("=" * 65)
