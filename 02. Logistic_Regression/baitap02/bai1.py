# -*- coding: utf-8 -*-
"""Bài tập 1: Thống kê theo nhóm điểm giữa kỳ.

Phân tích tỷ lệ qua môn theo mốc điểm giữa kỳ >= 7.0 và < 7.0.
"""

import pandas as pd

# Đọc bộ dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

# Lọc theo điều kiện điểm giữa kỳ từ 7.0 trở lên và nhóm còn lại
nhom_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_thap = df[df["diem_giua_ky"] < 7.0]

so_ban_cao = len(nhom_cao)
so_ban_thap = len(nhom_thap)
ty_le_cao = nhom_cao["qua_mon"].mean()
ty_le_thap = nhom_thap["qua_mon"].mean()

print("=" * 60)
print("BAI TAP 1: THONG KE THEO NHOM DIEM GIUA KY")
print("=" * 60)
print(f"1. So ban co diem giua ky >= 7.0: {so_ban_cao} ban (tren tong so {len(df)} sinh vien)")
print(f"2. Ty le qua mon cua nhom diem giua ky >= 7.0: {ty_le_cao:.4f} ({ty_le_cao * 100:.2f}%)")
print(f"3. Ty le qua mon cua nhom con lai (< 7.0)    : {ty_le_thap:.4f} ({ty_le_thap * 100:.2f}%)")
print("-" * 60)
print("NHAN XET:")
print("Diem giua ky phan biet rat ro hai nhom sinh vien:")
print(f"- Nhom dat diem giua ky >= 7.0 co ty le qua mon rat cao ({ty_le_cao * 100:.2f}%).")
print(f"- Nhom duoi 7.0 chi dat ty le qua mon {ty_le_thap * 100:.2f}%.")
print("Do do, diem giua ky la mot dac trung co gia tri phan loai manh,")
print("giup mo hinh phan biet tot hon giua nhom qua mon va rot mon.")
print("=" * 60)
