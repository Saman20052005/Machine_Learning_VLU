# -*- coding: utf-8 -*-
"""Bài tập 2: Vẽ hàm sigmoid.

Vẽ đồ thị hàm sigmoid với z chạy từ -8 tới 8, lưu thành tệp baitap02/sigmoid.png.
Thêm đường ngang tại y = 0.5, đường dọc tại z = 0 và điểm (0, 0.5).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Tạo dải giá trị z từ -8 tới 8
z = np.linspace(-8, 8, 400)
sigma = 1 / (1 + np.exp(-z))

# Khởi tạo đồ thị
plt.figure(figsize=(8, 5))
plt.plot(z, sigma, color="navy", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")

# Đường nét đứt ngang y = 0.5 và dọc z = 0
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="y = 0.5")
plt.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="z = 0")

# Đánh dấu điểm gốc (0, 0.5)
plt.scatter([0], [0.5], color="red", s=60, zorder=5, label=r"Diem goc $(0, 0.5)$")
plt.text(0.3, 0.47, r"$\sigma(0) = 0.5$", color="red", fontsize=11, fontweight="bold")

# Thiết lập giới hạn, tiêu đề, nhãn trục
plt.title("Do thi ham Sigmoid", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("z", fontsize=12)
plt.ylabel(r"$\sigma(z)$", fontsize=12)
plt.ylim(-0.05, 1.05)
plt.xlim(-8.5, 8.5)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right", fontsize=10)

# Tạo thư mục nếu chưa có và lưu file
os.makedirs("baitap02", exist_ok=True)
plt.savefig("baitap02/sigmoid.png", dpi=300, bbox_inches="tight")
plt.close()

print("Da ve do thi ham sigmoid va luu tai: baitap02/sigmoid.png")
