# Lab08 — Phân cụm K-Means

Bài làm theo `Lab08_KMeans.pdf`, gồm 5 chương trình thực hành (`code/k1_tu_viet.py` đến `code/k5_nen_anh.py`) và 6 bài tập mục 7 (`baitap08/bai1.py` đến `baitap08/bai6.py`). Các tệp CSV (`khach_hang.csv`, `hai_vang_trang.csv`) được sao chép nguyên bản từ nguồn cung cấp.

## Chạy bài

Mở PowerShell tại thư mục bài lab:

```powershell
cd "D:\Machine_Learning_VLU\08. K-Means"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
python -X utf8 chup_ket_qua.py
```

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Tự viết K-Means bằng numpy | [k1_tu_viet.py](code/k1_tu_viet.py) | [Văn bản](ket_qua/k1_tu_viet.txt) |
| Phân khúc khách hàng | [k2_khach_hang.py](code/k2_khach_hang.py) | [Văn bản](ket_qua/k2_khach_hang.txt) |
| Chọn số cụm K (khuỷu tay & silhouette) | [k3_chon_k.py](code/k3_chon_k.py) | [Văn bản](ket_qua/k3_chon_k.txt) · [Biểu đồ](bieu_do/k3_chon_k.png) |
| Ba cái bẫy của K-Means | [k4_ba_cai_bay.py](code/k4_ba_cai_bay.py) | [Văn bản](ket_qua/k4_ba_cai_bay.txt) |
| Nén màu ảnh | [k5_nen_anh.py](code/k5_nen_anh.py) | [Văn bản](ket_qua/k5_nen_anh.txt) |
| Bài 1 — Tự tính silhouette | [bai1.py](baitap08/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — Thêm cột tuổi | [bai2.py](baitap08/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — `k-means++` vs `random` | [bai3.py](baitap08/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Lưu ảnh nén 16 màu & bỏ kênh alpha | [bai4.py](baitap08/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) · [Ảnh 16 màu](bieu_do/anh_16_mau.png) |
| Bài 5 — DBSCAN trên hai vầng trăng | [bai5.py](baitap08/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — Phân cụm chữ số viết tay | [bai6.py](baitap08/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

Tổng hợp toàn bộ stdout: [tat_ca_ket_qua.txt](ket_qua/tat_ca_ket_qua.txt).

## Tóm tắt kết quả chính

- **7 điểm A–G**: K-Means tự viết dừng sau 3 vòng, `inertia = 8.5250`, trùng khớp `KMeans(n_clusters=2, init=P[[0, 1]], n_init=1)`.
- **Khách hàng**: với 2 cột (`thu_nhap`, `diem_chi_tieu`), `K = 5` cho `inertia = 23.02` và `silhouette = 0.723` (Bài 1 tự tính `0.722706`, khách chỉ số `113` có `s = 0.0286` nhỏ nhất). Thêm cột `tuoi` (Bài 2), `K = 5` vẫn cao nhất nhưng silhouette giảm còn `0.4808`.
- **Khởi tạo (Bài 3)**: `k-means++` đạt `inertia = 23.0` ở `50/50` lần (`100%`), còn `random` chỉ đạt `30/50` lần (`60%`).
- **Nén ảnh (Bài 4)**: `china.jpg` giảm từ `537.3 KB` xuống `131.3 KB` (`-75.6%`); `flower_rgba.png` (`427x640x4`) bỏ kênh alpha còn `(427, 640, 3)` và nén 16 màu từ `409.6 KB` xuống `54.3 KB` (`-86.7%`).
- **DBSCAN (Bài 5)**: với `eps` bằng `0.2, 0.3, 0.4` (`min_samples=5`), đều gom thành `1` cụm (số nhiễu lần lượt `40, 10, 3`) do hai vầng trăng có nhiễu nằm gần nhau.
