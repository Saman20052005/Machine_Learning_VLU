# Lab09 — Mạng nơ-ron nhiều lớp (MLP)

Bài làm theo `Lab09_MLP.pdf`, gồm 5 chương trình thực hành (`code/m1_lan_truyen_tien.py` đến `code/m5_chu_so.py`) và 6 bài tập mục 7 (`baitap09/bai1.py` đến `baitap09/bai6.py`). Bốn tệp CSV (`xor.csv`, `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv`) được sao chép nguyên bản từ nguồn cung cấp.

## Chạy bài

Mở PowerShell tại thư mục bài lab:

```powershell
cd "D:\Machine_Learning_VLU\09. MLP"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
python -X utf8 chup_ket_qua.py
```

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Tự tính lan truyền tiến bằng numpy | [m1_lan_truyen_tien.py](code/m1_lan_truyen_tien.py) | [Văn bản](ket_qua/m1_lan_truyen_tien.txt) |
| Một nơ-ron và lớp ẩn trên XOR | [m2_xor.py](code/m2_xor.py) | [Văn bản](ket_qua/m2_xor.txt) |
| Độ lớn mạng và điều chuẩn `alpha` | [m3_do_lon_mang.py](code/m3_do_lon_mang.py) | [Văn bản](ket_qua/m3_do_lon_mang.txt) |
| Mạng nơ-ron khi quên chuẩn hoá | [m4_chuan_hoa.py](code/m4_chuan_hoa.py) | [Văn bản](ket_qua/m4_chuan_hoa.txt) |
| Nhận dạng chữ số viết tay | [m5_chu_so.py](code/m5_chu_so.py) | [Văn bản](ket_qua/m5_chu_so.txt) · [Đường mất mát](bieu_do/m5_mat_mat.png) |
| Bài 1 — Tự dự đoán từ trọng số đã học | [bai1.py](baitap09/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — Một mạng XOR khác | [bai2.py](baitap09/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — So các hàm kích hoạt | [bai3.py](baitap09/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Tốc độ học (`learning_rate_init`) | [bai4.py](baitap09/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) · [Biểu đồ](bieu_do/toc_do_hoc.png) |
| Bài 5 — `GridSearchCV` cho MLP | [bai5.py](baitap09/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — `MLPRegressor` giá nhà | [bai6.py](baitap09/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

Tổng hợp toàn bộ stdout: [tat_ca_ket_qua.txt](ket_qua/tat_ca_ket_qua.txt).

## Tóm tắt kết quả chính

- **Lan truyền tiến & XOR**: mạng 2 nơ-ron ReLU tự đặt trọng số cho đúng XOR `[0, 1, 1, 0]`. Với `MLPClassifier` trên `xor.csv`, Logistic chỉ đạt `0.4500`, còn 4 và 8 nơ-ron ẩn đạt `0.9300` và `0.9100`. Bài 1 tự tính lại bằng numpy khớp `predict_proba` tới sai số `2.78e-17`.
- **Điều chuẩn & chuẩn hoá**: trên hai vầng trăng, mạng `(100, 100)` với `alpha=1e-5` học vẹt (train `1.0000`, test `0.7867`), chọn `alpha=1.0` bằng CV (`0.8578`) cho test `0.9067`. Trên `sinh_vien_mo_rong.csv`, quên chuẩn hoá làm mạng đoán tất cả rớt (`0.4000`), có chuẩn hoá đạt `0.9000`.
- **Chữ số viết tay**: `MLPClassifier(hidden_layer_sizes=(100,))` có `7510` tham số, hội tụ sau `144` vòng, đạt `0.9844` (sai 7/450 ảnh); dò lưới chọn `(50,), alpha=1` đạt `0.9822` (chưa vượt `SVC` `0.9911`).
- **Hồi quy giá nhà (Bài 6)**: `MLPRegressor` đạt MAE `0.5008` (LinearRegression `0.2578`), dự đoán căn 250 m² là `28.18` tỷ (ngoại suy tuyến tính nhưng lệch hơn đường thẳng `19.91` tỷ).
