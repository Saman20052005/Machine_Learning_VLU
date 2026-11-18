# Lab06 — Máy vectơ hỗ trợ (SVM)

Bài làm theo `Lab06_SVM.pdf`, gồm 5 chương trình thực hành (`code/g1_le_rong_nhat.py` đến `code/g5_chu_so.py`) và 6 bài tập mục 7 (`baitap06/bai1.py` đến `baitap06/bai6.py`). Các tệp CSV (`sinh_vien.csv`, `sinh_vien_mo_rong.csv`, `hai_vang_trang.csv`) được sao chép nguyên bản từ nguồn cung cấp.

## Chạy bài

Mở PowerShell tại thư mục bài lab:

```powershell
cd "D:\Machine_Learning_VLU\06. SVM"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
python -X utf8 chup_ket_qua.py
```

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Đường có lề rộng nhất | [g1_le_rong_nhat.py](code/g1_le_rong_nhat.py) | [Văn bản](ket_qua/g1_le_rong_nhat.txt) |
| Lề mềm, tham số C và chuẩn hoá | [g2_tham_so_c.py](code/g2_tham_so_c.py) | [Văn bản](ket_qua/g2_tham_so_c.txt) |
| Nâng chiều bằng tay và kernel | [g3_kernel.py](code/g3_kernel.py) | [Văn bản](ket_qua/g3_kernel.txt) |
| Ảnh hưởng của gamma và dò lưới | [g4_chon_tham_so.py](code/g4_chon_tham_so.py) | [Văn bản](ket_qua/g4_chon_tham_so.txt) |
| Nhận dạng chữ số viết tay | [g5_chu_so.py](code/g5_chu_so.py) | [Văn bản](ket_qua/g5_chu_so.txt) |
| Bài 1 — Tính tay khoảng cách tới ranh giới | [bai1.py](baitap06/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — Thêm một điểm mới | [bai2.py](baitap06/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — Kernel đa thức | [bai3.py](baitap06/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Dò lưới mịn | [bai4.py](baitap06/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) |
| Bài 5 — Ma trận nhầm lẫn chữ số | [bai5.py](baitap06/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — SVM trên dữ liệu lớn | [bai6.py](baitap06/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

Tổng hợp toàn bộ stdout: [tat_ca_ket_qua.txt](ket_qua/tat_ca_ket_qua.txt).

## Tóm tắt kết quả chính

- **24 điểm tách được**: `w1 = -0.831, w2 = -0.355, b = 4.820`, độ rộng lề `2.2126`, có `3` vectơ hỗ trợ.
- **Lề mềm & chuẩn hoá**: tăng `C` từ `0.01` lên `100` giảm số vectơ hỗ trợ từ `74` xuống `27`. Trên `sinh_vien_mo_rong.csv`, không chuẩn hoá chỉ đạt `0.6000` (bằng đoán bừa), có chuẩn hoá đạt `0.9000`.
- **Hai vầng trăng & dò lưới**: kernel RBF đạt `0.9200` test; lưới thưa chọn `C=10, gamma=1` (`CV=0.8711`, `test=0.9200`), lưới mịn chọn `C=30, gamma=1` (`CV=0.8756`, `test=0.9200`). Kernel đa thức bậc 3 tốt nhất chỉ đạt `0.8267` test.
- **Chữ số viết tay**: `SVC(kernel="rbf")` đạt `0.9911` (sai 4/450 ảnh), vượt KNN (`0.9867`) và Logistic có chuẩn hoá (`0.9711`).
