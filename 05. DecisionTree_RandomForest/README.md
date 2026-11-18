# Lab05 — Cây quyết định và Rừng ngẫu nhiên

Bài làm theo `Lab05_CayQuyetDinh_RungNgauNhien.pdf`, gồm 7 chương trình thực hành (`code/f1_gini.py` đến `code/f7_cay_hoi_quy.py`) và 6 bài tập mục 9 (`baitap05/bai1.py` đến `baitap05/bai6.py`). Hai tệp dữ liệu (`khach_vay.csv`, `gia_nha.csv`) được sao chép nguyên bản từ nguồn cung cấp.

## Chạy bài

Mở PowerShell tại thư mục bài lab:

```powershell
cd "D:\Machine_Learning_VLU\05. DecisionTree_RandomForest"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
python -X utf8 chup_ket_qua.py
```

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Tự tính Gini và chọn câu hỏi | [f1_gini.py](code/f1_gini.py) | [Văn bản](ket_qua/f1_gini.txt) |
| Cây năm lá trên dữ liệu khách vay | [f2_cay_nam_la.py](code/f2_cay_nam_la.py) | [Văn bản](ket_qua/f2_cay_nam_la.txt) |
| Cây học vẹt và chọn độ sâu | [f3_do_sau.py](code/f3_do_sau.py) | [Văn bản](ket_qua/f3_do_sau.txt) · [Biểu đồ](bieu_do/f3_do_sau.png) |
| Cây không cần chuẩn hoá | [f4_khong_chuan_hoa.py](code/f4_khong_chuan_hoa.py) | [Văn bản](ket_qua/f4_khong_chuan_hoa.txt) |
| Trồng rừng ngẫu nhiên | [f5_rung.py](code/f5_rung.py) | [Văn bản](ket_qua/f5_rung.txt) · [Biểu đồ](bieu_do/f5_rung.png) |
| Độ quan trọng của đặc trưng | [f6_do_quan_trong.py](code/f6_do_quan_trong.py) | [Văn bản](ket_qua/f6_do_quan_trong.txt) · [Biểu đồ](bieu_do/f6_do_quan_trong.png) |
| Cây cho bài toán hồi quy | [f7_cay_hoi_quy.py](code/f7_cay_hoi_quy.py) | [Văn bản](ket_qua/f7_cay_hoi_quy.txt) · [Biểu đồ](bieu_do/f7_cay_hoi_quy.png) |
| Bài 1 — Tự vẽ cây | [bai1.py](baitap05/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) · [Cây 5 lá](bieu_do/cay_nam_la.png) |
| Bài 2 — Tìm câu hỏi gốc | [bai2.py](baitap05/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — Gini hay entropy | [bai3.py](baitap05/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — `min_samples_leaf` | [bai4.py](baitap05/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) |
| Bài 5 — `permutation_importance` | [bai5.py](baitap05/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — Rừng hồi quy giá nhà | [bai6.py](baitap05/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

Tổng hợp toàn bộ stdout: [tat_ca_ket_qua.txt](ket_qua/tat_ca_ket_qua.txt).

## Tóm tắt kết quả chính

- **Gini (10 khách)**: Gini gốc `0.4800`, ngưỡng tốt nhất `thu_nhap <= 12.0` (Gini giảm `0.1633`), trùng khớp `DecisionTreeClassifier(max_depth=1)`.
- **Cây 5 lá**: train `0.7978`, test `0.7467`. Cây tự do sâu `13` với `92` lá (train `1.0000`, test `0.7667`). Chọn `max_depth=5` bằng CV 5 phần (`0.8333`) cho test `0.8533`.
- **Rừng 200 cây**: test `0.9000`. Khi thêm cột `so_ngau_nhien`, `feature_importances_` xếp cột này hạng 3/5 (`0.1499`), còn `permutation_importance` trên test đưa về hạng 5/5 (`0.0017`).
- **Hồi quy giá nhà**: `RandomForestRegressor` (200 cây) đạt MAE test `0.4559` (cây sâu 3: `0.5039`, đường thẳng: `0.2578`), dự đoán căn 250 m² là `9.20` tỷ (không ngoại suy được).
