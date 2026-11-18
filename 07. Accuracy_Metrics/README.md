# Lab07 — Đánh giá mô hình

Bài làm theo `Lab07_DanhGiaMoHinh.pdf`, gồm 6 chương trình thực hành (`code/h1_do_bua.py` đến `code/h6_can_bang.py`) và 6 bài tập mục 8 (`baitap07/bai1.py` đến `baitap07/bai6.py`). Hai tệp dữ liệu (`sinh_vien_bo_hoc.csv`, `gia_nha.csv`) được sao chép nguyên bản từ nguồn cung cấp.

## Chạy bài

Mở PowerShell tại thư mục bài lab:

```powershell
cd "D:\Machine_Learning_VLU\07. Accuracy_Metrics"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
python -X utf8 chup_ket_qua.py
```

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Accuracy đánh lừa trên dữ liệu lệch lớp | [h1_do_bua.py](code/h1_do_bua.py) | [Văn bản](ket_qua/h1_do_bua.txt) |
| Tự đếm ma trận nhầm lẫn | [h2_ma_tran.py](code/h2_ma_tran.py) | [Văn bản](ket_qua/h2_ma_tran.txt) |
| Precision, recall và F1 | [h3_precision_recall.py](code/h3_precision_recall.py) | [Văn bản](ket_qua/h3_precision_recall.txt) |
| Đổi ngưỡng quyết định | [h4_nguong.py](code/h4_nguong.py) | [Văn bản](ket_qua/h4_nguong.txt) |
| Đường ROC, AUC và so các mô hình | [h5_roc_auc.py](code/h5_roc_auc.py) | [Văn bản](ket_qua/h5_roc_auc.txt) |
| Kiểm định chéo nhiều thước đo & cân bằng lớp | [h6_can_bang.py](code/h6_can_bang.py) | [Văn bản](ket_qua/h6_can_bang.txt) |
| Bài 1 — Tự viết `danh_gia` bằng numpy | [bai1.py](baitap07/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — `classification_report` 10 chữ số | [bai2.py](baitap07/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — Chọn ngưỡng đạt recall >= 0.80 | [bai3.py](baitap07/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Tự vẽ đường ROC | [bai4.py](baitap07/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) · [Biểu đồ](bieu_do/roc.png) |
| Bài 5 — `class_weight="balanced"` | [bai5.py](baitap07/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — MAE, RMSE, R² khi có ngoại lệ +5 tỷ | [bai6.py](baitap07/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

Tổng hợp toàn bộ stdout: [tat_ca_ket_qua.txt](ket_qua/tat_ca_ket_qua.txt).

## Tóm tắt kết quả chính

- **Dữ liệu lệch lớp**: tỷ lệ bỏ học `0.099` (`119/1200`). Mô hình lười đạt accuracy `0.9000` nhưng recall `0.0000`; Logistic đạt `0.9467` (`TP=17, FP=3, FN=13, TN=267`, precision `0.8500`, recall `0.5667`, F1 `0.6800`).
- **Đổi ngưỡng & cân bằng lớp**: ngưỡng chọn trên tập học bằng `cross_val_predict` cho F1 cao nhất là `0.25` (test: precision `0.6571`, recall `0.7667`, F1 `0.7077`). Với yêu cầu recall `>= 0.80` (Bài 3), ngưỡng `0.17` đạt test recall `0.8000`, precision `0.5106`, báo `47/300` bạn.
- **ROC & AUC**: Logistic đạt `AUC = 0.9449`, vượt Rừng 200 cây (`0.9249`), Naive Bayes (`0.9336`), KNN (`0.9274`), SVM RBF (`0.8996`), Cây sâu 4 (`0.8803`).
- **MAE vs RMSE (Bài 6)**: cộng thêm 5 tỷ vào một dự đoán làm MAE tăng `+0.4167` (`0.2578 -> 0.6745`), trong khi RMSE tăng `+1.1775` (`0.3730 -> 1.5505`) và R² giảm `-0.6148` (`0.9622 -> 0.3475`).
