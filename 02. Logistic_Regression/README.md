# Lab 02 - Hồi quy Logistic (Logistic Regression)

Bài thực hành môn **Học máy ứng dụng** - Khoa Công nghệ Thông tin, Trường Đại học Văn Lang.

---

## 1. Mục tiêu bài thực hành
- Nắm vững bài toán phân loại nhị phân (Binary Classification) với nhãn 0 (Rớt môn) và 1 (Qua môn).
- Hiểu nguyên lý và cài đặt hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- Huấn luyện mô hình hồi quy Logistic với thư viện `scikit-learn` (`LogisticRegression`).
- Đánh giá mô hình bằng Ma trận nhầm lẫn (Confusion Matrix) và 4 thước đo: Accuracy, Precision, Recall, $F_1$-Score.
- Phân tích sự đánh đổi giữa Precision và Recall khi điều chỉnh ngưỡng quyết định (Decision Threshold).
- Mở rộng mô hình từ 1 đặc trưng (`gio_on`) sang 2 đặc trưng (`gio_on`, `diem_giua_ky`).

---

## 2. Cấu trúc thư mục

```text
02. Logistic_Regression/
├── data/
│   └── sinh_vien.csv              # Bộ dữ liệu 120 sinh viên
├── code/
│   ├── c1_doc_du_lieu.py          # Bước 1: Đọc và khám phá dữ liệu
│   ├── c2_sigmoid.py              # Bước 2: Tự cài đặt hàm Sigmoid
│   ├── c3_khop_mo_hinh.py         # Bước 3: Khớp mô hình LogisticRegression 1 biến
│   ├── c4_danh_gia.py             # Bước 4: Đánh giá mô hình với Confusion Matrix & 4 thước đo
│   ├── c5_doi_nguong.py           # Bước 5: Thử nghiệm thay đổi ngưỡng quyết định
│   └── c6_hai_bien.py             # Bước 6: Mở rộng mô hình 2 biến (gio_on, diem_giua_ky)
├── baitap02/
│   ├── bai1.py                    # Bài 1: Thống kê tỷ lệ qua môn theo điểm giữa kỳ (>= 7.0 vs < 7.0)
│   ├── bai2.py                    # Bài 2: Vẽ đồ thị hàm Sigmoid và lưu file sigmoid.png
│   ├── bai3.py                    # Bài 3: Dự đoán xác suất cho các mốc giờ và giải thích mốc 12.89h
│   ├── bai4.py                    # Bài 4: Tự tính 4 thước đo thủ công không dùng hàm sklearn
│   ├── bai5.py                    # Bài 5: Dò tìm ngưỡng quyết định tối ưu theo F1-Score
│   ├── bai6.py                    # Bài 6: Đổi lớp dương sang lớp rớt môn (pos_label=0) và so sánh
│   └── sigmoid.png                # Đồ thị hàm Sigmoid xuất ra từ bai2.py
├── requirements.txt               # Danh sách thư viện cần thiết
└── README.md                      # Tài liệu hướng dẫn và báo cáo kết quả
```

---

## 3. Cài đặt môi trường & Chạy mã nguồn

### Kích hoạt môi trường và cài đặt thư viện:
```bash
cd "02. Logistic_Regression"
pip install -r requirements.txt
```

> **Lưu ý quan trọng về đường dẫn:** Mọi lệnh Python phải được thực thi từ thư mục gốc của bài lab (`02. Logistic_Regression`), không chạy từ bên trong thư mục `code/` hay `baitap02/` để đảm bảo đường dẫn tương đối `data/sinh_vien.csv` luôn chính xác.

### Chạy các tệp thực hành hướng dẫn (`code/`):
```bash
python code/c1_doc_du_lieu.py
python code/c2_sigmoid.py
python code/c3_khop_mo_hinh.py
python code/c4_danh_gia.py
python code/c5_doi_nguong.py
python code/c6_hai_bien.py
```

### Chạy các tệp bài tập bắt buộc (`baitap02/`):
```bash
python baitap02/bai1.py
python baitap02/bai2.py
python baitap02/bai3.py
python baitap02/bai4.py
python baitap02/bai5.py
python baitap02/bai6.py
```

---

## 4. Tóm tắt kết quả chính

### Bảng đối chiếu kết quả chuẩn (Phụ lục A - Bảng 8 trong giáo trình):

| Đại lượng | Giá trị chuẩn giáo trình | Kết quả chạy thực tế | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Kích thước dữ liệu** | 120 dòng (71 qua, 49 rớt) | 120 dòng (71 qua, 49 rớt) | Khớp 100% |
| **Giờ ôn trung bình (rớt / qua)** | 8.29 và 19.89 giờ | 8.29 và 19.89 giờ | Khớp 100% |
| **Hệ số mô hình 1 biến** | $w = 0.392819, b = -5.064890$ | $w = 0.392819, b = -5.064890$ | Khớp 100% |
| **Mốc giờ phân vân ($p = 0.5$)** | 12.89 giờ | 12.89 giờ | Khớp 100% |
| **Phân chia dữ liệu** | 90 học (train), 30 kiểm tra (test) | 90 train, 30 test (`test_size=0.25, random_state=17, stratify=y`) | Khớp 100% |
| **Ma trận nhầm lẫn** | $TN=9, FP=3, FN=1, TP=17$ | $TN=9, FP=3, FN=1, TP=17$ | Khớp 100% |
| **Accuracy** | 0.8667 | 0.8667 | Khớp 100% |
| **Precision (lớp 1)** | 0.8500 | 0.8500 | Khớp 100% |
| **Recall (lớp 1)** | 0.9444 | 0.9444 | Khớp 100% |
| **$F_1$-Score (lớp 1)** | 0.8947 | 0.8947 | Khớp 100% |
| **Accuracy mô hình 2 biến** | 0.9000 | 0.9000 | Khớp 100% |
| **Hệ số mô hình 2 biến** | $w_1 = +0.3035, w_2 = +0.5885, b = -6.9811$ | $w_1 = +0.3035, w_2 = +0.5885, b = -6.9811$ | Khớp 100% |

---

## 5. Tóm tắt kết quả phần Bài tập bắt buộc (`baitap02/`)

- **Bài 1 (`bai1.py`):**
  - Số sinh viên có `diem_giua_ky >= 7.0`: 31/120 sinh viên.
  - Tỷ lệ qua môn nhóm $\ge 7.0$: **90.32%** (`0.9032`).
  - Tỷ lệ qua môn nhóm $< 7.0$: **48.31%** (`0.4831`).
  - *Nhận xét:* Điểm giữa kỳ có khả năng phân biệt lớp rất mạnh; sinh viên đạt từ 7.0 điểm giữa kỳ trở lên có tỷ lệ qua môn vượt trội gấp gần 2 lần nhóm còn lại.

- **Bài 2 (`bai2.py`):**
  - Xuất thành công đồ thị hàm Sigmoid với $z \in [-8, 8]$, có đầy đủ đường gióng $y = 0.5$, trục đối xứng $z = 0$, điểm gốc $(0, 0.5)$ và chú thích công thức toán học. Đồ thị được lưu tại `baitap02/sigmoid.png`.

- **Bài 3 (`bai3.py`):**
  - Dự đoán cho các mốc giờ:
    - 3.0h: $z = -3.8864 \implies p = 0.0201$ (Rớt môn)
    - 8.0h: $z = -1.9223 \implies p = 0.1276$ (Rớt môn)
    - 12.89h: $z = -0.0015 \implies p = 0.4996 \approx 0.5$ (Ranh giới 50/50)
    - 18.0h: $z = +2.0059 \implies p = 0.8814$ (Qua môn)
    - 26.0h: $z = +5.1484 \implies p = 0.9942$ (Qua môn)
  - *Giải thích toán học:* Tại $x = 12.89$, ta có $z = wx + b \approx 0$. Do $\sigma(0) = \frac{1}{1 + e^0} = 0.5$, đây chính là nghiệm của phương trình đường biên quyết định (decision boundary).

- **Bài 4 (`bai4.py`):**
  - Tính toán thủ công thuần túy từ công thức:
    - $\text{Accuracy} = \frac{17 + 9}{30} = 0.8667$
    - $\text{Precision} = \frac{17}{17 + 3} = 0.8500$
    - $\text{Recall} = \frac{17}{17 + 1} = 0.9444$
    - $F_1 = 2 \cdot \frac{0.8500 \cdot 0.9444}{0.8500 + 0.9444} = 0.8947$
  - Trình bày chi tiết ý nghĩa thực tế trong bài toán giáo dục.

- **Bài 5 (`bai5.py`):**
  - Dò ngưỡng từ $0.05$ đến $0.95$ với bước $0.05$.
  - Ngưỡng cho $F_1$-score cao nhất là **$t = 0.60$** và **$t = 0.65$** với $F_1 = 0.9412$ (cao hơn mức $0.8947$ tại ngưỡng mặc định $0.5$).
  - *Giải thích:* Ngưỡng $0.5$ chỉ là đối xứng toán học mặc định. Trên tập dữ liệu cụ thể, nâng ngưỡng lên $0.60$ giúp loại bỏ hoàn toàn các lỗi False Positive (Precision đạt 1.0000 trong khi Recall vẫn duy trì ở mức cao 0.8889), từ đó tối ưu hóa trung bình điều hòa $F_1$.

- **Bài 6 (`bai6.py`):**
  - Chuyển lớp dương sang lớp 0 (Rớt môn, `pos_label=0`):
    - **Lớp 1 (Qua môn):** $\text{Precision} = 0.8500$, $\text{Recall} = 0.9444$.
    - **Lớp 0 (Rớt môn):** $\text{Precision} = 0.9000$, $\text{Recall} = 0.7500$.
  - *Giải thích:* Khi đổi lớp dương, các vai trò $TP \leftrightarrow TN$ và $FP \leftrightarrow FN$ bị hoán đổi ($TP_0 = 9, FP_0 = 1, FN_0 = 3, TN_0 = 17$). Tỷ lệ mẫu thực tế của 2 lớp khác nhau nên mẫu số tính toán thay đổi, dẫn đến hai bộ chỉ số hoàn toàn khác biệt dù dữ liệu và mô hình giữ nguyên.
