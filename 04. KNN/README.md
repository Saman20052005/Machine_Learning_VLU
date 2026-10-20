# Lab04 — K láng giềng gần nhất (KNN)

Bài làm theo `Lab04_KNN.pdf`, gồm 5 chương trình thực hành và 6 bài tập mục 7 (trang in 14–15). Ba tệp dữ liệu được sao chép nguyên bản từ đường dẫn đã cung cấp, không sửa dữ liệu gốc.

## Chạy bài

Mở PowerShell tại thư mục repository:

```powershell
cd "D:\Machine_Learning_VLU\04. KNN"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
```

Chạy từng chương trình:

```powershell
python -X utf8 code/e1_khoang_cach.py
python -X utf8 code/e2_chuan_hoa.py
python -X utf8 code/e3_chon_k.py
python -X utf8 code/e4_so_sanh.py
python -X utf8 code/e5_knn_hoi_quy.py
python -X utf8 baitap04/bai1.py
python -X utf8 baitap04/bai2.py
python -X utf8 baitap04/bai3.py
python -X utf8 baitap04/bai4.py
python -X utf8 baitap04/bai5.py
python -X utf8 baitap04/bai6.py
```

Đường dẫn dữ liệu được tính từ vị trí tệp mã nên cũng có thể chạy từ thư mục khác. Chú thích dùng tiếng Việt có dấu, phần in kết quả không dấu theo mẫu trong đề. Các biểu đồ được lưu vào `bieu_do/`, không mở cửa sổ tương tác.

## Dữ liệu và cách đánh giá

| Tệp CSV | Kích thước | Dùng trong bài |
| --- | --- | --- |
| `sinh_vien_mo_rong.csv` | 120 dòng, 4 cột | Chuẩn hóa, đổi đơn vị chi tiêu |
| `hai_vang_trang.csv` | 300 dòng, 3 cột | Phân loại, chọn k, so sánh metric và trọng số |
| `gia_nha.csv` | 60 dòng, 4 cột | Hồi quy chỉ với cột `dien_tich` như đề |

Các bài phân loại dùng `test_size=0.25`, `random_state=17`, `stratify=y`. Hồi quy giá nhà dùng `test_size=0.2`, `random_state=42`, không phân tầng. Không có giá trị thiếu trong ba CSV.

Khi có chuẩn hóa, `StandardScaler` nằm trong pipeline và chỉ fit trên train. Chọn k phân loại bằng CV 5 phần trên train, thử các k lẻ 1–51; chọn k hồi quy bằng CV 5 phần trên train, thử 1–20. Giữ cấu hình CV mặc định của đề, không xáo trộn các fold. Khi điểm bằng nhau, ưu tiên k nhỏ nhất theo thứ tự duyệt. Các giá trị cùng làm tròn bốn chữ số chưa chắc bằng nhau ở độ chính xác đầy đủ.

Hai cột của bộ hai vầng trăng được giữ nguyên thang đo như đề. Chương trình bước 3 khóa k bằng CV trước khi chấm test; bảng train/test theo nhiều k chỉ dùng minh họa học vẹt và làm trơn quá mức, không dùng chọn k.

## Mã nguồn và kết quả

| Nội dung | Mã nguồn | Kết quả |
| --- | --- | --- |
| Khoảng cách và 5 láng giềng | [e1_khoang_cach.py](code/e1_khoang_cach.py) | [Văn bản](ket_qua/e1_khoang_cach.txt) |
| Tác động của chuẩn hóa | [e2_chuan_hoa.py](code/e2_chuan_hoa.py) | [Văn bản](ket_qua/e2_chuan_hoa.txt) |
| Chọn k bằng CV | [e3_chon_k.py](code/e3_chon_k.py) | [Văn bản](ket_qua/e3_chon_k.txt) · [Biểu đồ](bieu_do/e3_chon_k.png) |
| So sánh ba mô hình | [e4_so_sanh.py](code/e4_so_sanh.py) | [Văn bản](ket_qua/e4_so_sanh.txt) · [Biên quyết định](bieu_do/e4_bien_quyet_dinh.png) |
| Hồi quy giá nhà | [e5_knn_hoi_quy.py](code/e5_knn_hoi_quy.py) | [Văn bản](ket_qua/e5_knn_hoi_quy.txt) · [Biểu đồ](bieu_do/e5_hoi_quy_gia_nha.png) |
| Bài 1 — KNN tự viết | [bai1.py](baitap04/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — Manhattan | [bai2.py](baitap04/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — Đổi đơn vị | [bai3.py](baitap04/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Trọng số khoảng cách | [bai4.py](baitap04/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) |
| Bài 5 — Chọn k hồi quy | [bai5.py](baitap04/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — Thời gian thực tế | [bai6.py](baitap04/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [CSV](ket_qua/bai6_thoi_gian.csv) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

[Toàn bộ kết quả chạy](ket_qua/tat_ca_ket_qua.txt). Các ảnh `*.png` không có hậu tố trong `ket_qua/` được dựng từ stdout bằng Pillow. Các ảnh `*_screenshot.png` là ảnh Edge headless chụp trang HTML hiển thị nguyên văn stdout, không phải ảnh chụp cửa sổ PowerShell. Tạo lại ảnh chụp sau khi chạy mã:

```powershell
python -X utf8 chup_ket_qua.py
```

Lệnh chụp ảnh cần Microsoft Edge trên Windows; các bài tập chỉ cần thư viện trong `requirements.txt`. Nếu nơi nộp yêu cầu ảnh cửa sổ terminal, chạy từng bài và chụp PowerShell. Các tệp `ket_qua/baiN.html` cũng có thể mở trực tiếp để đọc kết quả.

## Phần thực hành

Khoảng cách ban đầu `d(A,B)=102.2`, `d(A,C)=3000.0`; sau chuẩn hóa lần lượt là `3.85` và `1.79`. Cột chi tiêu có độ lệch chuẩn khoảng 1691.67 nghìn đồng, lớn hơn nhiều hai cột học tập nên lấn át khoảng cách thô. Năm láng giềng của bạn ôn 12 giờ, điểm 6.5 có 4 bạn qua, 1 bạn rớt; dự đoán qua môn.

| Cấu hình sinh viên, k=7 | Accuracy test |
| --- | ---: |
| Luôn đoán qua môn | 0.6000 |
| Ba cột không chuẩn hóa | 0.5333 |
| Ba cột có chuẩn hóa | 0.8667 |
| Bỏ chi tiêu, chuẩn hóa hai cột học tập | 0.9000 |

Chuẩn hóa xử lý thang đo nhưng không biến cột không liên quan thành cột hữu ích. Vì vậy bỏ chi tiêu vẫn cải thiện kết quả ở lần chia này.

Trên hai vầng trăng, CV chọn `k=13`, accuracy CV `0.8667`, test `0.9200`. LogisticRegression và GaussianNB đều đạt `0.8533`. KNN theo được cấu trúc cục bộ có biên cong; logistic chỉ có biên tuyến tính trên hai đặc trưng gốc. GaussianNB chịu ràng buộc từ giả định phân phối của từng lớp.

Với giá nhà, KNN k=5 có MAE `0.4030` tỷ đồng; LinearRegression có MAE `0.2578` tỷ đồng. KNN dự đoán cả căn 150 m² lẫn 250 m² khoảng `9.32` tỷ vì cùng lấy trung bình của năm căn lớn nhất trong train: 117.5, 115.9, 111.7, 111.2, 104.2 m². Đây là giới hạn ngoại suy của KNN với trọng số đều. Đường thẳng tiếp tục kéo dài xu hướng, nhưng dự đoán xa vùng dữ liệu vẫn cần thận trọng.

## Bài 1 — Tự viết KNN

Hàm `knn_du_doan(X_hoc, y_hoc, x_moi, k)` thực hiện ba bước: tính khoảng cách Euclid tới mọi mẫu train, lấy chỉ số k khoảng cách nhỏ nhất bằng `np.argsort`, rồi đếm nhãn và chọn lớp nhiều phiếu nhất. Phần thuật toán không gọi bộ phân loại sklearn.

Với k=13, bộ tự viết và `KNeighborsClassifier` cùng đạt **0.9200**, không khác nhãn dự đoán nào trên 75 mẫu test. Mã có assert kiểm tra từng nhãn, thay vì chỉ so accuracy vì hai bộ dự đoán khác nhau vẫn có thể có cùng accuracy.

Khi hòa phiếu, hàm chọn nhãn nhỏ nhất; k lẻ trên hai lớp tránh hòa phiếu với trọng số đều. Hòa khoảng cách tại ranh giới láng giềng thứ k là vấn đề khác: hàm ưu tiên thứ tự dòng train, còn backend sklearn có thể chọn khác. Trên dữ liệu đã chạy, điều này không gây khác biệt dự đoán.

## Bài 2 — Manhattan

| Metric | k chọn bằng CV | Accuracy CV | Accuracy test |
| --- | ---: | ---: | ---: |
| Euclid | 13 | 0.8667 | 0.9200 |
| Manhattan | 27 | 0.8667 | 0.9067 |

Manhattan dùng tổng trị tuyệt đối độ lệch, Euclid dùng căn tổng bình phương. Đổi metric thay đổi thứ tự láng giềng và k tốt nhất; lần này test giảm 1.33 điểm phần trăm, tương đương chỉ một dự đoán đúng trong 75 mẫu. Mức chênh nhỏ trên một lần chia chưa đủ kết luận metric nào luôn tốt hơn.

## Bài 3 — Đổi đơn vị

Đổi chi tiêu từ nghìn đồng sang triệu đồng bằng chia cho 1000, giữ nguyên dòng train/test và k=7. Chương trình in cả bốn cấu hình để đối chiếu.

| Đơn vị chi tiêu | Không chuẩn hóa | Có chuẩn hóa |
| --- | ---: | ---: |
| Nghìn đồng | 0.5333 | 0.8667 |
| Triệu đồng | 0.8667 | 0.8667 |

Đổi đơn vị làm accuracy thô tăng 33.33 điểm phần trăm. Sau chuẩn hóa, kết quả giữ nguyên 0.8667. Cùng accuracy giữa cấu hình thô triệu đồng và cấu hình chuẩn hóa ở lần chia này không có nghĩa hai phép đo khoảng cách tương đương.

Khoảng cách thô có `d² = delta_gio² + delta_diem² + delta_chi_tieu²`. Chia chi tiêu cho 1000 làm phần bình phương của nó giảm một triệu lần, nên thứ tự hàng xóm có thể thay đổi dù thông tin thực tế không đổi.

Khi chuẩn hóa và hệ số đổi đơn vị a dương, `(a*x − a*mu)/(a*sigma) = (x − mu)/sigma`. Vì vậy các tọa độ chuẩn hóa bằng nhau trong sai số số thực và dự đoán giữ nguyên. Mã kiểm tra cả dữ liệu test sau chuẩn hóa bằng `assert_allclose` và nhãn bằng `assert_array_equal`. Scaler luôn fit trên train để tránh rò rỉ test.

## Bài 4 — Láng giềng có trọng số

So sánh `weights="uniform"` và `weights="distance"` với k=1, 13, 101, giữ nguyên tập chia và khoảng cách Euclid. Trọng số nghịch khoảng cách `w=1/d` tăng ảnh hưởng của láng giềng gần, giảm ảnh hưởng của láng giềng xa. Khi k lớn, cách này có thể giữ lại cấu trúc cục bộ thay vì để nhiều điểm xa lấn át phiếu của điểm gần.

| k | Test accuracy — uniform | Test accuracy — distance |
| ---: | ---: | ---: |
| 1 | 0.8000 | 0.8000 |
| 13 | 0.9200 | 0.9333 |
| 101 | 0.8533 | 0.9067 |

Với k=101, trọng số khoảng cách giúp tăng 5.33 điểm phần trăm, từ 64 lên 68 dự đoán đúng trong 75 mẫu. Ở lần chia này, nó giúp mô hình bớt làm trơn quá mức; không bảo đảm luôn cải thiện trên dữ liệu khác.

Với k=1, cả hai cách có cùng nhãn vì chỉ có một hàng xóm bỏ phiếu. Khoảng cách bằng 0 được sklearn xử lý riêng bằng các điểm trùng tọa độ, tránh chia cho 0. Điểm train thường rất cao với trọng số khoảng cách vì mỗi mẫu trùng chính nó; cần đánh giá trên test.

## Bài 5 — Chọn k hồi quy

Dùng một đặc trưng diện tích như phần thực hành. Với mỗi k=1–20, lấy trung bình `cross_val_score(..., cv=5, scoring="neg_mean_absolute_error")` trên train. Điểm trả về mang dấu âm nên chọn **score lớn nhất**, tương đương MAE dương nhỏ nhất. Bảng trong kết quả in MAE dương cho dễ đọc.

Sau khi khóa k mới tính test MAE và so với đường thẳng trên cùng dữ liệu. Chọn k tốt hơn không bảo đảm KNN thắng: quan hệ giá–diện tích trong bộ này gần tuyến tính nên LinearRegression có lợi thế nhờ mô hình hóa xu hướng chung.

CV chọn **k=2**, MAE CV **0.349800** tỷ đồng. MAE test KNN là **0.486667**, cao hơn MAE test đường thẳng **0.257784** nên KNN không vượt được đường thẳng. k=2 tốt nhất theo CV vẫn có thể kém k=5 trên test do biến động giữa các tập con; không đổi k sau khi nhìn test.

## Bài 6 — Đo thời gian

Dữ liệu tạo bằng `np.random.default_rng(0)`, gồm 1,000, 10,000 và 100,000 mẫu, hai đặc trưng và nhãn ngẫu nhiên nhị phân. Dùng cùng 1,000 mẫu mới để đo predict. Việc sinh dữ liệu và khởi động đường mã nằm ngoài thời gian đo; dùng `time.perf_counter`, lấy trung vị 5 lần. Bảng số đo thực tế lưu ở [bai6.txt](ket_qua/bai6.txt) và [bai6_thoi_gian.csv](ket_qua/bai6_thoi_gian.csv).

| Số mẫu train | Fit (ms) | Predict 1,000 mẫu (ms) | Backend |
| ---: | ---: | ---: | --- |
| 1,000 | 1.790 | 4.138 | KD-tree |
| 10,000 | 9.144 | 4.861 | KD-tree |
| 100,000 | 125.848 | 7.908 | KD-tree |

Đây là số đo của lần chạy đã lưu, với Python 3.11.9, scikit-learn 1.9.1, `n_jobs=1`. Chạy lại có thể thay đổi số đo trong TXT/CSV; bảng trên ghi lại lần chạy ban đầu.

KNN là học lười vì giữ dữ liệu để tra cứu, không tối ưu một bộ trọng số mô tả ranh giới trong lúc fit; tìm hàng xóm và bỏ phiếu diễn ra ở predict. Tuy nhiên fit không hoàn toàn miễn phí: với dữ liệu hai chiều này, `algorithm="auto"` chọn KD-tree và cần xây chỉ mục. KD-tree có thể giúp predict tăng chậm theo kích thước train, khác với duyệt toàn bộ bằng brute-force. Không bắt buộc thời gian predict tăng tuyến tính hoặc luôn lớn hơn fit.

Độ phức tạp tính khoảng cách brute-force xấp xỉ `O(n_train × n_query × d)`. Không suy ra độ phức tạp chỉ từ ba điểm đo; thời gian còn phụ thuộc backend, hình dạng dữ liệu, CPU và tải máy. Chạy lại sẽ cho số đo khác.
