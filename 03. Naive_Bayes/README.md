# Lab03 — Naive Bayes

Bài làm theo `Lab03_Naive_Bayes.pdf`, gồm 5 chương trình thực hành và 6 bài tập mục 7 (trang in 19–20). Dữ liệu được sao chép nguyên bản từ hai tệp CSV đã cung cấp: 300 tin nhắn (180 thường, 120 rác) và 120 sinh viên.

## Chạy bài

Mở PowerShell tại thư mục repository:

```powershell
cd "D:\Machine_Learning_VLU\03. Naive_Bayes"
python -m pip install -r requirements.txt
python -X utf8 chay_tat_ca.py
```

Chạy riêng từng bài:

```powershell
python -X utf8 code/d1_bayes_tay.py
python -X utf8 code/d2_tui_tu.py
python -X utf8 code/d3_loc_thu_rac.py
python -X utf8 code/d4_lam_tron.py
python -X utf8 code/d5_gaussian.py
python -X utf8 baitap03/bai1.py
python -X utf8 baitap03/bai2.py
python -X utf8 baitap03/bai3.py
python -X utf8 baitap03/bai4.py
python -X utf8 baitap03/bai5.py
python -X utf8 baitap03/bai6.py
```

Đường dẫn dữ liệu tính từ vị trí tệp mã nên cũng có thể chạy từ thư mục khác. Chú thích và tài liệu dùng tiếng Việt có dấu; thông báo chương trình dùng tiếng Việt không dấu theo mẫu trong đề, riêng từ trích từ CSV giữ nguyên dấu.

## Tệp bài làm và kết quả

| Nội dung | Mã nguồn | Kết quả chạy |
| --- | --- | --- |
| Bayes hai nguyên nhân | [d1_bayes_tay.py](code/d1_bayes_tay.py) | [Văn bản](ket_qua/d1_bayes_tay.txt) |
| Túi từ và thống kê | [d2_tui_tu.py](code/d2_tui_tu.py) | [Văn bản](ket_qua/d2_tui_tu.txt) |
| Bộ lọc tin rác | [d3_loc_thu_rac.py](code/d3_loc_thu_rac.py) | [Văn bản](ket_qua/d3_loc_thu_rac.txt) |
| Làm trơn Laplace | [d4_lam_tron.py](code/d4_lam_tron.py) | [Văn bản](ket_qua/d4_lam_tron.txt) |
| GaussianNB và logistic | [d5_gaussian.py](code/d5_gaussian.py) | [Văn bản](ket_qua/d5_gaussian.txt) |
| Bài 1 — Ba nguyên nhân | [bai1.py](baitap03/bai1.py) | [Văn bản](ket_qua/bai1.txt) · [Ảnh chụp](ket_qua/bai1_screenshot.png) |
| Bài 2 — Từ phân biệt mạnh | [bai2.py](baitap03/bai2.py) | [Văn bản](ket_qua/bai2.txt) · [Ảnh chụp](ket_qua/bai2_screenshot.png) |
| Bài 3 — Thay alpha | [bai3.py](baitap03/bai3.py) | [Văn bản](ket_qua/bai3.txt) · [Ảnh chụp](ket_qua/bai3_screenshot.png) |
| Bài 4 — Bộ lọc tự viết | [bai4.py](baitap03/bai4.py) | [Văn bản](ket_qua/bai4.txt) · [Ảnh chụp](ket_qua/bai4_screenshot.png) |
| Bài 5 — Sai biến thể | [bai5.py](baitap03/bai5.py) | [Văn bản](ket_qua/bai5.txt) · [Ảnh chụp](ket_qua/bai5_screenshot.png) |
| Bài 6 — Lớp hiếm | [bai6.py](baitap03/bai6.py) | [Văn bản](ket_qua/bai6.txt) · [Ảnh chụp](ket_qua/bai6_screenshot.png) |

[Toàn bộ kết quả chạy](ket_qua/tat_ca_ket_qua.txt). Các tệp `*.png` không có hậu tố là ảnh dựng từ stdout bằng Pillow. Các tệp `*_screenshot.png` là ảnh chụp bằng Edge headless của trang HTML hiển thị nguyên văn stdout, không phải ảnh chụp cửa sổ terminal. Có thể mở `ket_qua/baiN.html` để xem và chạy lại lệnh sau để chụp mới sau khi cập nhật kết quả:

```powershell
python -X utf8 chup_ket_qua.py
```

Nếu nơi nộp yêu cầu ảnh terminal, chạy từng bài bằng các lệnh ở trên rồi chụp cửa sổ PowerShell.

## Kết quả và nhận xét

### Bài 1

Theo công thức xác suất toàn phần:

`P(qua) = 0.50 × 0.95 + 0.30 × 0.70 + 0.20 × 0.30 = 0.745`.

| Nhóm | Tử số | Hậu nghiệm khi đã qua môn |
| --- | --- | --- |
| Học đều | 0.475 | 0.637584 (63.76%) |
| Học dồn | 0.210 | 0.281879 (28.19%) |
| Ít học | 0.060 | 0.080537 (8.05%) |

Tổng ba xác suất bằng 1; chương trình có kiểm tra bằng `isclose`. Nhóm học đều có tỷ trọng trong số người qua môn lớn hơn tỷ trọng ban đầu vì xác suất qua môn của nhóm này cao nhất.

### Bài 2

Huấn luyện trên toàn bộ 300 tin theo yêu cầu, với `alpha=1`. Điểm của từ là `log P(từ|rác) − log P(từ|thường)`.

| Nghiêng về rác | Điểm | Nghiêng về thường | Điểm |
| --- | ---: | --- | ---: |
| mãi | +3.203435 | lớp | −2.612187 |
| tay | +2.905600 | em | −2.561756 |
| ngay | +2.661837 | bài | −2.294553 |
| miễn | +2.479516 | báo | −2.274074 |
| trúng | +2.374155 | phòng | −2.274074 |

Điểm dương lớn cho thấy tỷ số xác suất nghiêng mạnh về rác; điểm âm nhỏ nghiêng về thường. Đây là so sánh xác suất có điều kiện đã làm trơn, không phải chỉ xếp theo số lần xuất hiện.

### Bài 3

| Alpha | Accuracy | Precision rác | Recall rác |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.9467 | 1.0000 | 0.8667 |
| 0.1 | 0.9467 | 1.0000 | 0.8667 |
| 1.0 | 0.9467 | 1.0000 | 0.8667 |
| 10.0 | 0.9467 | 1.0000 | 0.8667 |
| 100.0 | 0.9467 | 1.0000 | 0.8667 |

Trong phạm vi đã thử, dự đoán trên 75 tin kiểm tra không đổi dù xác suất bên trong thay đổi. Không thể kết luận alpha lớn luôn vô hại: khi alpha quá lớn, `(đếm + alpha)/(tổng + alpha × V)` tiến về `1/V` ở cả hai lớp, làm mất khả năng phân biệt của từ. Trong giới hạn đó, tiên nghiệm sẽ quyết định và mô hình chọn lớp phổ biến. Muốn chọn alpha cần tập validation hoặc kiểm định chéo trên tập train; không chọn bằng tập test này.

### Bài 4

Bộ lọc tự viết dùng `Counter`, `re` và `math.log` để tách từ, đếm, làm trơn và dự đoán. Phần học và dự đoán không gọi bộ vector hóa hay bộ phân loại sklearn; sklearn chỉ dùng để chia dữ liệu, tính accuracy và làm đối chứng.

`score(k) = log P(k) + Σ count(từ) × log[(đếm(từ,k) + alpha)/(tổng_từ(k) + alpha × V)]`.

Hai bộ lọc cùng đạt **0.9467**, không khác dự đoán nào trên 75 tin kiểm tra. Sai số điểm log lớn nhất khoảng `1.421e-14`, phù hợp sai số số thực. Chương trình kiểm tra cả điểm log và nhãn bằng các phép assert.

Laplace tránh xác suất 0 cho từ có trong từ điển nhưng chưa gặp ở một lớp. Tổng loga tránh tràn dưới do nhân nhiều xác suất nhỏ. Từ hoàn toàn mới ngoài từ điển bị bỏ qua; nếu không có từ nào đã biết thì chỉ còn tiên nghiệm. Từ lặp lại được tính đủ số lần. Khi hai lớp hòa điểm, chọn lớp nhỏ hơn theo thứ tự của sklearn.

Nếu hai kết quả lệch nhiều, cần kiểm tra cách viết thường, cách tách từ một ký tự, số lần lặp, tiên nghiệm và mẫu số `tổng + alpha × V`. Cũng cần bảo đảm hai bộ lọc dùng đúng cùng tập train/test và cùng từ điển chỉ học từ tập train.

### Bài 5

Trên cùng 90 sinh viên huấn luyện và 30 sinh viên kiểm tra, **MultinomialNB đạt 0.8000**, **GaussianNB đạt 0.9000**. Chương trình không báo lỗi vì số giờ và điểm đều không âm; sklearn vẫn chấp nhận các giá trị thực không âm.

Tuy nhiên, số giờ và điểm là hai đại lượng liên tục khác đơn vị, không phải số đếm các loại sự kiện trong phân phối multinomial. Gộp chúng thành số đếm làm sai ý nghĩa mô hình, và việc đổi đơn vị có thể ảnh hưởng kết quả. GaussianNB mô tả mật độ liên tục theo trung bình, phương sai của từng đặc trưng trong từng lớp nên phù hợp hơn về loại dữ liệu. Accuracy đơn lẻ không chứng minh giả định phân phối là đúng.

### Bài 6

Giữ toàn bộ 180 tin thường và lấy đúng 20 tin rác đầu tiên theo thứ tự CSV. Sau chia có 135 thường + 15 rác trong train, 45 thường + 5 rác trong test; tiên nghiệm học được là `0.9/0.1`.

| Cách dự đoán | Accuracy | Precision rác | Recall rác |
| --- | ---: | ---: | ---: |
| MultinomialNB | 0.9800 | 1.0000 | 0.8000 |
| Luôn đoán thường | 0.9000 | Không có dự đoán rác | 0.0000 |

Ma trận của MultinomialNB: `TN=45, FP=0, FN=1, TP=4`. Accuracy 98% vẫn đi cùng việc bỏ lọt 20% số tin rác trong tập test. Một bộ lọc không chặn được tin rác nào cũng đạt 90% accuracy, nên accuracy không đủ phản ánh chất lượng.

Tiên nghiệm khiến lớp thường có lợi thế 9:1; bằng chứng từ ngữ phải đủ mạnh để bù lợi thế đó. Việc chỉ có 15 tin rác để học còn làm ước lượng xác suất từ kém ổn định. Cần xem recall rác để đánh giá bỏ lọt, và xem precision/FP để đánh giá chặn nhầm. Tập test chỉ có 5 tin rác nên mỗi tin làm recall đổi 20 điểm phần trăm; chưa thể suy rộng từ một lần chia nhỏ.

## Ghi chú thực hiện

- Các bài đánh giá dùng `test_size=0.25`, `random_state=17`, `stratify=y` và `token_pattern=r"(?u)\S+"`.
- Tách train/test trước khi học từ điển để tránh rò rỉ từ tập test; bài 2 dùng toàn bộ dữ liệu đúng yêu cầu thống kê của đề. Trên các tệp này, cách làm trên vẫn cho các chỉ số kiểm tra chính giống tài liệu.
- Ba tin nhập thử ở bước 3 dùng tiếng Việt có dấu cho nhất quán với CSV. Mẫu trong PDF dùng không dấu nên xác suất cho tin mới có thể khác. Con số in `0.0000` hoặc `1.0000` đã làm tròn bốn chữ số, không khẳng định xác suất chính xác bằng 0 hoặc 1.
- Môi trường đã chạy: Python 3.11.9, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, Pillow 12.3.0.
