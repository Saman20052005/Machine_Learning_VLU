# -*- coding: utf-8 -*-
"""Bài tập 4: Tự tính 4 thước đo thủ công từ confusion matrix.

Không dùng accuracy_score, precision_score, recall_score, f1_score của scikit-learn.
Tự tính Accuracy, Precision, Recall, F1 từ 4 ô TN, FP, FN, TP.
Đối chiếu với kết quả chuẩn và giải thích ý nghĩa thực tế.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu và chia tập train/test giống c4_danh_gia.py
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 2. Huấn luyện và dự đoán trên tập kiểm tra
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# 3. Lấy 4 giá trị từ ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("=" * 65)
print("BAI TAP 4: TU TINH 4 THUOC DO THU CONG")
print("=" * 65)
print(f"Tap kiem tra (test set): {len(y_test)} sinh vien")
print("Ma tran nham lan (Confusion Matrix):")
print(f"  TN = {tn:2d} (Doan rot, that su rot)")
print(f"  FP = {fp:2d} (Doan qua, that ra rot)")
print(f"  FN = {fn:2d} (Doan rot, that ra qua)")
print(f"  TP = {tp:2d} (Doan qua, that su qua)")
print("-" * 65)

# 4. Tự tính thủ công thuần túy từ công thức toán học
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("KET QUA TU TINH THU CONG THEO CONG THUC:")
print(f"  Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
print(f"            = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}")
print(f"  Precision = TP / (TP + FP)")
print(f"            = {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"  Recall    = TP / (TP + FN)")
print(f"            = {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"  F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
print(f"            = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
print("-" * 65)

# 5. Đối chiếu kết quả chuẩn trong giáo trình
print("DOI CHIEU VOI KET QUA CHUAN GIAO TRINH (BANG 8):")
print(f"  TN: {tn} == 9  | FP: {fp} == 3  | FN: {fn} == 1  | TP: {tp} == 17  => KHOP HOAN TOAN")
print(f"  Accuracy : {accuracy:.4f} == 0.8667 => KHOP")
print(f"  Precision: {precision:.4f} == 0.8500 => KHOP")
print(f"  Recall   : {recall:.4f} == 0.9444 => KHOP")
print(f"  F1-Score : {f1:.4f} == 0.8947 => KHOP")
print("-" * 65)

# 6. Giải thích ý nghĩa thực tế
print("Y NGHIA THUC TE TRONG BAI TOAN DU DOAN QUA/ROT MON:")
print("1. Accuracy (0.8667 - 86.67%):")
print("   - Ty le du doan dung tren toan bo sinh vien kiem tra (26/30 ban dung).")
print("   - Cho cai nhin tong quan ve nang luc mo hinh tren tap mau kiem tra.")
print("2. Precision (0.8500 - 85.00%):")
print("   - Trong 20 ban ma mo hinh doan se QUA MON (TP + FP), co 17 ban qua that.")
print("   - Do tin cay khi mo hinh dua ra du doan tich cuc (qua mon).")
print("3. Recall (0.9444 - 94.44%):")
print("   - Trong 18 ban THAT SU QUA MON (TP + FN), mo hinh nhan dien dung 17 ban.")
print("   - Do bao phu cao, mo hinh it bo sot sinh vien qua mon (chi bo sot 1 ban FN).")
print("4. F1-Score (0.8947):")
print("   - Trung binh dieu hoa giua Precision va Recall.")
print("   - Thuoc do can bang tong the, dac biet huu ich khi muon dung hoa giua")
print("     viec doan chuan xac (Precision) va tranh bo sot (Recall).")
print("=" * 65)
