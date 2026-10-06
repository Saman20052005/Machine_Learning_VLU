# -*- coding: utf-8 -*-
"""Bài tập 6: Đổi lớp dương sang lớp rớt môn (pos_label=0).

Giữ nguyên mô hình và phân chia dữ liệu, nhưng đổi lớp dương sang lớp 0.
Tính Precision và Recall với pos_label=0, so sánh với lớp 1.
Giải thích nguyên nhân sự khác biệt.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu và chia tập train/test (giữ nguyên random_state=17, stratify=y)
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Ma trận nhầm lẫn gốc
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 2. Tính Precision và Recall cho lớp 1 (Qua môn)
prec_1 = precision_score(y_test, y_pred, pos_label=1)
rec_1 = recall_score(y_test, y_pred, pos_label=1)

# 3. Tính Precision và Recall cho lớp 0 (Rớt môn)
prec_0 = precision_score(y_test, y_pred, pos_label=0)
rec_0 = recall_score(y_test, y_pred, pos_label=0)

print("=" * 65)
print("BAI TAP 6: DOI LOP DUONG SANG LOP ROT MON (POS_LABEL=0)")
print("=" * 65)
print("Ma tran nham lan ban dau (Lop 1 la Qua mon, Lop 0 la Rot mon):")
print(f"  TN = {tn} (That 0, Doan 0) | FP = {fp} (That 0, Doan 1)")
print(f"  FN = {fn} (That 1, Doan 0) | TP = {tp} (That 1, Doan 1)")
print("-" * 65)
print("BANG SO SANH THUOC DO GIUA HAI LOP:")
print(f"{'Thuoc do':<15} | {'Lop 1 (Qua mon)':<18} | {'Lop 0 (Rot mon)':<18}")
print("-" * 65)
print(f"{'Precision':<15} | {prec_1:<18.4f} | {prec_0:<18.4f}")
print(f"{'Recall':<15} | {rec_1:<18.4f} | {rec_0:<18.4f}")
print("-" * 65)

print("GIAI THICH NGUYEN NHAN SU KHAC BIET:")
print("1. Su hoan doi vai tro trong ma tran nham lan:")
print("   Khi coi Lop 0 (Rot mon) la lop duong (positive):")
print(f"   - TP_lop0 = TN = {tn} (Doan rot dung)")
print(f"   - FP_lop0 = FN = {fn} (Doan rot nhung that ra qua mon)")
print(f"   - FN_lop0 = FP = {fp} (Doan qua nhung that ra rot mon)")
print(f"   - TN_lop0 = TP = {tp} (Doan qua dung)")
print()
print("2. Tinh toan cu the cho Lop 0:")
print(f"   - Precision_0 = TP_0 / (TP_0 + FP_0) = {tn} / ({tn} + {fn}) = {tn}/{tn+fn} = {prec_0:.4f}")
print(f"     Y nghia: Trong 10 ban mo hinh bao rot mon, co 9 ban rot that (chinh xac 90%).")
print(f"   - Recall_0    = TP_0 / (TP_0 + FN_0) = {tn} / ({tn} + {fp}) = {tn}/{tn+fp} = {rec_0:.4f}")
print(f"     Y nghia: Trong 12 ban that su rot mon, mo hinh chi bat duoc 9 ban (bo sot 3 ban rot sang qua).")
print()
print("3. Ket luan:")
print("   Du mo hinh va du lieu khong he thay doi, Precision va Recall cua hai")
print("   lop van khac nhau vi ti le mau that su cua hai lop khac nhau va cac loai")
print("   sai sot (FP, FN) hoan doi vi tri cho nhau khi chuyen tieu diem lop duong.")
print("   Do do, luon can xac dinh ro muc tieu bai toan va chi dinh chinh xac lop can danh gia (pos_label).")
print("=" * 65)
