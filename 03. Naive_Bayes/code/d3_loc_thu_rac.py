"""Bước 3: huấn luyện và đánh giá bộ lọc tin nhắn rác."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/tin_nhan.csv")
    train, test, y_train, y_test = train_test_split(
        df.noi_dung, df.la_rac, test_size=0.25, random_state=17, stratify=df.la_rac
    )
    # Chỉ học từ điển trên tập huấn luyện để tránh rò rỉ dữ liệu kiểm tra.
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train, X_test = vec.fit_transform(train), vec.transform(test)
    model = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"So tin de hoc: {len(train)}; so tin de kiem tra: {len(test)}")
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    print(f"TN = {tn}; FP = {fp}; FN = {fn}; TP = {tp}")
    print(f"Accuracy  = {accuracy_score(y_test, pred):.4f}")
    print(f"Precision = {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall    = {recall_score(y_test, pred):.4f}")
    tin_moi = ["trúng thưởng khuyến mãi nhanh tay click ngay",
               "họp nhóm báo cáo chiều mai phòng 305",
               "bạn gửi giúp em tài liệu bài tập nhé"]
    print("\nThu tren ba tin nhan moi:")
    for tin, prob in zip(tin_moi, model.predict_proba(vec.transform(tin_moi))[:, 1]):
        print(f"[{'RAC' if prob >= 0.5 else 'thuong'}] P(rac) = {prob:.4f}: {tin}")


if __name__ == "__main__":
    main()
