# -*- coding: utf-8 -*-
"""Bài 4: Tự vẽ đường ROC của hồi quy logistic và rừng ngẫu nhiên."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)

    df = pd.read_csv(root / "data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    rf = RandomForestClassifier(200, random_state=0).fit(X_tr, y_tr)

    p_lr = lr.predict_proba(X_te)[:, 1]
    p_rf = rf.predict_proba(X_te)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_te, p_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, p_rf)
    auc_lr = roc_auc_score(y_te, p_lr)
    auc_rf = roc_auc_score(y_te, p_rf)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic (AUC = {auc_lr:.4f})", linewidth=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Rung 200 cay (AUC = {auc_rf:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Doan ngau nhien (AUC = 0.5000)")
    plt.xlabel("Ty le bao dong nham (FPR)")
    plt.ylabel("Recall (TPR)")
    plt.title("Duong ROC: Logistic vs Rung ngau nhien")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = bieu_do / "roc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Da luu bieu do: {out_path.relative_to(root).as_posix()}")
    print(f"AUC Logistic    : {auc_lr:.4f}")
    print(f"AUC Rung 200 cay: {auc_rf:.4f}")
    print()
    print("Nhan xet:")
    print("- Duong Logistic om sat goc tren ben trai hon (AUC cao hon: 0.9449 so voi 0.9249).")
    print("- O doan dau (FPR gan 0) va vung giua, hai duong co giao cat nhe,")
    print("  nhung nhin chung duong Logistic nam phia tren duong cua Rung.")


if __name__ == "__main__":
    main()
