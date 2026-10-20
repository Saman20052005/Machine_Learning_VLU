"""Bước 4: so sánh mô hình trên cùng dữ liệu có biên cong."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data/hai_vang_trang.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[["x1", "x2"]], df.nhan, test_size=0.25, random_state=17, stratify=df.nhan)
    models = [("LogisticRegression", LogisticRegression()), ("GaussianNB", GaussianNB()),
              ("KNN k=13 (theo de)", KNeighborsClassifier(13))]
    print("SO SANH TREN HAI VANG TRANG, CUNG TRAIN/TEST")
    a, b = np.meshgrid(np.linspace(df.x1.min()-.3, df.x1.max()+.3, 220),
                       np.linspace(df.x2.min()-.3, df.x2.max()+.3, 180))
    grid = pd.DataFrame({"x1": a.ravel(), "x2": b.ravel()})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, (ten, model) in zip(axes, models):
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"{ten:24s}: accuracy = {acc:.4f}")
        ax.contourf(a, b, model.predict(grid).reshape(a.shape), levels=[-.5, .5, 1.5],
                    colors=["#b8d9f2", "#ffd4b5"], alpha=.7)
        ax.scatter(X_tr.x1, X_tr.x2, c=y_tr, cmap="coolwarm", s=13, edgecolors="none")
        ax.scatter(X_te.x1, X_te.x2, c=y_te, cmap="coolwarm", s=30, marker="x")
        ax.set(title=f"{ten}\nTest accuracy={acc:.4f}", xlabel="x1")
    axes[0].set_ylabel("x2")
    fig.suptitle("Bien quyet dinh: cham = train; dau x = test")
    fig.tight_layout()
    (root / "bieu_do").mkdir(exist_ok=True)
    fig.savefig(root / "bieu_do/e4_bien_quyet_dinh.png", dpi=160)
    plt.close(fig)
    print("Logistic co bien tuyen tinh; GaussianNB phu thuoc gia dinh Gaussian tung lop.")
    print("KNN bam theo cau truc cuc bo nen co the theo duoc hai vang trang.")


if __name__ == "__main__":
    main()
