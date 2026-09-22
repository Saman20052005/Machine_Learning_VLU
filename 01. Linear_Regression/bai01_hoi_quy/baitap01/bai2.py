import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    du_lieu_nha = pd.read_csv("data/gia_nha.csv")

    plt.figure(figsize=(8, 5))
    plt.scatter(du_lieu_nha["so_phong"], du_lieu_nha["gia"], color="tab:blue")
    plt.xlabel("So phong")
    plt.ylabel("Gia (ty dong)")
    plt.title("Gia nha theo so phong")
    plt.xticks(sorted(du_lieu_nha["so_phong"].unique()))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("baitap01/bai2.png", dpi=150)
    plt.close()

    print("Nhin chung, so phong tang thi gia nha co xu huong tang, tuy nhien van co do phan tan giua cac mau.")


if __name__ == "__main__":
    main()
