import numpy as np
import pandas as pd


def main() -> None:
    du_lieu_nha = pd.read_csv("data/gia_nha.csv")
    tuoi_nha = du_lieu_nha["tuoi_nha"].to_numpy(dtype=float)
    gia_nha = du_lieu_nha["gia"].to_numpy(dtype=float)

    trung_binh_tuoi = np.mean(tuoi_nha)
    trung_binh_gia = np.mean(gia_nha)
    w = np.sum((tuoi_nha - trung_binh_tuoi) * (gia_nha - trung_binh_gia)) / np.sum(
        (tuoi_nha - trung_binh_tuoi) ** 2
    )
    b = trung_binh_gia - w * trung_binh_tuoi

    print(f"w = {w:.6f}")
    print(f"b = {b:.6f}")
    print("w am cho thay tuoi nha tang thi gia du doan giam theo mo hinh nay.")
    print("Tuy nhien, chi dau am cua w chua du chung minh tuoi nha va gia co quan he tuyen tinh chat che.")


if __name__ == "__main__":
    main()
