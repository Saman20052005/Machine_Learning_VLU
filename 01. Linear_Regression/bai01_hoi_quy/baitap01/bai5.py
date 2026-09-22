import argparse

import numpy as np
import pandas as pd


SO_VONG = 200


def gradient_descent(dien_tich: np.ndarray, gia: np.ndarray, learning_rate: float) -> float:
    dien_tich_chuan_hoa = (dien_tich - np.mean(dien_tich)) / np.std(dien_tich)
    w = 0.0
    b = 0.0
    so_mau = len(dien_tich_chuan_hoa)

    for _ in range(SO_VONG):
        gia_du_doan = w * dien_tich_chuan_hoa + b
        sai_so = gia_du_doan - gia
        dao_ham_w = (2 / so_mau) * np.sum(sai_so * dien_tich_chuan_hoa)
        dao_ham_b = (2 / so_mau) * np.sum(sai_so)
        w -= learning_rate * dao_ham_w
        b -= learning_rate * dao_ham_b

    gia_du_doan = w * dien_tich_chuan_hoa + b
    return float(np.mean((gia_du_doan - gia) ** 2))


def chay_thi_nghiem(learning_rate: float, dien_tich: np.ndarray, gia: np.ndarray) -> None:
    mse = gradient_descent(dien_tich, gia, learning_rate)
    print(f"Learning rate {learning_rate}: MSE tai vong {SO_VONG} = {mse:.4f}")
    if learning_rate == 0.001:
        print("Learning rate 0.001 qua nho nen sau 200 vong chua den diem toi uu.")
    elif learning_rate == 1.02:
        print("Learning rate 1.02 qua lon nen thuat toan vuot qua diem toi uu va phan ky.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient descent cho hoi quy tuyen tinh.")
    parser.add_argument("--learning-rate", type=float, choices=[0.001, 1.02])
    args = parser.parse_args()

    du_lieu_nha = pd.read_csv("data/gia_nha.csv")
    dien_tich = du_lieu_nha["dien_tich"].to_numpy(dtype=float)
    gia = du_lieu_nha["gia"].to_numpy(dtype=float)
    learning_rates = [args.learning_rate] if args.learning_rate is not None else [0.001, 1.02]

    for learning_rate in learning_rates:
        chay_thi_nghiem(learning_rate, dien_tich, gia)


if __name__ == "__main__":
    main()
