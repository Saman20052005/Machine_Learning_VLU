import pandas as pd


def main() -> None:
    du_lieu_nha = pd.read_csv("data/gia_nha.csv")
    can_ho_lon = du_lieu_nha[du_lieu_nha["dien_tich"] > 100]
    so_can_ho = len(can_ho_lon)
    gia_trung_binh = can_ho_lon["gia"].mean()

    print(f"So can co dien tich > 100 m2: {so_can_ho}")
    print(f"Gia trung binh: {gia_trung_binh:.3f} ty dong")


if __name__ == "__main__":
    main()
