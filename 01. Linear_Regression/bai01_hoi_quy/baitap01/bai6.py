W = 0.078367
B = 0.401752
DIEN_TICH_NHO_NHAT = 35.5
DIEN_TICH_LON_NHAT = 117.5


def du_doan_gia(dien_tich: float) -> float:
    if not DIEN_TICH_NHO_NHAT <= dien_tich <= DIEN_TICH_LON_NHAT:
        print(
            f"Canh bao: {dien_tich:g} m2 nam ngoai khoang du lieu "
            f"{DIEN_TICH_NHO_NHAT:g}-{DIEN_TICH_LON_NHAT:g} m2, ket qua la ngoai suy."
        )
    return W * dien_tich + B


def main() -> None:
    for dien_tich in (60, 80, 200):
        gia_du_doan = du_doan_gia(dien_tich)
        print(f"Dien tich {dien_tich} m2: {gia_du_doan:.3f} ty dong")


if __name__ == "__main__":
    main()
