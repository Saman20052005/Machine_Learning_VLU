"""Bước 1: định lý Bayes với hai nguyên nhân."""


def bayes(tien_nghiem, kha_nang, kha_nang_nguoc):
    tu_so = tien_nghiem * kha_nang
    return tu_so / (tu_so + (1 - tien_nghiem) * kha_nang_nguoc)


if __name__ == "__main__":
    print("VI DU 1: XET NGHIEM BENH")
    print(f"P(benh | duong tinh) = {bayes(0.01, 0.99, 0.01):.4f}")
    print("Ty le mac benh   Hau nghiem khi duong tinh")
    for ty_le in [0.001, 0.01, 0.05, 0.10, 0.50]:
        print(f"{ty_le:14.3f}   {bayes(ty_le, 0.99, 0.01):.4f}")
    print("\nVI DU 2: HAI DAY CHUYEN SAN XUAT")
    print(f"P(day B | san pham loi) = {bayes(0.20, 0.05, 0.01):.4f}")
    print(f"Dem tren 1000 san pham: A co 8 loi, B co 10 loi: {10 / 18:.4f}")
