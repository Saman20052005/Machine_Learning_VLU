"""Bước 4: tự đếm từ để minh họa vấn đề số không và làm trơn."""
from collections import Counter


def main():
    kho = [("trung thuong ngay", 1), ("khuyen mai giam gia", 1),
           ("trung thuong click", 1), ("hop nhom chieu mai", 0),
           ("bai tap nhom nop mai", 0), ("thay gui tai lieu", 0)]
    tu_dien = sorted({tu for cau, _ in kho for tu in cau.split()})
    dem = {0: Counter(), 1: Counter()}
    for cau, nhan in kho:
        dem[nhan].update(cau.split())
    print(f"Tu dien co {len(tu_dien)} tu:")
    print(" ".join(tu_dien))
    print(f"{'Tu':12s} {'Rac':>5s} {'Thuong':>7s}")
    for tu in tu_dien:
        print(f"{tu:12s} {dem[1][tu]:5d} {dem[0][tu]:7d}")
    cau_thu = "trung thuong bai tap"
    print(f"\nTin nhan thu: {cau_thu}")
    for alpha in [0.0, 1.0]:
        diem = {}
        for nhan in [0, 1]:
            diem[nhan] = 0.5
            for tu in cau_thu.split():
                if tu in tu_dien:
                    diem[nhan] *= (dem[nhan][tu] + alpha) / (
                        sum(dem[nhan].values()) + alpha * len(tu_dien))
        print(f"alpha = {alpha}: rac = {diem[1]:.10f}; thuong = {diem[0]:.10f}")
        print("Ket luan:", "rac" if diem[1] > diem[0] else "thuong")
    print("Khong lam tron: ca hai diem bang 0; ket luan khi hoa chi la quy uoc.")
    print("Lop rac chua gap 'bai', lop thuong chua gap 'trung'.")
    print("Laplace cong alpha vao so dem, tranh xac suat 0 cho tu trong tu dien.")
    print("Tu ngoai tu dien bi bo qua; lam tron khong them tu moi vao tu dien.")


if __name__ == "__main__":
    main()
