"""Bài 1: tính hậu nghiệm cho ba nguyên nhân bằng xác suất toàn phần."""
from math import isclose


def bayes_nhieu_nhanh(tien_nghiem, kha_nang):
    trong_so = [p * likelihood for p, likelihood in zip(tien_nghiem, kha_nang)]
    bang_chung = sum(trong_so)
    return bang_chung, [gia_tri / bang_chung for gia_tri in trong_so]


def main():
    p_qua, hau_nghiem = bayes_nhieu_nhanh([0.50, 0.30, 0.20], [0.95, 0.70, 0.30])
    print("BAI 1: BAYES VOI BA NGUYEN NHAN")
    print("P(qua) = 0.50*0.95 + 0.30*0.70 + 0.20*0.30")
    print(f"       = {p_qua:.4f}")
    for nhom, p in zip(["Hoc deu", "Hoc don", "It hoc"], hau_nghiem):
        print(f"P({nhom} | qua mon) = {p:.6f} ({p:.2%})")
    print(f"Tong hau nghiem = {sum(hau_nghiem):.6f}")
    assert isclose(sum(hau_nghiem), 1.0)
    print("Nhan xet: nhom hoc deu chiem 50% ban dau nhung chiem")
    print("63.76% trong so nguoi qua mon, do ty le qua cua nhom nay cao.")


if __name__ == "__main__":
    main()
