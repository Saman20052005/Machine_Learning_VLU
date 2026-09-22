# Lab 1 - Hoi quy tuyen tinh

Lab nay thuc hanh doc du lieu gia nha, trinh bay du lieu va xay dung cac mo hinh hoi quy tuyen tinh co ban.

## Cau truc

```text
bai01_hoi_quy/
├── data/gia_nha.csv
├── baitap01/
│   ├── bai1.py ... bai6.py
│   └── bai2.png
├── outputs/
│   └── ket qua chay thuc te cua tung bai
├── README.md
└── requirements.txt
```

## Cai dat tren macOS

Tu thu muc `bai01_hoi_quy`, tao va kich hoat moi truong ao:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chay bai tap

```bash
python3 baitap01/bai1.py
python3 baitap01/bai2.py
python3 baitap01/bai3.py
python3 baitap01/bai4.py
python3 baitap01/bai5.py --learning-rate 0.001
python3 baitap01/bai5.py --learning-rate 1.02
python3 baitap01/bai6.py
```

## Tom tat ket qua

| Bai | Ket qua chinh |
| --- | --- |
| 1 | Tim duoc 10 can co dien tich lon hon 100 m2; gia trung binh 8.962 ty dong. |
| 2 | Bieu do scatter cho thay gia co xu huong tang theo so phong, nhung co phan tan. |
| 3 | He so cua tuoi nha am: `w = -0.037858`, `b = 6.983593`. |
| 4 | Mo hinh dien tich va so phong co R2 khoang 0.969757 tren tap test. |
| 5 | So sanh gradient descent voi learning rate 0.001 va 1.02 sau 200 vong. |
| 6 | Du doan gia va canh bao khi dien tich nam ngoai mien du lieu. |
