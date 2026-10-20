"""Bài 6: đo thời gian thật bằng perf_counter, lấy trung vị 5 lần."""
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier


def main():
    rng = np.random.default_rng(0)
    X_moi = rng.random((1000, 2))
    so_lan = 5
    print("BAI 6: THOI GIAN FIT VA PREDICT 1000 MAU MOI")
    print(f"Python {platform.python_version()}, sklearn {sklearn.__version__}, {platform.machine()}")
    print("k=5, algorithm=auto, n_jobs=1; trung vi 5 lan, don vi ms.")
    print("Du lieu/nhan ngau nhien, seed=0; sinh du lieu nam ngoai khoang do.")
    print("  N train   Fit (ms)   Predict (ms)   Backend")
    bang = []
    for n in [1000, 10000, 100000]:
        X_hoc = rng.random((n, 2))
        y_hoc = rng.integers(0, 2, size=n)
        # Khởi động các đường mã trước khi đo, không tính khởi tạo thư viện.
        warm = KNeighborsClassifier(5, algorithm="auto", n_jobs=1).fit(X_hoc, y_hoc)
        warm.predict(X_moi)
        fit_times, pred_times = [], []
        for _ in range(so_lan):
            model = KNeighborsClassifier(5, algorithm="auto", n_jobs=1)
            start = perf_counter()
            model.fit(X_hoc, y_hoc)
            fit_times.append((perf_counter() - start) * 1000)
            start = perf_counter()
            pred = model.predict(X_moi)
            pred_times.append((perf_counter() - start) * 1000)
            assert pred.shape == (1000,)
        row = {"n_train": n, "fit_ms_median": float(np.median(fit_times)),
               "predict_1000_ms_median": float(np.median(pred_times)),
               "backend": model._fit_method, "repeats": so_lan}
        bang.append(row)
        print(f"{n:9d} {row['fit_ms_median']:10.3f} {row['predict_1000_ms_median']:14.3f}   {row['backend']}")
    out = Path(__file__).resolve().parents[1] / "ket_qua"
    out.mkdir(exist_ok=True)
    pd.DataFrame(bang).to_csv(out / "bai6_thoi_gian.csv", index=False)
    print("\nNhan xet: KNN luu du lieu, khong toi uu mot bo trong so/toan cuc khi fit.")
    print("Predict moi tim hang xom va bo phieu, nen goi la hoc luoi (lazy learning).")
    print("Auto o du lieu 2 chieu chon KD-tree: fit van can xay chi muc, khong mien phi.")
    print("KD-tree giup predict khong phai duyet tat ca diem nhu brute-force;")
    print("vi vay thoi gian predict co the tang cham hoac dao dong, khong tang tuyen tinh.")
    print("Brute-force co chi phi khoang O(n_train*n_query*d) cho phan tinh khoang cach.")
    print("Thoi gian phu thuoc CPU, tai may, backend va du lieu; khong suy ra do phuc tap")
    print("chi tu ba phep do, cung khong bat buoc fit phai nhanh hon predict voi moi N.")


if __name__ == "__main__":
    main()
