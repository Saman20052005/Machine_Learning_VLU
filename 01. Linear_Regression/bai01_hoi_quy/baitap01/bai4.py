import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    du_lieu_nha = pd.read_csv("data/gia_nha.csv")
    dac_trung_hai_bien = du_lieu_nha[["dien_tich", "so_phong"]]
    dac_trung_dien_tich = du_lieu_nha[["dien_tich"]]
    gia_nha = du_lieu_nha["gia"]

    x_train, x_test, y_train, y_test = train_test_split(
        dac_trung_hai_bien, gia_nha, test_size=0.2, random_state=42
    )
    mo_hinh_hai_bien = LinearRegression().fit(x_train, y_train)
    r2_hai_bien = mo_hinh_hai_bien.score(x_test, y_test)

    x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(
        dac_trung_dien_tich, gia_nha, test_size=0.2, random_state=42
    )
    r2_dien_tich = LinearRegression().fit(x_train_dt, y_train_dt).score(x_test_dt, y_test_dt)

    print(f"Train: {len(x_train)}")
    print(f"Test: {len(x_test)}")
    print(f"R2 hai bien: {r2_hai_bien:.6f}")
    print(f"R2 chi dien tich: {r2_dien_tich:.4f}")
    print(
        "Them so phong co cai thien mo hinh so voi chi dung dien tich, "
        "nhung muc cai thien la nho."
    )


if __name__ == "__main__":
    main()
