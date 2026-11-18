# -*- coding: utf-8 -*-
"""Bài 4: Lưu ảnh 16 màu, so dung lượng với ảnh gốc và thử với ảnh PNG có kênh alpha."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


def nen_anh_16_mau(anh_rgb_float):
    cao, rong, _ = anh_rgb_float.shape
    diem = anh_rgb_float.reshape(-1, 3)
    rng = np.random.default_rng(0)
    so_mau = min(5000, len(diem))
    mau = diem[rng.choice(len(diem), so_mau, replace=False)]
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
    nhan = km.predict(diem)
    anh_moi = np.clip(km.cluster_centers_[nhan].reshape(cao, rong, 3), 0.0, 1.0)
    return anh_moi


def main():
    root = Path(__file__).resolve().parents[1]
    bieu_do = root / "bieu_do"
    bieu_do.mkdir(exist_ok=True)

    # Phần 1: ảnh mẫu china.jpg
    anh = load_sample_image("china.jpg") / 255.0
    anh_goc_path = bieu_do / "anh_goc.png"
    anh_16_path = bieu_do / "anh_16_mau.png"
    plt.imsave(anh_goc_path, anh)
    anh_16 = nen_anh_16_mau(anh)
    plt.imsave(anh_16_path, anh_16)

    kb_goc = anh_goc_path.stat().st_size / 1024
    kb_16 = anh_16_path.stat().st_size / 1024
    print(f"Anh china.jpg ({anh.shape[0]}x{anh.shape[1]}):")
    print(f"  anh_goc.png   : {kb_goc:7.1f} KB")
    print(f"  anh_16_mau.png: {kb_16:7.1f} KB (giam {(1 - kb_16 / kb_goc):.1%})")
    print()

    # Phần 2: ảnh PNG có 4 kênh RGBA (tạo từ flower.jpg kèm kênh alpha để minh họa)
    flower = load_sample_image("flower.jpg")
    alpha = np.full((flower.shape[0], flower.shape[1], 1), 255, dtype=np.uint8)
    flower_rgba = np.concatenate([flower, alpha], axis=2)
    flower_rgba_path = root / "data/flower_rgba.png"
    plt.imsave(flower_rgba_path, flower_rgba)

    doc_vao = plt.imread(flower_rgba_path)
    print(f"Doc flower_rgba.png bang plt.imread: shape = {doc_vao.shape}")
    if doc_vao.shape[2] == 4:
        doc_vao = doc_vao[..., :3]
        print(f"Da bo kenh thu 4 (alpha), shape con lai: {doc_vao.shape}")

    flower_16 = nen_anh_16_mau(doc_vao.astype(float))
    flower_16_path = bieu_do / "flower_16_mau.png"
    plt.imsave(flower_16_path, flower_16)
    kb_f_goc = flower_rgba_path.stat().st_size / 1024
    kb_f_16 = flower_16_path.stat().st_size / 1024
    print(f"  flower_rgba.png  : {kb_f_goc:7.1f} KB")
    print(f"  flower_16_mau.png: {kb_f_16:7.1f} KB (giam {(1 - kb_f_16 / kb_f_goc):.1%})")


if __name__ == "__main__":
    main()
