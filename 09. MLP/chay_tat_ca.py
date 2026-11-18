"""Chạy toàn bộ lab 09; lưu stdout và ảnh PNG được dựng từ kết quả thật."""
from pathlib import Path
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def luu_anh(ten, noi_dung, thu_muc):
    """Dựng ảnh văn bản từ stdout; đây không phải ảnh chụp cửa sổ terminal."""
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/consola.ttf"
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
    dong = [f"> python {ten}", ""] + noi_dung.expandtabs(4).splitlines()
    buoc = 28
    width = max(900, int(max(font.getlength(s) for s in dong)) + 64)
    anh = Image.new("RGB", (width, len(dong) * buoc + 56), "#10202e")
    draw = ImageDraw.Draw(anh)
    for i, s in enumerate(dong):
        draw.text((28, 24 + i * buoc), s, font=font, fill="#81e6d9" if i == 0 else "#e6edf3")
    anh.save(thu_muc / f"{Path(ten).stem}.png")


def main():
    ket_qua = ROOT / "ket_qua"
    ket_qua.mkdir(exist_ok=True)
    tep = sorted((ROOT / "code").glob("m*.py")) + sorted((ROOT / "baitap09").glob("bai*.py"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    tong_hop = []
    for script in tep:
        ten = script.relative_to(ROOT).as_posix()
        result = subprocess.run([sys.executable, "-X", "utf8", str(script)], cwd=ROOT,
                                env=env, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"Loi khi chay {ten}:\n{result.stdout}\n{result.stderr}")
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        (ket_qua / f"{script.stem}.txt").write_text(result.stdout, encoding="utf-8")
        luu_anh(ten, result.stdout, ket_qua)
        tong_hop.append(f"{'=' * 72}\n{ten}\n{'=' * 72}\n{result.stdout}")
        print(f"OK: {ten}", flush=True)
    (ket_qua / "tat_ca_ket_qua.txt").write_text("\n".join(tong_hop), encoding="utf-8")
    print(f"Da chay thanh cong {len(tep)} chuong trinh. Ket qua: {ket_qua}")


if __name__ == "__main__":
    main()
