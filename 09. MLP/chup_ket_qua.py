"""Chụp trang hiển thị stdout thật của sáu bài tập bằng Edge headless."""
from html import escape
from pathlib import Path
import os
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent
    edge = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
    if not edge.exists():
        raise FileNotFoundError("Can Microsoft Edge de chup anh trang ket qua.")
    profile = tempfile.mkdtemp(prefix="lab09-edge-")
    for i in range(1, 7):
        stem = f"bai{i}"
        txt = (root / "ket_qua" / f"{stem}.txt").read_text(encoding="utf-8")
        html_path = root / "ket_qua" / f"{stem}.html"
        html_path.write_text(
            '<!doctype html><html lang="vi"><meta charset="utf-8">'
            f'<title>Lab09 - {stem}</title><style>'
            'body{margin:0;background:#10202e;color:#e6edf3;padding:28px;}'
            'pre{font:19px/28px Consolas,monospace;margin:0;white-space:pre-wrap;}'
            '.cmd{color:#81e6d9;margin-bottom:24px;}</style>'
            f'<pre class="cmd">&gt; python baitap09/{stem}.py</pre>'
            f'<pre>{escape(txt)}</pre></html>', encoding="utf-8")
        screenshot = root / "ket_qua" / f"{stem}_screenshot.png"
        height = max(500, 28 * len(txt.splitlines()) + 150)
        result = subprocess.run([
            str(edge), "--headless", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={profile}",
            f"--screenshot={screenshot}", f"--window-size=1150,{height}",
            "--hide-scrollbars", html_path.as_uri(),
        ], capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode or not screenshot.exists():
            raise RuntimeError(f"Khong chup duoc {stem}: {result.stderr!r}")
        print(f"OK: {screenshot.name}", flush=True)


if __name__ == "__main__":
    main()
