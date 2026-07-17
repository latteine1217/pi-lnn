#!/usr/bin/env python3
"""Build a submission-safe vector PDF from an SVG figure.

Pipeline:  SVG --(Chrome headless, @font-face Latin Modern)--> PDF
                --(Ghostscript -dNoOutputFonts)--> outlined vector PDF

Why this chain:
  * Chrome has the most complete SVG renderer available here and embeds fonts,
    but it emits **Type 3** fonts for OTF/CFF sources -- a common desk-reject.
  * Ghostscript -dNoOutputFonts converts every glyph to vector outlines, so the
    result contains *no fonts at all*: no Type 3, no embedding question, and the
    rendering is identical everywhere. Text is no longer selectable, which is
    the accepted trade for figures.
  * Latin Modern is the OpenType descendant of Computer Modern, so figure text
    matches the thesis body face.

Usage:
    uv run python scripts/build_svg_figure.py <in.svg> <out.pdf>
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LM_DIR = Path("/usr/local/texlive/2026basic/texmf-dist/fonts/opentype/public/lm")
# font-family name -> (file, weight, style)
FACES = [
    ("lmroman10-regular.otf", 400, "normal"),
    ("lmroman10-bold.otf", 700, "normal"),
    ("lmroman10-italic.otf", 400, "italic"),
]
PX_PER_PT = 4.0 / 3.0  # CSS px -> PDF pt is 0.75


def _face_css() -> str:
    out = []
    for fname, weight, style in FACES:
        p = LM_DIR / fname
        if not p.exists():
            raise SystemExit(f"font not found: {p}")
        b64 = base64.b64encode(p.read_bytes()).decode()
        out.append(
            "@font-face{font-family:LM;"
            f"src:url(data:font/otf;base64,{b64}) format('opentype');"
            f"font-weight:{weight};font-style:{style};}}"
        )
    return "".join(out)


def _svg_size(svg: str) -> tuple[float, float]:
    import re

    w = re.search(r'<svg[^>]*\bwidth="([\d.]+)', svg)
    h = re.search(r'<svg[^>]*\bheight="([\d.]+)', svg)
    if not (w and h):
        raise SystemExit("SVG must declare numeric width/height in px on <svg>")
    return float(w.group(1)), float(h.group(1))


def build(svg_path: Path, out_pdf: Path) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    w, h = _svg_size(svg)
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        + _face_css()
        + f"@page{{size:{w}px {h}px;margin:0}}"
        # <svg> is an inline element by default: it sits on a text baseline, so ~4 px of
        # descender space is added below it, the page overflows and Chrome emits a blank
        # second page. display:block removes the baseline.
        "html,body{margin:0;padding:0;overflow:hidden}svg{display:block}</style></head><body>"
        + svg
        + "</body></html>"
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "f.html").write_text(html, encoding="utf-8")
        raw = tmp / "raw.pdf"
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={raw}", str(tmp / "f.html")],
            check=True, capture_output=True,
        )
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gs", "-o", str(out_pdf), "-sDEVICE=pdfwrite", "-dNoOutputFonts", str(raw)],
            check=True, capture_output=True,
        )

    # verify: no fonts, no raster, expected page size
    data = out_pdf.read_bytes()
    fonts = subprocess.run(["pdffonts", str(out_pdf)], capture_output=True, text=True).stdout
    n_fonts = len([l for l in fonts.splitlines()[2:] if l.strip()])
    n_img = data.count(b"/Subtype /Image") + data.count(b"/Subtype/Image")
    info = subprocess.run(["pdfinfo", str(out_pdf)], capture_output=True, text=True).stdout
    size = [l for l in info.splitlines() if l.startswith("Page size")]
    pages = int(next(l.split()[-1] for l in info.splitlines() if l.startswith("Pages:")))
    print(f"  {out_pdf.name}: pages={pages} fonts={n_fonts} raster={n_img}  "
          f"{size[0] if size else ''}")
    # A spurious 2nd page means the canvas overflowed @page. \includegraphics would
    # silently take page 1 and hide it, so assert rather than trust the eye.
    if pages != 1:
        raise SystemExit(f"FAILED: expected a single page, got {pages} (canvas overflow)")
    # Fonts must be zero: gs outlines every glyph, so any survivor means the
    # outlining silently failed and a Type 3 / unembedded face would ship.
    # Raster IS allowed: field and contour panels are genuine bitmaps (rendered
    # at 300 dpi), which is the correct use of raster in a vector figure.
    if n_fonts:
        raise SystemExit("FAILED: output still carries fonts; text was not outlined")
    print(f"  target width {w * 0.75:.1f}pt x {h * 0.75:.1f}pt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME}")
    if not shutil.which("gs"):
        raise SystemExit("ghostscript (gs) not found")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
