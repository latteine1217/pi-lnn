"""check_slide_greek.py — 抓出被 CSS uppercase 改寫的希臘字母。

What: 掃 thesis/slide/slides.md，找出位於 uppercase 標籤內、且未以 <span class="raw">
      包住的小寫希臘字母。
Why : text-transform: uppercase 會把 ω→Ω、π→Π、ν→Ν、σ→Σ、τ→Τ。在本論文的符號系統中
      這些是不同的量（ω 渦度 vs Ω 計算域；π 圓周率 vs Π 連乘），印錯即為錯誤內容。
      style.css 的全域 .raw 規則提供豁免，但「記得包」無法靠自律 —— 此檔把「忘記」
      變成可偵測，而非下次再踩。
用法: uv run python scripts/check_slide_greek.py     （exit 1 表示有未豁免的符號）

限制: 靜態檢查，只涵蓋已知會套 uppercase 的容器（LabelTiny / SectionTag / <th>）。
      若日後新增其他 uppercase 樣式，需把它的容器加進 UPPERCASE_CONTAINERS。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "thesis/slide/slides.md"

# 會被 uppercase 改寫、且在本論文有大寫同形異義的字母
RISKY = "ωπνστβγδθμφχψλρ"

# 元件層：LabelTiny.vue / SectionTag.vue 本身宣告 uppercase，內容一律受影響。
COMPONENT_CONTAINERS = [
    ("LabelTiny", re.compile(r"<LabelTiny>(.*?)</LabelTiny>", re.S)),
    ("SectionTag", re.compile(r"<SectionTag>(.*?)</SectionTag>", re.S)),
]

# 表格層：只有「其 class 有 `.<cls> th { … uppercase }` 規則」的表，th 才受影響。
# 不可假設所有 <th> 都是 uppercase —— slides.md 內多數 <table class="w-full"> 並未套用，
# 對那些表誤報會讓 guard 失去可信度。
TABLE_OPEN = re.compile(r'<table[^>]*\bclass="([^"]*)"[^>]*>', re.S)
TH = re.compile(r"<th\b[^>]*>(.*?)</th>", re.S)
UPPERCASE_TH_RULE = re.compile(r"\.([a-zA-Z0-9_-]+)\s+th\s*\{[^}]*text-transform:\s*uppercase", re.S)

RAW_SPAN = re.compile(r'<span class="raw">.*?</span>', re.S)


def main() -> int:
    text = SLIDES.read_text(encoding="utf-8")
    line_of = lambda i: text.count("\n", 0, i) + 1  # noqa: E731

    def exposed_greek(inner: str) -> str:
        """移除已 .raw 豁免的部分，回傳仍暴露在 uppercase 下的高風險字母。"""
        return "".join(sorted({c for c in RAW_SPAN.sub("", inner) if c in RISKY}))

    bad: list[tuple[int, str, str]] = []

    for label, pat in COMPONENT_CONTAINERS:
        for m in pat.finditer(text):
            hits = exposed_greek(m.group(1))
            if hits:
                bad.append((line_of(m.start()), label, f"{hits}  in  {m.group(1).strip()[:64]}"))

    # 哪些 table class 的 th 真的套了 uppercase
    upper_classes = set(UPPERCASE_TH_RULE.findall(text))

    for tm in TABLE_OPEN.finditer(text):
        classes = set(tm.group(1).split())
        if not (classes & upper_classes):
            continue  # 這張表的 th 未套 uppercase，跳過
        close = text.find("</table>", tm.end())
        body = text[tm.end(): close if close != -1 else len(text)]
        for m in TH.finditer(body):
            hits = exposed_greek(m.group(1))
            if hits:
                bad.append((line_of(tm.end() + m.start()),
                            f'<th> in table.{"/".join(sorted(classes & upper_classes))}',
                            f"{hits}  in  {m.group(1).strip()[:64]}"))

    if not bad:
        print(f"[greek] OK — no unwrapped Greek inside uppercase labels ({SLIDES.name})")
        return 0

    print(f"[greek] {len(bad)} unwrapped Greek symbol(s) inside uppercase labels:\n")
    for ln, label, detail in sorted(bad):
        print(f"  slides.md:{ln}  {label}\n      {detail}")
    print('\n  Fix: wrap the symbol —  <span class="raw">ω</span>')
    print("  (.raw exemption is declared globally in thesis/slide/style.css)")
    return 1


# UnoCSS attributify mode 會把 SVG 的 font-size="12" 當成 utility attribute，
# 生成 [font-size~="12"] { font-size: 3rem }，靜靜地把 12 個單位變成 48px。
# 實測：文字爆出 viewBox，但 attribute 讀回來仍是 "12" —— 只有 getComputedStyle 看得到。
FONT_SIZE_ATTR = re.compile(r'<(?:text|tspan)\b[^>]*\bfont-size="')


def check_svg_font_size() -> int:
    """SVG 內用 font-size 屬性 → 被 UnoCSS attributify 劫持。必須改 inline style。"""
    text = SLIDES.read_text(encoding="utf-8")
    hits = [text[:m.start()].count("\n") + 1 for m in FONT_SIZE_ATTR.finditer(text)]
    if not hits:
        print(f"[svg] OK — no font-size attribute on SVG text ({SLIDES.name})")
        return 0
    print(f"[svg] {len(hits)} SVG font-size attribute(s) — UnoCSS attributify will hijack these:\n")
    for ln in hits:
        print(f"  slides.md:{ln}")
    print('\n  Fix: use inline style —  <text style="font-size:12px">')
    return 1


if __name__ == "__main__":
    sys.exit(main() | check_svg_font_size())
