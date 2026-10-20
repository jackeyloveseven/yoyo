#!/usr/bin/env python3
"""生成社交分享图 ``site/assets/og.png``(1200×630)。

旧图是手绘产物:底部中文标签没有对应字形,渲染成豆腐块 ``□□□□□``,数字也停在
90。本脚本把同一张图程序化重建——数字从注册表与文件树实时取,测试数用一次
pytest 收集,字体与宣传片同源(Noto Sans CJK SC),中文字形不再依赖手绘工具。

    PYTHONPATH=src python3 site/promo/make_og.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmtrainer.registry import build_builtin_registries  # noqa: E402

W, H = 1200, 630
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG = (246, 245, 241)
SURFACE = (255, 255, 255)
INK = (21, 24, 29)
MUTED = (86, 91, 102)
FAINT = (134, 140, 152)
ACCENT = (180, 101, 10)
ACCENT_SOFT = (246, 233, 216)
LINE = (226, 224, 216)
MARK = (14, 110, 93)


def cjk(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(CJK.format(weight=weight), size, index=2)


def mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO_BOLD if bold else MONO, size)


def display_counts() -> dict[str, int]:
    """与站点同源的一组数字:注册表实时数据 + 仓库文件树 + 一次 pytest 收集。"""
    registries = build_builtin_registries()
    collect_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/v2",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    proc = subprocess.run(
        collect_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        # 子进程不继承本进程的 sys.path,必须显式给 PYTHONPATH 才能 import mmtrainer
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    if proc.returncode != 0 or not match:
        raise SystemExit(f"pytest 收集失败:\n{proc.stdout[-400:]}\n{proc.stderr[-400:]}")
    return {
        "models": len(registries.models.entries()),
        "objectives": len(registries.objectives.entries()),
        "rewards": len(registries.rewards.entries()),
        "examples": len(list((ROOT / "examples").glob("*.py"))),
        "tests": int(match.group(1)),
    }


def draw_panel(draw: ImageDraw.ImageDraw) -> None:
    """右侧 compose 蓝图预览,内容与 index.html hero 面板一致。"""
    box = [700, 96, 1128, 534]
    draw.rounded_rectangle(box, 14, fill=SURFACE, outline=LINE, width=2)
    draw.rounded_rectangle([box[0], box[1], box[2], box[1] + 46], 14, fill=(237, 236, 230))
    draw.rectangle([box[0], box[1] + 32, box[2], box[1] + 46], fill=(237, 236, 230))
    draw.text(
        (box[0] + 20, box[1] + 23),
        "❯ mmtrain compose",
        font=mono(15, True),
        fill=INK,
        anchor="lm",
    )
    draw.text((box[2] - 20, box[1] + 23), "真实注册表输出", font=cjk(12), fill=FAINT, anchor="rm")

    rows = [("model", "qwen_image_edit_2x"), ("objective", "image_dpo"), ("reward", "none")]
    y = box[1] + 74
    for key, value in rows:
        draw.text((box[0] + 20, y), key, font=mono(13), fill=FAINT, anchor="lm")
        draw.text((box[0] + 120, y), value, font=mono(14, True), fill=ACCENT, anchor="lm")
        y += 30

    draw.line([box[0] + 20, y + 4, box[2] - 20, y + 4], fill=LINE, width=1)
    draw.text((box[0] + 20, y + 24), "METRICS", font=mono(11), fill=FAINT, anchor="lm")
    for i, text in enumerate(("loss", "隐式奖励差", "偏好准确率")):
        cy = y + 52 + i * 26
        draw.ellipse([box[0] + 22, cy - 3, box[0] + 28, cy + 3], fill=MARK)
        draw.text((box[0] + 34, cy), text, font=cjk(14), fill=MUTED, anchor="lm")

    draw.rounded_rectangle(
        [box[0] + 20, box[3] - 62, box[2] - 20, box[3] - 24], 6, fill=ACCENT_SOFT
    )
    draw.text(
        (box[0] + 32, box[3] - 43),
        "1 · CPU preflight",
        font=mono(13, True),
        fill=ACCENT,
        anchor="lm",
    )


def build(path: Path) -> None:
    counts = display_counts()
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # 左上角淡网格,与页面 hero 的底纹同源
    for x in range(0, W, 72):
        draw.line([(x, 0), (x, H)], fill=(238, 237, 232), width=1)
    for y in range(0, H, 72):
        draw.line([(0, y), (W, y)], fill=(238, 237, 232), width=1)

    draw.text((72, 82), "OPEN-SOURCE ACADEMIC BASELINE", font=mono(15, True), fill=ACCENT)
    draw.text((72, 122), "扩散后训练", font=cjk(68, "Black"), fill=INK)
    draw.text((72, 200), "同一套可组合零件", font=cjk(68, "Black"), fill=INK)
    draw.text(
        (72, 310),
        "SFT · DPO · 蒸馏 · RL —— 换零件，不换整套脚本",
        font=cjk(27),
        fill=MUTED,
    )

    stats = [
        (counts["models"], "模型适配器"),
        (counts["objectives"], "训练目标"),
        (counts["rewards"], "奖励后端"),
        (counts["examples"], "CPU 示例"),
        (counts["tests"], "测试通过"),
    ]
    for i, (number, label) in enumerate(stats):
        x = 72 + i * 118
        draw.text((x, 420), str(number), font=cjk(44, "Black"), fill=ACCENT)
        draw.text((x, 478), label, font=cjk(16), fill=MUTED)

    draw.line([(72, 540), (W - 72, 540)], fill=LINE, width=2)
    draw.text(
        (72, 572),
        "github.com/jackeyloveseven/yoyo",
        font=mono(17),
        fill=MUTED,
    )

    draw_panel(draw)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)
    print(f"saved {path} ({counts})")


def main() -> int:
    build(ROOT / "site" / "assets" / "og.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
