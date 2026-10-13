#!/usr/bin/env python3
"""YoYo 宣传视频渲染器(PIL 逐帧绘制 + ffmpeg 合成)。

产出 1080p / 30fps / 带配乐 MP4(10 分镜 73.5 秒)与海报帧:

    python3 render_promo.py                       # 全片渲染 + 编码
    python3 render_promo.py --frame hero:0.8     # 单帧调试,输出 PNG

分镜:hero(品牌) → pain(传统脚本复制) → parts(零件拆解) → compose(交互演示)
→ flow(流匹配概念) → numbers(真实数字) → gates(五道质量门) → baseline(学术基线)
→ quickstart(CPU 跑通) → cta。配乐为程序化合成(见 build_audio)。

营销红线:FireRed 不出现;Qwen 主打 Edit 2.x;AWM/RAM 不宣传;
"源码可用 ≠ GPU 就绪" 的诚实边界必须可见。所有数字与主仓库注册表同源。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parent
CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc"
CJK_INDEX = 2  # ttc 内的 SC(Simplified Chinese)字面
MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# ---- 调色板(与站点设计 token 对齐;子代理定稿后统一改这里) ----
PALETTE = {
    # 对齐站点 dark 主题 token(index.html/site.css:--bg/--surface/--fg/--accent...)
    "bg": (16, 19, 24),  # #101318
    "panel": (23, 27, 34),  # #171B22
    "line": (52, 59, 71),  # 分隔线,页面 --fg-faint 调暗
    "fg": (236, 239, 244),  # #ECEFF4
    "muted": (162, 170, 184),  # #A2AAB8
    "accent": (240, 169, 59),  # #F0A93B 琥珀
    "accent2": (79, 209, 176),  # #4FD1B0 可跑/正向(页面 --mark)
    "warn": (240, 169, 59),  # 警示同琥珀系,避免引入页面外新色相
    "danger": (224, 108, 117),
}


class Fonts:
    """字体缓存:CJK 用 Noto Sans SC,代码用 DejaVu Sans Mono。"""

    def __init__(self) -> None:
        self._cache: dict[str, ImageFont.FreeTypeFont] = {}

    def _get(self, path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
        key = f"{path}:{index}:{size}"
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(path, size, index=index)
        return self._cache[key]

    def cjk(self, size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
        return self._get(CJK_PATH.format(weight=weight), size, CJK_INDEX)

    def mono(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return self._get(MONO_BOLD_PATH if bold else MONO_PATH, size)


# ---- 缓动与包络 ----


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smooth(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def scene_alpha(t01: float) -> float:
    """分镜首尾渐显渐隐。时长越短配比越大(render_frame 用它做背景浮现)。"""
    rise = smooth(t01 * 24)
    fall = smooth((1.0 - t01) * 12)
    return min(rise, fall)


def stagger(t01: float, index: int, per: float = 0.10) -> float:
    """第 index 个元素的入场进度:窗口固定 0.15,各元素延迟 index*per,不随场景长度伸缩。"""
    start_offset = index * per
    return smooth((t01 - start_offset) * (1.0 / 0.15))


def mix(c0, c1, t: float):
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1, strict=True))


def with_alpha(color, alpha: float):
    r, g, b = color[:3]
    return (int(r * alpha), int(g * alpha), int(b * alpha), int(alpha * 255))


# ---- 通用绘制 ----


def draw_background(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, seed: int) -> None:
    """点阵 + 上浮微粒的深色底。"""
    bg = PALETTE["bg"]
    draw.rectangle([0, 0, W, H], fill=bg)
    # 点阵(带轻微视差)
    off = (t * 10) % 64
    dot = PALETTE["line"]
    for y in range(int(-off), H + 64, 64):
        fade_y = 1.0 - abs(y - H / 2) / (H / 2)
        col = mix(PALETTE["bg"], dot, 0.35 * fade_y)
        for x in range(0, W, 64):
            r = 1.2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
    # 微粒(确定性随机)
    rng = np.random.default_rng(20261013 + seed)
    xs = rng.uniform(0, W, 56)
    speed = rng.uniform(24, 60, 56)
    sizes = rng.uniform(0.8, 2.0, 56)
    for x, sp, sz in zip(xs, speed, sizes, strict=True):
        y = H + 20 - ((sp * t) % (H + 40))
        tw = 0.5 + 0.5 * math.sin(t * 1.5 + x * 0.013)
        col = mix(PALETTE["bg"], PALETTE["accent"], 0.16 + 0.14 * tw)
        draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=col)


def glow(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    font,
    fill,
    glow_color,
    radius: int = 22,
) -> None:
    """柔光字:模糊光晕叠在正文后面。"""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, text, font=font, fill=glow_color, anchor="mm")
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius)))
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def panel(
    draw: ImageDraw.ImageDraw,
    box,
    radius: int = 16,
    fill=None,
    outline=None,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)


def title(
    draw: ImageDraw.ImageDraw,
    text: str,
    t_in: float,
    y: float = 130,
    size: int = 56,
) -> None:
    """分镜标题:淡入 + 轻微上移,居中。"""
    if t_in <= 0:
        return
    k = smooth(t_in)
    y_off = (1 - k) * 24
    color = mix(PALETTE["bg"], PALETTE["fg"], k)
    draw.text((W / 2, y + y_off), text, font=FONTS.cjk(size, "Bold"), fill=color, anchor="mm")


def caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    t_in: float,
    y: float = H - 96,
    size: int = 30,
) -> None:
    if t_in <= 0:
        return
    k = smooth(t_in)
    color = mix(PALETTE["bg"], PALETTE["muted"], k)
    draw.text((W / 2, y), text, font=FONTS.cjk(size), fill=color, anchor="mm")


FONTS = Fonts()

# ---- 分镜 ----

SCENES: list[tuple[str, float, object]] = []


def scene(name: str, seconds: float):
    def deco(fn):
        SCENES.append((name, seconds, fn))
        return fn

    return deco


@scene("hero", 4.5)
def sc_hero(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 3.2, seed=0)
    k = smooth(t01 * 4)
    y = H / 2 + (1 - k) * 46
    glow(
        img,
        draw,
        (W / 2, y - 70),
        "YoYo",
        FONTS.cjk(150, "Black"),
        PALETTE["fg"],
        with_alpha(PALETTE["accent"], 130 * scene_alpha(t01)),
        radius=30,
    )
    sub = "扩散 / 流匹配模型的后训练 · 可组合零件库"
    k2 = smooth(t01 * 8 - 0.25)
    if k2 > 0:
        color = mix(PALETTE["bg"], PALETTE["muted"], k2)
        draw.text((W / 2, y + 26), sub, font=FONTS.cjk(44), fill=color, anchor="mm")
    pills = ["SFT", "DPO", "蒸馏", "RL", "对齐"]
    pw, ph, gap = 220, 66, 22
    x0 = W / 2 - (len(pills) * pw + (len(pills) - 1) * gap) / 2
    for i, label in enumerate(pills):
        kk = stagger(t01, i, 0.06)
        if kk <= 0:
            continue
        x = x0 + i * (pw + gap)
        box = [x, y + 92, x + pw, y + 92 + ph]
        panel(
            draw,
            box,
            33,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], PALETTE["accent"], kk),
            width=2,
        )
        draw.text(
            (x + pw / 2, y + 92 + ph / 2 - 2),
            label,
            font=FONTS.cjk(30, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["fg"], kk),
            anchor="mm",
        )
    url = "github.com/jackeyloveseven/yoyo"
    k4 = smooth(t01 * 8 - 1.2)
    if k4 > 0:
        color = mix(PALETTE["bg"], PALETTE["accent"], k4)
        draw.text((W / 2, H - 88), url, font=FONTS.mono(28), fill=color, anchor="mm")


@scene("pain", 7.5)
def sc_pain(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 1.5, seed=1)
    title(draw, "传统做法:每个任务,复制一整套脚本", smooth(t01 * 10))
    names = ["SFT 脚本", "DPO 脚本", "RL 脚本", "蒸馏脚本"]
    inner = ["模型加载", "加噪 x_t", "ODE 求解器", "学习率表", "FSDP 封装"]
    bw, bh = 380, 470
    gap = (W - 4 * bw) / 5
    flash = math.sin(t01 * math.pi * 6) * 0.5 + 0.5 if 0.35 < t01 < 0.8 else 0.0
    for i, name in enumerate(names):
        kk = stagger(t01, i, 0.08)
        if kk <= 0:
            continue
        x = gap + i * (bw + gap)
        y0 = 210
        panel(
            draw,
            [x, y0, x + bw, y0 + bh],
            18,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], PALETTE["line"], kk),
            width=2,
        )
        draw.text(
            (x + bw / 2, y0 + 44),
            name,
            font=FONTS.cjk(30, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["fg"], kk),
            anchor="mm",
        )
        for j, block in enumerate(inner):
            kj = smooth(kk * 1.6 - j * 0.05)
            if kj <= 0:
                continue
            by = y0 + 92 + j * 72
            base = mix(PALETTE["panel"], PALETTE["line"], 0.8)
            col = mix(base, PALETTE["danger"], flash * 0.55)
            panel(draw, [x + 26, by, x + bw - 26, by + 54], 10, fill=mix(PALETTE["bg"], col, kj))
            draw.text(
                (x + bw / 2, by + 27),
                block,
                font=FONTS.cjk(26),
                fill=mix(PALETTE["bg"], PALETTE["muted"], kj),
                anchor="mm",
            )
    caption(draw, "同一段逻辑复制了 N 份 —— 改一处,要改十处", smooth(t01 * 8 - 0.9))


@scene("parts", 8.0)
def sc_parts(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 1.2, seed=2)
    title(draw, "YoYo 的做法:零件各一份,拼装组合", smooth(t01 * 10))
    parts = [
        ("nn/", "流插值 · Euler/DPM2 求解器 · 学习率表"),
        ("algorithms/", "SFT / DPO / 蒸馏 / RL 损失,纯函数"),
        ("models/", "模型适配器,25 行接入新模型"),
        ("datasets/", "JSONL 清单加载"),
        ("rewards/", "ArcFace 身份 · VLM 评分 · 美学分"),
    ]
    bw, bh = 1040, 96
    y0 = 214
    for i, (name, desc) in enumerate(parts):
        kk = stagger(t01, i, 0.07)
        if kk <= 0:
            continue
        x = W / 2 - bw / 2 + (1 - kk) * 60
        y = y0 + i * (bh + 22)
        panel(
            draw,
            [x, y, x + bw, y + bh],
            16,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], PALETTE["accent"], kk * 0.9),
            width=2,
        )
        draw.text(
            (x + 36, y + bh / 2 - 4),
            name,
            font=FONTS.mono(30, bold=True),
            fill=mix(PALETTE["bg"], PALETTE["accent"], kk),
            anchor="lm",
        )
        draw.text(
            (x + 36, y + bh / 2 + 34),
            desc,
            font=FONTS.cjk(26),
            fill=mix(PALETTE["bg"], PALETTE["muted"], kk),
            anchor="lm",
        )
    k_hook = smooth(t01 * 8 - 0.85)
    if k_hook > 0:
        text = "换一种训练方式?换零件,不换整套脚本。"
        draw.text(
            (W / 2, y0 + 5 * (bh + 22) + 22),
            text,
            font=FONTS.cjk(32, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["accent"], k_hook),
            anchor="mm",
        )
    caption(draw, "每个零件独立单测 · 单文件 ≤800 行 · 零 50-token 重复", smooth(t01 * 8 - 1.3))


@scene("compose", 8.0)
def sc_compose(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 0.9, seed=3)
    title(draw, "mmtrain compose:选好零件,出一份训练蓝图", smooth(t01 * 10))
    # 终端窗口
    tw, th = 1280, 640
    x0, y0 = W / 2 - tw / 2, 190
    k = smooth(t01 * 8)
    panel(
        draw,
        [x0, y0, x0 + tw, y0 + th],
        18,
        fill=mix(PALETTE["bg"], PALETTE["panel"], k),
        outline=mix(PALETTE["bg"], PALETTE["line"], k),
        width=2,
    )
    for i, c in enumerate((PALETTE["danger"], PALETTE["warn"], PALETTE["accent"])):
        cx = x0 + 34 + i * 34
        draw.ellipse([cx - 8, y0 + 30 - 8, cx + 8, y0 + 30 + 8], fill=mix(PALETTE["bg"], c, k))
    cmd = "mmtrain compose --model qwen_image_edit_2x --objective image_sft"
    out_lines = [
        ("模型", "qwen_image_edit_2x —— Qwen-Image-Edit 2.x(1-3 张控制图)", "fg"),
        ("目标", "image_sft(流匹配 SFT,无需 reward)", "fg"),
        ("指标", "total_loss · per_sample_mse", "fg"),
        ("可视化", "目标图 vs 学生一步采样,逐 epoch 对比", "fg"),
        ("下一步", "CPU preflight → GPU no-update → one-update", "accent"),
    ]
    cmd_font = FONTS.mono(27)
    # 打字机:命令行 0.1~0.45,输出行 0.45~0.9 逐行浮现
    typed = int(len(cmd) * smooth((t01 - 0.10) * (1 / 0.32)))
    if typed > 0:
        draw.text(
            (x0 + 40, y0 + 78), "$ " + cmd[:typed], font=cmd_font, fill=PALETTE["fg"], anchor="lm"
        )
    if t01 > 0.45 and int(t01 * FPS) % 16 < 11 and t01 < 0.47:
        wcmd = draw.textlength("$ " + cmd[:typed], font=cmd_font)
        cursor = [x0 + 40 + wcmd + 6, y0 + 64, x0 + 40 + wcmd + 19, y0 + 92]
        draw.rectangle(cursor, fill=PALETTE["accent"])
    for i, (label, content, tone) in enumerate(out_lines):
        ki = smooth((t01 - 0.48 - i * 0.07) * 6)
        if ki <= 0:
            continue
            # pragma: no cover
        ly = y0 + 150 + i * 88
        color = PALETTE["accent"] if tone == "accent" else PALETTE["fg"]
        draw.text(
            (x0 + 60, ly),
            label,
            font=FONTS.cjk(28, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["muted"], ki),
            anchor="lm",
        )
        draw.text(
            (x0 + 60 + 140, ly),
            content,
            font=FONTS.cjk(28),
            fill=mix(PALETTE["bg"], color, ki),
            anchor="lm",
        )
    caption(draw, "诚实边界:源码可用 ≠ GPU 就绪,蓝图含晋级阶梯", smooth(t01 * 8 - 0.75))


def _draw_velocity_arrows(draw, noise: np.ndarray, x0: np.ndarray, box, appear_base: float) -> None:
    """在 box 内画 v = noise - x_0 的下采样箭头场(逐像素 2 维真实方向)。

    appear_base 是总体入场进度(0..1),箭头按行优先序错峰浮现;
    长度随模长缩放(强速度 → 长箭头)。
    """
    step = 70
    S = x0.shape[0]
    n = S // step
    vx = (noise - x0)[::step, ::step, 0]
    vy = (noise - x0)[::step, ::step, 1]
    speed = np.sqrt(vx * vx + vy * vy)
    px, py = box
    total_arrows = n * n
    for iy in range(n):
        for ix in range(n):
            appear = smooth((appear_base * total_arrows - (iy * n + ix)) * 0.5)
            if appear <= 0.05:
                continue
            sp = max(speed[iy, ix], 1e-6)
            ux, uy = vx[iy, ix] / sp, vy[iy, ix] / sp
            bx = px + ix * step + step / 2
            by = py + iy * step + step / 2
            length = 26 * (0.55 + 0.45 * min(1.0, sp / 120)) * appear
            ex, ey = bx + ux * length, by + uy * length
            col = mix(PALETTE["bg"], PALETTE["accent"], appear * 0.9)
            draw.line([bx, by, ex, ey], fill=col, width=2)
            ang = math.atan2(ey - by, ex - bx)
            for da in (2.6, -2.6):
                hx = ex + 9 * math.cos(ang + da)
                hy = ey + 9 * math.sin(ang + da)
                draw.line([ex, ey, hx, hy], fill=col, width=2)


@scene("flow", 10.0)
def sc_flow(img, draw, t01: float) -> None:
    """核心概念可视化:x_t=(1-t)x_0+t·noise 的真实混合过程。

    x_0 是 2 通道结构化图案场(通道 0 同心圆波纹,通道 1 放射波纹),噪声是同形
    2 通道高斯场——v = noise - x_0 逐像素即 2 维速度向量,右图箭头是真实方向,
    不是装饰。t 前半程 0→1(淹没),后半程 1→0(找回),画面即公式本身。
    """
    draw_background(img, draw, t01 * 1.1, seed=8)
    title(draw, "一条约定贯穿全仓库:x_t = (1-t)·x_0 + t·noise", smooth(t01 * 10))

    S = 560  # 可视化方形画布边长
    gx, gy = W / 2 - S - 90, 300  # 左图:混合场 x_t
    px = W / 2 + 90  # 右图 x 起点(y 与左图同)

    # --- 干净图 x_0:2 通道图案场(确定性) ---
    rng = np.random.default_rng(20261013)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy = S / 2, S / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    ring_ch = 127 + 128 * np.cos(r / 18)  # 通道 0:同心圆
    ray_ch = 127 + 128 * np.cos(theta * 6 + r / 90)  # 通道 1:6 重放射
    x0 = np.stack([ring_ch, ray_ch], axis=-1).astype(np.float32)  # (S,S,2)
    noise = rng.normal(128, 255 * 0.30, (S, S, 2)).astype(np.float32)

    # --- 前半程加噪 0→1,后半程去噪 1→0 ---
    phase = t01 * 2.0
    t_val = smooth(min(phase, 1.0)) if phase < 1.3 else smooth(2.6 - phase)
    field = (1 - t_val) * x0 + t_val * noise  # (S,S,2):x_t

    # 左图:两通道的均值渲染成灰度场(伪 RGB)
    gray = field.mean(axis=-1, keepdims=True)
    field_rgb = np.repeat(gray, 3, axis=-1)

    k_vis = smooth(t01 * 8)
    if k_vis > 0:
        field_img = Image.fromarray(np.uint8(np.clip(field_rgb, 0, 255))).convert("RGBA")
        mask = Image.new("L", field_img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], 20, fill=int(255 * k_vis))
        img.paste(field_img, (int(gx), int(gy)), mask)

    # 左图标签与 t 值
    k_lbl = smooth(t01 * 8 - 0.2)
    if k_lbl > 0:
        col = mix(PALETTE["bg"], PALETTE["fg"], k_lbl)
        draw.text(
            (gx + S / 2, gy + S + 44),
            "x_t(混合场)",
            font=FONTS.cjk(30, "Bold"),
            fill=col,
            anchor="mm",
        )
        draw.text(
            (gx + S / 2, gy + S + 92),
            f"t = {t_val:.2f}",
            font=FONTS.mono(34, bold=True),
            fill=mix(PALETTE["bg"], PALETTE["accent"], k_lbl),
            anchor="mm",
        )

    # --- 右图:速度场 v = noise - x_0(逐像素 2 维向量,下采样箭头) ---
    k_arrow = smooth(t01 * 8 - 0.45)
    if k_arrow > 0:
        _draw_velocity_arrows(draw, noise, x0, (int(px), int(gy)), k_arrow)
        lab_col = mix(PALETTE["bg"], PALETTE["fg"], k_arrow)
        draw.text(
            (px + S / 2, gy + S + 44),
            "v = noise - x_0(要学的方向)",
            font=FONTS.cjk(30, "Bold"),
            fill=lab_col,
            anchor="mm",
        )

    caption(draw, "模型要学的:给我 (x_t, t),告诉我指向干净样本的方向", smooth(t01 * 8 - 0.9))


@scene("numbers", 8.0)
def sc_numbers(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 1.1, seed=4)
    title(draw, "零件清单(每个数字与代码同源)", smooth(t01 * 10))
    tiles = [
        ("7", "模型适配器"),
        ("10", "训练目标"),
        ("7", "奖励后端"),
        ("8", "CPU 可跑示例"),
        ("94", "测试全绿"),
        ("6", "篇 ADR"),
    ]
    tw_, th_ = 540, 250
    cols = 3
    gap_x, gap_y = 60, 50
    grid_w = cols * tw_ + (cols - 1) * gap_x
    x_start = W / 2 - grid_w / 2
    y_start = 230
    for i, (num, label) in enumerate(tiles):
        kk = stagger(t01, i, 0.06)
        if kk <= 0:
            continue
        row, col = divmod(i, cols)
        x = x_start + col * (tw_ + gap_x)
        y = y_start + row * (th_ + gap_y) + (1 - kk) * 30
        panel(
            draw,
            [x, y, x + tw_, y + th_],
            20,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], PALETTE["line"], kk),
            width=2,
        )
        draw.text(
            (x + tw_ / 2, y + th_ / 2 - 26),
            num,
            font=FONTS.cjk(96, "Black"),
            fill=mix(PALETTE["bg"], PALETTE["accent"], kk),
            anchor="mm",
        )
        draw.text(
            (x + tw_ / 2, y + th_ / 2 + 62),
            label,
            font=FONTS.cjk(32),
            fill=mix(PALETTE["bg"], PALETTE["fg"], kk),
            anchor="mm",
        )
    cap = "文档同步测试守护:README / 能力矩阵 / compose 指标表永不脱离注册表"
    caption(draw, cap, smooth(t01 * 8 - 0.9))


@scene("gates", 7.0)
def sc_gates(img, draw, t01: float) -> None:
    """五道质量门可视化:每道门在提交时强制执行(仓库真实差异化卖点)。"""
    draw_background(img, draw, t01 * 0.9, seed=9)
    title(draw, "每次提交,五道门全绿才放行", smooth(t01 * 10))
    gates = [
        ("ruff check / format", "风格与格式,零放纵"),
        ("check_v2_quality", "AST 门:行数/复杂度/去重"),
        ("check_structure", "依赖方向与组件存在"),
        ("check_docs", "断链/重复/文档预算"),
        ("pytest tests/v2", "95 个数值不变量"),
    ]
    bw, bh = 1000, 88
    y0 = 220
    for i, (name, desc) in enumerate(gates):
        kk = stagger(t01, i, 0.06)
        if kk <= 0:
            continue
        x = W / 2 - bw / 2 + (1 - kk) * 60
        y = y0 + i * (bh + 18)
        panel(
            draw,
            [x, y, x + bw, y + bh],
            14,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], PALETTE["accent2"], kk * 0.85),
            width=2,
        )
        # 通过标记:圆点 + 勾(绿灯 = --mark 色)
        check_col = mix(PALETTE["bg"], PALETTE["accent2"], kk)
        cx, cy = x + 44, y + bh / 2
        draw.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], outline=check_col, width=3)
        draw.line([cx - 8, cy, cx - 2, cy + 7], fill=check_col, width=3)
        draw.line([cx - 2, cy + 7, cx + 9, cy - 8], fill=check_col, width=3)
        draw.text(
            (x + 88, y + bh / 2),
            name,
            font=FONTS.mono(28, bold=True),
            fill=mix(PALETTE["bg"], PALETTE["fg"], kk),
            anchor="lm",
        )
        draw.text(
            (x + 88 + 400, y + bh / 2),
            desc,
            font=FONTS.cjk(25),
            fill=mix(PALETTE["bg"], PALETTE["muted"], kk),
            anchor="lm",
        )
    k_hook = smooth(t01 * 8 - 0.7)
    if k_hook > 0:
        note = "文档同步测试让 README 的能力矩阵永不脱离注册表"
        draw.text(
            (W / 2, y0 + 5 * (bh + 18) + 20),
            note,
            font=FONTS.cjk(30, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["accent"], k_hook),
            anchor="mm",
        )


@scene("baseline", 9.0)
def sc_baseline(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 1.0, seed=5)
    title(draw, "不止是工具箱 —— 扩散后训练的开源学术基线", smooth(t01 * 10))
    cards = [
        ("统一约定", "x_t = (1-t)·x_0 + t·noise\n所有算法共享一套符号约定\n对比实验从第一天就可比"),
        ("结果可核对", "每个算法配契约卡\n数值不变量测试(NaN 拒收、\n教师 detach、KL 走 FP32)"),
        ("论文到零件", "上游雷达盯 8 个活跃项目\nimport SOP + 人工把关点\nLCM / OPD 已完整走通"),
    ]
    cw_, ch_ = 560, 420
    gap = 50
    total = 3 * cw_ + 2 * gap
    x0 = W / 2 - total / 2
    y0 = 210
    for i, (name, body) in enumerate(cards):
        kk = stagger(t01, i, 0.07)
        if kk <= 0:
            continue
        x = x0 + i * (cw_ + gap)
        border = PALETTE["accent2"] if i == 2 else PALETTE["line"]
        panel(
            draw,
            [x, y0, x + cw_, y0 + ch_],
            20,
            fill=mix(PALETTE["bg"], PALETTE["panel"], kk),
            outline=mix(PALETTE["bg"], border, kk),
            width=2,
        )
        draw.text(
            (x + cw_ / 2, y0 + 62),
            name,
            font=FONTS.cjk(38, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["fg"], kk),
            anchor="mm",
        )
        for j, line in enumerate(body.split("\n")):
            draw.text(
                (x + cw_ / 2, y0 + 140 + j * 52),
                line,
                font=FONTS.cjk(27),
                fill=mix(PALETTE["bg"], PALETTE["muted"], kk),
                anchor="mm",
            )
    k_note = smooth(t01 * 8 - 0.8)
    if k_note > 0:
        y_note = y0 + ch_ + 70
        note = "诚实边界:所有 recipe 默认 launch_enabled=false —— 源码可用 ≠ GPU 就绪"
        bw = 1400
        panel(
            draw,
            [W / 2 - bw / 2, y_note - 40, W / 2 + bw / 2, y_note + 40],
            40,
            fill=mix(PALETTE["bg"], (60, 42, 16), k_note * 0.5),
            outline=mix(PALETTE["bg"], PALETTE["warn"], k_note),
            width=2,
        )
        draw.text(
            (W / 2, y_note),
            note,
            font=FONTS.cjk(28, "Bold"),
            fill=mix(PALETTE["bg"], PALETTE["warn"], k_note),
            anchor="mm",
        )


@scene("quickstart", 6.5)
def sc_quickstart(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 0.8, seed=6)
    title(draw, "5 分钟跑通(CPU 即可)", smooth(t01 * 10))
    tw, th = 1360, 560
    x0, y0 = W / 2 - tw / 2, 200
    k = smooth(t01 * 8)
    panel(
        draw,
        [x0, y0, x0 + tw, y0 + th],
        18,
        fill=mix(PALETTE["bg"], PALETTE["panel"], k),
        outline=mix(PALETTE["bg"], PALETTE["line"], k),
        width=2,
    )
    lines = [
        "git clone https://github.com/jackeyloveseven/yoyo",
        'cd yoyo && pip install -e ".[test]"',
        "mmtrain registry list          # 查看全部零件",
        "PYTHONPATH=src python examples/train_sft_edit.py",
    ]
    t_start, per = 0.12, 0.15
    for i, line in enumerate(lines):
        seg = smooth((t01 - t_start - i * per) * 5)
        if seg <= 0:
            continue
        typed = int(len(line) * seg)
        ly = y0 + 110 + i * 96
        prompt = "$ " if i != 3 else "# "
        col = PALETTE["fg"] if i != 3 else PALETTE["accent"]
        draw.text((x0 + 44, ly), prompt + line[:typed], font=FONTS.mono(30), fill=col, anchor="lm")
    caption(draw, "8 个自包含示例:SFT · DPO · 多目标蒸馏 · RL · LCM · OPD", smooth(t01 * 8 - 0.8))


@scene("cta", 5.0)
def sc_cta(img, draw, t01: float) -> None:
    draw_background(img, draw, t01 * 1.6, seed=7)
    glow(
        img,
        draw,
        (W / 2, H / 2 - 80),
        "积木已备好,来拼你的训练",
        FONTS.cjk(76, "Black"),
        PALETTE["fg"],
        with_alpha(PALETTE["accent2"], 110 * scene_alpha(t01)),
        radius=26,
    )
    k2 = smooth(t01 * 8 - 0.3)
    if k2 > 0:
        draw.text(
            (W / 2, H / 2 + 40),
            "YoYo · You Only Yield Once",
            font=FONTS.cjk(34),
            fill=mix(PALETTE["bg"], PALETTE["muted"], k2),
            anchor="mm",
        )
    k3 = smooth(t01 * 8 - 0.55)
    if k3 > 0:
        bw, bh = 780, 84
        box = [W / 2 - bw / 2, H / 2 + 120, W / 2 + bw / 2, H / 2 + 120 + bh]
        panel(
            draw,
            box,
            42,
            fill=mix(PALETTE["bg"], PALETTE["panel"], k3),
            outline=mix(PALETTE["bg"], PALETTE["accent"], k3),
            width=3,
        )
        color = mix(PALETTE["bg"], PALETTE["accent"], k3)
        draw.text(
            (W / 2, H / 2 + 162),
            "github.com/jackeyloveseven/yoyo",
            font=FONTS.mono(34, bold=True),
            fill=color,
            anchor="mm",
        )
    caption(draw, "MIT License · 可组合零件 · 欢迎认领 Roadmap", smooth(t01 * 8 - 0.8))


# ---- 渲染主流程 ----


def scene_table() -> list[tuple[str, int, int, object]]:
    """展开为 (name, 起始帧, 结束帧, fn)。"""
    table = []
    cursor = 0
    for name, seconds, fn in SCENES:
        frames = int(round(seconds * FPS))
        table.append((name, cursor, cursor + frames, fn))
        cursor += frames
    return table


def _scene_t01(row, frame: int) -> float:
    """帧号 → 该分镜内的归一化进度(末帧含 1.0,防除零)。"""
    _, start, end, _ = row
    return (frame - start) / max(1, end - start - 1)


def render_frame(table, frame: int) -> Image.Image:
    """渲染一帧;分镜切换处用 CROSSFADE 帧与前一场景尾帧交叉淡化,无黑场。"""
    idx = 0
    for i, row in enumerate(table):
        if row[1] <= frame < row[2]:
            idx = i
            break
    cur = table[idx]
    start = cur[1]

    CROSSFADE = int(FPS * 0.5)  # 场景边界 0.5s 交叉淡化
    tail_frames = frame - start  # 进入新场景的帧数
    if idx > 0 and tail_frames < CROSSFADE:
        prev_row = table[idx - 1]
        prev_img = render_frame_at(prev_row, prev_row[2] - 1)  # 前场冻结于尾帧
        cur_img = render_frame_at(cur, frame)
        alpha = (tail_frames + 1) / max(1, CROSSFADE + 1)
        return Image.blend(prev_img, cur_img, smooth(alpha))
    return render_frame_at(cur, frame)


def render_frame_at(row, frame: int) -> Image.Image:
    """在给定分镜内渲染绝对帧号(不处理跨场,供 render_frame 与海报调用)。"""
    fn = row[3]
    t01 = _scene_t01(row, frame)
    img = Image.new("RGBA", (W, H), PALETTE["bg"] + (255,))
    draw = ImageDraw.Draw(img)
    fn(img, draw, t01)
    return img


def build_audio(track: Path, table) -> None:
    """程序化配乐:低音 pad(整片) + 分镜切换琶音打点 + 首尾渐变。

    无外部音频文件;正弦波叠加,PCM wav 写盘,encode 阶段由 ffmpeg 转 AAC。
    基调 A(55/110Hz)+ 纯五度;每个分镜起点(除首帧)一个 A 大调琶音短打点。
    """
    import wave

    rate = 44100
    total_frames = sum(end - start for _, start, end, _ in table)
    n_samples = total_frames * rate // FPS  # 音频采样数 = 视频时长 × 采样率
    t = np.arange(n_samples) / rate
    # 和声 pad:A1 + A2 + E3,随时间轻微起伏
    pad = (
        0.30 * np.sin(2 * np.pi * 55 * t)
        + 0.22 * np.sin(2 * np.pi * 110 * t + 0.5 * np.sin(2 * np.pi * 0.1 * t))
        + 0.16 * np.sin(2 * np.pi * 164.8 * t)
    )
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)  # 呼吸感
    # 分镜起点琶音(A4-C#5-E5,各 0.22s,指数衰减)—— start 是帧号,换算成秒
    for _, start, _, _ in table:
        if start == 0:
            continue  # 开头不加打点,pad 从零渐入
        t0 = start / FPS
        for i, freq in enumerate((440.0, 554.4, 659.3)):
            onset = t0 + i * 0.05
            n0 = int(onset * rate)
            n1 = min(n_samples, n0 + int(0.22 * rate))
            if n0 >= n_samples:
                continue
            seg = np.arange(n1 - n0) / rate
            env = np.exp(-seg * 9)
            pad[n0:n1] += 0.16 * np.sin(2 * np.pi * freq * seg) * env
    # 全片首 1.2s 淡入、末 2.5s 淡出
    fade_in = int(1.2 * rate)
    fade_out = int(2.5 * rate)
    env = np.ones(n_samples)
    env[:fade_in] = np.linspace(0.0, 1.0, fade_in)
    env[-fade_out:] = np.linspace(1.0, 0.0, fade_out)
    audio = np.clip(pad * env * 0.9, -0.98, 0.98)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(track), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def encode(mp4: Path, table, limit_frames: int | None = None, audio: Path | None = None) -> None:
    total = sum(end - start for _, start, end, _ in table)
    if limit_frames:
        total = min(total, limit_frames)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
    ]
    if audio is not None:
        cmd += ["-i", str(audio)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio is not None:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for i in range(total):
        img = render_frame(table, i)
        proc.stdin.write(img.convert("RGB").tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{total}", flush=True)
    proc.stdin.close()
    proc.wait()
    suffix = " (+audio)" if audio is not None else ""
    print(f"encoded {mp4} ({total} frames, {total / FPS:.1f}s){suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", help="单帧调试,格式 scene:t01(如 hero:0.8),输出 PNG")
    parser.add_argument("--outdir", type=Path, default=ROOT, help="单帧 PNG 输出目录")
    parser.add_argument("--mp4", type=Path, default=ROOT.parent / "assets" / "promo.mp4")
    parser.add_argument("--poster", type=Path, default=ROOT.parent / "assets" / "promo_poster.png")
    parser.add_argument("--limit", type=int, help="只渲染前 N 帧(冒烟测试)")
    parser.add_argument("--poster-scene", default="numbers:0.72", help="海报取帧(scene:t01)")
    parser.add_argument("--no-audio", action="store_true", help="不合成程序化配乐")
    args = parser.parse_args(argv)

    table = scene_table()
    total_frames = sum(end - start for _, start, end, _ in table)
    print(f"scenes: {[(n, (e - s) / FPS) for n, s, e, _ in table]}")
    print(f"total: {total_frames} frames = {total_frames / FPS:.1f}s")

    if args.frame:
        name, _, t01 = args.frame.partition(":")
        for row in table:
            if row[0] == name:
                t = float(t01) if t01 else 0.5
                img = render_frame(table, row[1] + int(t * (row[2] - row[1] - 1)))
                args.outdir.mkdir(parents=True, exist_ok=True)
                out = args.outdir / f"frame_{name}_{t01.replace('.', '_')}.png"
                img.convert("RGB").save(out)
                print(f"saved {out}")
                return 0
        print(f"unknown scene: {name}", file=sys.stderr)
        return 2

    args.mp4.parent.mkdir(parents=True, exist_ok=True)
    audio = None
    if not args.no_audio:
        import tempfile

        audio = Path(tempfile.mkdtemp(prefix="yoyo_promo_")) / "track.wav"
        build_audio(audio, table)
        print(f"built audio track {audio}")
    encode(args.mp4, table, args.limit, audio=audio)

    name, _, t01 = args.poster_scene.partition(":")
    for row in table:
        if row[0] == name:
            t = float(t01) if t01 else 0.5
            img = render_frame(table, row[1] + int(t * (row[2] - row[1] - 1)))
            args.poster.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(args.poster)
            print(f"saved {args.poster}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
