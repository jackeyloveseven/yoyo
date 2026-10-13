"""宣传视频渲染器的分镜表与营销红线自检(静态断言 + build_audio 数值自检,无需 ffmpeg)。"""

from __future__ import annotations

import re
from pathlib import Path

PROMO_DIR = Path(__file__).resolve().parents[2] / "site" / "promo"
RENDERER = PROMO_DIR / "render_promo.py"


def _renderer_source() -> str:
    assert RENDERER.exists(), f"renderer missing: {RENDERER}"
    return RENDERER.read_text(encoding="utf-8")


def test_scene_table_durations_and_total() -> None:
    """分镜时长合理、总时长落在宣传片区间(无需加载 PIL 渲染)。"""
    text = _renderer_source()
    durations = [float(m) for m in re.findall(r"@scene\(\s*\"[a-z_]+\"\s*,\s*([\d.]+)\s*\)", text)]
    assert durations, "no @scene decorator found — renderer lost its shot list?"
    assert all(2.0 <= d <= 10.0 for d in durations), f"scene out of range: {durations}"
    total = sum(durations)
    assert 40.0 <= total <= 90.0, f"promo total {total}s outside 40..90s window"
    assert len(durations) >= 6, f"only {len(durations)} scenes"


def test_shot_list_covers_the_pitch() -> None:
    """分镜必须覆盖品牌、痛点、零件、流匹配概念、compose、数字、质量门、基线、上手、CTA。"""
    text = _renderer_source()
    scenes = re.findall(r"@scene\(\s*\"([a-z_]+)\"\s*,\s*[\d.]+\s*\)", text)
    expected = {
        "hero",
        "pain",
        "parts",
        "flow",
        "compose",
        "numbers",
        "gates",
        "baseline",
        "quickstart",
        "cta",
    }
    assert expected <= set(scenes), f"missing scenes: {sorted(expected - set(scenes))}"


def _on_screen_text() -> str:
    """只取分镜函数体(真实上镜内容),排除 docstring/注释里的红线说明文字。"""
    text = _renderer_source()
    body = text.split("SCENES: list", 1)[-1]
    # 去掉注释与 docstring 前的文件头
    body = body.split("@scene", 1)[-1]
    return "@scene" + body


def test_marketing_redlines_enforced_in_video() -> None:
    """宣传物料红线:FireRed 不上镜、Qwen 主打 2.x、AWM/RAM 不宣传、诚实边界可见。"""
    text = _on_screen_text()
    assert "firered" not in text.lower(), "FireRed must not appear in promo material"
    assert "AWM" not in text and "RAM" not in text, "AWM/RAM must not be advertised"
    assert "qwen_image_edit_2x" in text, "promo must spotlight Qwen-Image-Edit 2.x"
    # 诚实边界必须是可见文案(至少一处)
    assert "GPU 就绪" in text or "launch_enabled" in text


def test_video_and_poster_artifacts_exist() -> None:
    """成品必须入库(GitHub Pages 从 git 提供 site/,不入库即上线 404)。"""
    assets = PROMO_DIR.parent / "assets"
    for name in ("promo.mp4", "promo_poster.png"):
        path = assets / name
        assert path.exists(), f"{name} missing — Pages will 404"
        assert path.stat().st_size > 10_000, f"{name} suspiciously small"


def _renderer_module():
    """导入渲染器模块(模块级只有常量与懒加载缓存,import 安全)。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_promo", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_build_audio_matches_video_length_and_marks_scene_onsets() -> None:
    """配乐时长=视频时长,且每个分镜起点附近有琶音能量峰(需 numpy)。"""
    import tempfile
    import wave

    import numpy as np

    mod = _renderer_module()
    table = mod.scene_table()
    with tempfile.TemporaryDirectory() as tmp:
        track = Path(tmp) / "promo_audio_test.wav"
        mod.build_audio(track, table)
        with wave.open(str(track), "rb") as handle:
            rate = handle.getframerate()
            n_frames = handle.getnframes()
            pcm = np.frombuffer(handle.readframes(n_frames), dtype=np.int16).astype(np.float64)
            pcm /= 32768.0
    # ① 时长一致:音频采样数 / 采样率 = 视频帧数 / FPS(允许 1 帧 rounding)
    video_seconds = sum(end - start for _, start, end, _ in table) / mod.FPS
    audio_seconds = n_frames / rate
    assert abs(audio_seconds - video_seconds) < 1.0 / mod.FPS, (
        f"audio {audio_seconds:.2f}s != video {video_seconds:.2f}s"
    )
    # ② 每个分镜起点(除 hero,开头是 pad 淡入)在 ±0.15s 内有琶音能量峰。
    # 不能比总振幅——pad 本身峰值 ≈0.77,琶音只有 ±0.16,同域打架(实测比率仅 1.15)。
    # 琶音是 440/554/659Hz,pad 是 55/110/164.8Hz,频率域不重叠,
    # 故用一阶差分(高频能量)判据:实测各起点比率 2.9~14.2。
    high_band = np.abs(np.diff(pcm))
    win = int(0.15 * rate)
    for name, start, _, _ in table:
        if start == 0:
            continue
        t0 = int(start / mod.FPS * rate)
        near = float(high_band[t0 - win : t0 + win].max())
        before = float(high_band[t0 - 3 * win : t0 - win].max())
        assert near > 0.01 and near > 1.5 * before, f"no onset peak at scene {name}"
