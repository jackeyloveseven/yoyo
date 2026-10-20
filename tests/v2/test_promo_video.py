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


def test_formulas_are_typeset_not_literal_ascii() -> None:
    """公式必须经 mathtext 排版上镜,不得以 x_t / x_0 / noise 这类字面出现在文案里。

    TeX 源码误上镜(如 `$x_t$` 未被剥成公式体)是肉眼很难发现的缺陷,
    这里断言 math_image 会显式拒绝含 $ 的公式体,并守住历史踩坑点。
    """
    import numpy as np

    mod = _renderer_module()
    # ① math_image 渲染真公式成功(透明底,有墨迹)
    glyph = mod.math_image(r"x_t = (1-t)\,x_0 + t\,\varepsilon", 40, (236, 239, 244))
    arr = np.asarray(glyph)
    assert arr.shape[2] == 4, "math glyph must be RGBA (transparent background)"
    assert arr[..., 3].min() == 0, "math glyph background must be transparent"
    assert (arr[..., 3] > 128).sum() > 50, "math glyph has no ink"
    # ② 含 $ 的公式体必须被拒绝(否则会静默渲染成字面文本上镜)
    import pytest

    with pytest.raises(ValueError, match="不应含"):
        mod.math_image(r"$x_t$", 40, (236, 239, 244))


def test_no_latex_source_leaks_into_on_screen_copy() -> None:
    """上镜文案里不得出现未剥净的 $...$ LaTeX 源码(会原样上镜)。

    历史踩坑:baseline 卡片 body 用 `$x_t=...$` 包裹,只剥了前导 $ 而漏了尾随 $,
    于是 `$x_t = (1-t)\\,x_0...` 整串作为字面文本渲染上镜。
    判据:凡形如 `$变量`(紧跟字母/反斜杠,即 LaTeX 开头)的文案,其 $ 必须成对;
    shell 提示符 `$ ` 与 `$ mmtrain ...`(后跟空格/命令)不在此列。
    """
    on_screen = _on_screen_text()
    for match in re.finditer(r'"([^"\n]*\$[a-zA-Z\\][^"\n]*)"', on_screen):
        literal = match.group(1)
        assert literal.count("$") % 2 == 0, f"未剥净的 LaTeX 会上镜: {literal!r}"


def test_video_and_poster_artifacts_exist() -> None:
    """成品必须入库(GitHub Pages 从 git 提供 site/,不入库即上线 404)。"""
    assets = PROMO_DIR.parent / "assets"
    for name in ("promo.mp4", "promo_poster.png"):
        path = assets / name
        assert path.exists(), f"{name} missing — Pages will 404"
        assert path.stat().st_size > 10_000, f"{name} suspiciously small"


def _mp4_duration_seconds(path: Path) -> float:
    """从 MP4 的 mvhd box 读时长(纯字节解析,不依赖 ffprobe)。"""
    blob = path.read_bytes()
    idx = blob.find(b"mvhd")
    assert idx > 0, "promo.mp4 has no mvhd box"
    assert blob[idx + 4] == 0, "mvhd version 1 unsupported"
    timescale = int.from_bytes(blob[idx + 16 : idx + 20], "big")
    duration = int.from_bytes(blob[idx + 20 : idx + 24], "big")
    return duration / timescale


def test_rendered_video_duration_matches_scene_table() -> None:
    """入库的 MP4 时长必须等于分镜表总和——防止"改了分镜表但忘记重渲"。

    源码侧的时长断言只能证明表被正确读取,证明不了线上这支片子真的按新表渲染过。
    成品是 Pages 直接提供的文件,它和源码分镜表脱节时,页面文案会与画面矛盾。
    """
    mod = _renderer_module()
    table = mod.scene_table()
    expected = sum(end - start for _, start, end, _ in table) / mod.FPS
    actual = _mp4_duration_seconds(PROMO_DIR.parent / "assets" / "promo.mp4")
    assert abs(actual - expected) < 1.0, (
        f"promo.mp4 时长 {actual:.1f}s != 分镜表 {expected:.1f}s——分镜改了但没重渲?"
    )


def _renderer_module():
    """导入渲染器模块(模块级只有常量与懒加载缓存,import 安全)。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_promo", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_build_audio_has_musical_content_and_correct_length() -> None:
    """配乐时长=视频时长,全片无死寂,首尾渐弱,且含预期和弦的基频能量。

    旧的"分镜起点有琶音峰"判据在连续琶音下不再成立(每一拍都有音),
    改为验证音乐本身的属性——这才是 build_audio 真正该保证的东西。
    """
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
    assert abs(n_frames / rate - video_seconds) < 1.0 / mod.FPS, (
        f"audio {n_frames / rate:.2f}s != video {video_seconds:.2f}s"
    )
    # ② 无死寂:全片按 0.5s 切片,每片 RMS 都高于底噪(排除首尾渐弱区)
    body = pcm[int(2.0 * rate) : int((video_seconds - 3.5) * rate)]
    hop = int(0.5 * rate)
    rms = [float(np.sqrt(np.mean(body[i : i + hop] ** 2))) for i in range(0, len(body) - hop, hop)]
    assert min(rms) > 0.01, f"dead air found: min window RMS {min(rms):.4f}"
    assert max(rms) < 0.95, "audio clipping-suspect (flat-topped?)"
    # ③ 首尾渐弱:起始与结尾 0.2s 的振幅显著低于片中
    head = float(np.abs(pcm[: int(0.2 * rate)]).max())
    tail = float(np.abs(pcm[-int(0.2 * rate) :]).max())
    mid = float(
        np.abs(pcm[int(video_seconds / 2 * rate) - hop : int(video_seconds / 2 * rate)]).max()
    )
    assert head < 0.2 * mid, f"no fade-in (head {head:.3f} vs mid {mid:.3f})"
    assert tail < 0.2 * mid, f"no fade-out (tail {tail:.3f} vs mid {mid:.3f})"
    # ④ 和声内容:频谱在 A(110Hz)附近应有能量(Am 和弦根音,全片出现多次)
    spec = np.abs(np.fft.rfft(pcm[: int(4.0 * rate)] * np.hanning(int(4.0 * rate))))
    freqs = np.fft.rfftfreq(int(4.0 * rate), 1.0 / rate)

    def band_energy(lo: float, hi: float) -> float:
        return float(spec[(freqs >= lo) & (freqs <= hi)].max())

    assert band_energy(105, 115) > band_energy(300, 320) * 1.5, (
        "no A-root harmonic energy in the first chord (Am)"
    )
