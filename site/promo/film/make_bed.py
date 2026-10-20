#!/usr/bin/env python3
"""生成片子的底乐 ``site/promo/film/bed.wav``(时长读 timeline.json)。

平稳的电钢琶音 + 低音脉冲,Am–F–C–G 循环;人声出现时由 mix.py 压低。
没有采样素材,全部用正弦叠加程序化合成——与旧宣传片同一条思路,但时长跟着
当前时间线走,不再和分镜表绑死。

    python3 site/promo/film/make_bed.py
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RATE = 44100
BPM = 100
BEAT = 60.0 / BPM
BAR = BEAT * 4
CHORDS = {
    "Am": [220.00, 261.63, 329.63],
    "F": [174.61, 220.00, 261.63],
    "C": [130.81, 164.81, 196.00],
    "G": [196.00, 246.94, 293.66],
}
ORDER = ["Am", "F", "C", "G"]


def pluck(buf: np.ndarray, t0: float, freq: float, dur: float, amp: float) -> None:
    """电钢的一击:基频 + 八度/十二度泛音,8ms 起音,指数衰减。"""
    n0 = int(t0 * RATE)
    n1 = min(len(buf), n0 + int(dur * RATE))
    if n0 >= len(buf):
        return
    t = np.arange(n1 - n0) / RATE
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 3.4)
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.34 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.16 * np.sin(2 * np.pi * freq * 3 * t)
    )
    buf[n0:n1] += amp * env * wave


def sub(buf: np.ndarray, t0: float, freq: float, dur: float, amp: float) -> None:
    n0 = int(t0 * RATE)
    n1 = min(len(buf), n0 + int(dur * RATE))
    if n0 >= len(buf):
        return
    t = np.arange(n1 - n0) / RATE
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 6.0)
    buf[n0:n1] += amp * env * np.sin(2 * np.pi * freq * t)


def build(duration: float) -> np.ndarray:
    buf = np.zeros(int(duration * RATE) + RATE // 2)
    bar = 0
    t = 0.0
    while t < duration:
        name = ORDER[bar % len(ORDER)]
        notes = CHORDS[name]
        # 琶音:每拍一个音,第三拍加高八度
        for beat in range(4):
            freq = notes[beat % len(notes)]
            pluck(buf, t + beat * BEAT, freq, BEAT * 1.6, 0.16)
            if beat == 2:
                pluck(buf, t + beat * BEAT, freq * 2, BEAT * 1.2, 0.08)
        sub(buf, t, notes[0] / 2, BAR, 0.10)
        t += BAR
        bar += 1
    # 首尾渐弱 + tanh 软限幅
    n = int(duration * RATE)
    buf = buf[:n]
    fade = int(1.2 * RATE)
    buf[:fade] *= np.linspace(0, 1, fade)
    buf[-fade:] *= np.linspace(1, 0, fade)
    buf = np.tanh(buf * 1.2) * 0.55
    return buf


def main() -> int:
    timeline = json.loads((HERE / "timeline.json").read_text(encoding="utf-8"))
    duration = float(timeline["duration"])
    pcm = (np.clip(build(duration), -1, 1) * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    out = HERE / "bed.wav"
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(stereo.tobytes())
    print(f"saved {out} {duration:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
