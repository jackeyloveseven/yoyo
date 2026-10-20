#!/usr/bin/env python3
"""用本地 piper 中文模型生成配音与 ``timeline.json``(edge-tts 不可达时的离线方案)。

与 temp_voice.py 输出同一套结构:每句一个 ``voice/<id>.wav``,timeline 里带
逐字时刻(按字数比例分配到真实音频时长上)。画面只读 timeline,换配音不用改 film.js。

    python3 site/promo/film/make_piper_voice.py [--model /path/to/model.onnx]

模型下载(约 60MB):

    curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
    curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

from piper import PiperVoice
from piper.config import SynthesisConfig

HERE = Path(__file__).resolve().parent
# 模型不随仓库分发;下载地址见 docstring,或用 PIPER_MODEL 指向本地文件
DEFAULT_MODEL = Path(os.environ.get("PIPER_MODEL", "/tmp/piper/zh_CN-huayan-medium.onnx"))


def char_cost(ch: str) -> float:
    return 0.6 if ch in "，。；：、？！——" else 1.0


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    script = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
    voice = PiperVoice.load(str(args.model))
    # 默认参数偏快、底噪重;放慢 8% 并压低噪声,听感更接近解说而不是朗读器
    config = SynthesisConfig(length_scale=1.08, noise_scale=0.55, noise_w_scale=0.6)
    out_dir = HERE / "voice"
    out_dir.mkdir(exist_ok=True)

    t = 0.15 + float(script.get("pre", 0.0))
    beats = []
    for line in script["lines"]:
        text = line["text"]
        wav_path = out_dir / f"{line['id']}.wav"
        with wave.open(str(wav_path), "wb") as handle:
            voice.synthesize_wav(text, handle, syn_config=config)
        seconds = wav_seconds(wav_path)
        total = sum(char_cost(c) for c in text)
        chars = []
        cursor = t
        for ch in text:
            chars.append([ch, round(cursor, 4)])
            cursor += seconds * char_cost(ch) / total
        beats.append(
            {
                "id": line["id"],
                "text": text,
                "start": round(t, 4),
                "end": round(t + seconds, 4),
                "wav": f"voice/{line['id']}.wav",
                "chars": chars,
            }
        )
        t += seconds + float(line.get("gap", 0.28))

    timeline = {
        "fps": 30,
        "duration": round(t + 0.6, 4),
        "pre": script.get("pre", 0.0),
        "title": None,
        "beats": beats,
        "inserts": [],
    }
    payload = json.dumps(timeline, ensure_ascii=False, indent=1)
    (HERE / "timeline.json").write_text(payload, encoding="utf-8")
    for b in beats:
        print(f"{b['id']:4s} {b['start']:6.2f}–{b['end']:6.2f}  {b['text'][:26]}")
    print(f"总长 {timeline['duration']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
