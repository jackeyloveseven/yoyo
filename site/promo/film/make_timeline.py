#!/usr/bin/env python3
"""在 edge-tts 不可用时,按字数估算一份 ``timeline.json``(结构同 temp_voice.py)。

画面全部挂在 timeline 的 beats/chars 上;等配音恢复后重跑 temp_voice.py,
timeline.json 被真实逐字时刻覆盖,画面自动跟随,不需要改 film.js。

    python3 site/promo/film/make_timeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 中文口播约 4.8 字/秒(Yunxi,+2% 语速);标点按 0.6 个字计
CPS = 4.8


def char_cost(ch: str) -> float:
    return 0.6 if ch in "，。；：、？！——" else 1.0


def main() -> int:
    script = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
    t = 0.15 + float(script.get("pre", 0.0))
    beats = []
    for line in script["lines"]:
        text = line["text"]
        seconds = sum(char_cost(c) for c in text) / CPS
        chars = []
        cursor = t
        total = sum(char_cost(c) for c in text)
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
