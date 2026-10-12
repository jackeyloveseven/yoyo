"""Structured append-only events shared by local and WebTrainer execution."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "schema_version": "mmtrainer.event.v2",
            "event": event,
            "monotonic_seconds": time.monotonic(),
            "payload": dict(payload),
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        self.events.append((event, dict(payload)))
