from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLogger:
    run_dir: Path
    metrics_path: Path

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        rec = {"ts": time.time(), "event": event, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

