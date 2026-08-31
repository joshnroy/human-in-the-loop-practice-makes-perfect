"""Comparable write timestamps for event logs within one process."""

import json
import time
from datetime import datetime, timezone
from typing import Any, ClassVar


class LogTiming:
    """Elapsed time starts when logging is imported; wall time correlates processes."""

    started_at: ClassVar[float] = time.monotonic()

    @staticmethod
    def fields() -> dict[str, str | float]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - LogTiming.started_at,
        }

    @staticmethod
    def encode(*, record: dict[str, Any]) -> str:
        return json.dumps({**record, **LogTiming.fields()}, allow_nan=False) + "\n"
