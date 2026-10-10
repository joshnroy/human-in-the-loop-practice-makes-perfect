"""One serialized dollar budget shared by coding and held-out classifiers."""

import fcntl
import json
import math
from contextlib import contextmanager
from pathlib import Path

from hitl_pmp.step_protocol import StepFiles


class SharedBudget:
    def __init__(self, *, path: Path, limit: float, session: str):
        self.path, self.limit, self.session = path, limit, session
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        with self.path.with_suffix(".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def data(self):
        return (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"limit": self.limit, "sessions": {}}
        )

    @property
    def spent(self):
        return sum(self.data()["sessions"].values())

    @property
    def remaining(self):
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self):
        return self.remaining <= 0

    def record(self, *, cumulative):
        if cumulative is None or not math.isfinite(cumulative):
            raise ValueError("Missing/nonfinite model cost")
        data = self.data()
        if cumulative < data["sessions"].get(self.session, 0):
            raise ValueError("Resumed session cost decreased")
        data["sessions"][self.session] = cumulative
        data["spent"] = sum(data["sessions"].values())
        StepFiles.json(path=self.path, value=data)
