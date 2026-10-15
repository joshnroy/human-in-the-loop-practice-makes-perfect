"""Entry point copied into the disconnected sandbox to call one generated function.

Stdlib and numpy only: it runs inside the execution image, never imported by the host
library. Reads `request.json` beside itself, imports the frozen module, calls one
function and prints one result line on stdout.
"""

import importlib.abc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

RESULT_PREFIX = "HITL_CODED_SAMPLER_RESULT "


class ImportGuard(importlib.abc.MetaPathFinder):
    """Refuses an import *made by generated code* of a top-level module outside the
    allowed set and the standard library. Imports numpy and friends make internally
    are theirs, not the generated code's, and pass."""

    def __init__(self, *, allowed: set[str], code_dir: Path) -> None:
        self.allowed = allowed
        self.code_dir = str(code_dir)

    def importer_is_generated(self) -> bool:
        frame: Any = sys._getframe(2)
        while frame is not None and (
            frame.f_code.co_filename.startswith("<frozen") or frame.f_code.co_filename == __file__
        ):
            frame = frame.f_back
        return frame is not None and frame.f_code.co_filename.startswith(self.code_dir)

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:  # noqa: PLR0917
        top = fullname.split(".")[0]
        if top in self.allowed or top in sys.stdlib_module_names:
            return
        if self.importer_is_generated():
            raise ImportError(f"import of {fullname!r} is not allowed in generated code")


class CodedSamplerWorker:
    @staticmethod
    def jsonable(*, value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list | tuple):
            return [CodedSamplerWorker.jsonable(value=entry) for entry in value]
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value

    @staticmethod
    def main() -> None:
        here = Path(__file__).resolve().parent
        request = json.loads((here / "request.json").read_text())
        code_dir = Path(request["code_dir"])
        sys.path.insert(0, str(code_dir))
        local = {entry.stem for entry in code_dir.iterdir()}
        guard = ImportGuard(allowed={*request["allowed_modules"], *local}, code_dir=code_dir)
        sys.meta_path.insert(0, guard)
        try:
            path = code_dir / request["module"]
            spec = importlib.util.spec_from_file_location(f"generated_{request['skill']}", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            function = getattr(module, request["function"])
            payload = request["payload"]
            if request["function"] == "score":
                value = function(payload["sampler_inputs"])
            else:
                import numpy as np  # noqa: PLC0415 -- only the sampler mode needs it

                rng = np.random.default_rng(payload["seed"])
                value = function(payload["state"], payload["grounding"], rng)
            result = {"ok": True, "value": CodedSamplerWorker.jsonable(value=value)}
        except BaseException as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"[:4000]}
        sys.stdout.write(RESULT_PREFIX + json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    CodedSamplerWorker.main()
