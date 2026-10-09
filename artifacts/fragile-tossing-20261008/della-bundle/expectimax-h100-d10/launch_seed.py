import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parent
arm = sys.argv[1]
if arm not in ("ees", "pddl"):
    raise ValueError("Unsupported arm")
args = json.loads((root / f"{arm}-arguments.json").read_text())
method = args[args.index("--method") + 1]
del args[args.index("--method"):args.index("--method") + 2]
del args[args.index("--env"):args.index("--env") + 2]
command = [sys.executable, "-m", "scripts.run_sweep", "--env", "tossing3d", "--methods", method,
           "--num-seeds", "1", "--max-workers", "1", "--results-root", str(root / "results"),
           "--shared-args", shlex.join(args)]
(root / f"{arm}-command.json").write_text(json.dumps(command, indent=2) + "\n")
subprocess.run(command, check=True)
