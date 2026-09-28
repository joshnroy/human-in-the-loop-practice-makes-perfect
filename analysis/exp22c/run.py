"""Run after verified Della pull; rerun after additional workstation completions."""
import subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
for script in ("collect.py","exp23.py","report.py"):
    subprocess.run([sys.executable,str(here/script)],check=True)
print("Analysis complete",flush=True)
