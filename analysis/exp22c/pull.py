"""Final-only Della pull, verified local sizes, no remote deletion."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path("/home/josh/Documents/repos/research/hitl-pmp-results")
REMOTE="/scratch/gpfs/TSILVER/jr2860/results"
GROUPS=("gpoff","gpon","noreset","ees-gpoff","ees-gpon","ees-noreset")
# Called only after the parent has verified all63/63 jobs COMPLETED0:0.
for group in GROUPS:
    subprocess.run([sys.executable,str(ROOT/"della/della.py"),"pull","exp22c-"+group,"--light","--dest",str(ROOT/"exp22c-della"/group/"results")],check=True)
subprocess.run([sys.executable,str(ROOT/"della/della.py"),"pull","exp23c","--dest",str(ROOT/"exp23c-della/results")],check=True)
subprocess.run(["rsync","-a","--ignore-existing","--include=*/","--include=pomdp_decisions.jsonl","--exclude=*","-e","ssh -o BatchMode=yes",f"della:{REMOTE}/exp22c-gpoff/local_trend-lam3e-4/",str(ROOT/"exp22c-della/gpoff/results/local_trend-lam3e-4")+"/"],check=True)
# rsync transfers are checksummed in transit; a checksum dry-run below detects any mismatches.
checks=[]
for group in GROUPS:
    remote=f"della:{REMOTE}/exp22c-{group}/"
    local=str(ROOT/"exp22c-della"/group/"results")+"/"
    p=subprocess.run(["rsync","-anrc","--itemize-changes","--exclude=*decisions*.jsonl","-e","ssh -o BatchMode=yes",remote,local],capture_output=True,text=True,check=True)
    changes=p.stdout.strip()
    checks.append(dict(group=group,checksum_changes=changes))
    assert not changes, (group,changes)
for exp,dest,extra in [("exp23c",ROOT/"exp23c-della/results",[]),("exp22c-gpoff/local_trend-lam3e-4",ROOT/"exp22c-della/gpoff/results/local_trend-lam3e-4",["--include=*/","--include=pomdp_decisions.jsonl","--exclude=*"])]:
    p=subprocess.run(["rsync","-anrc","--itemize-changes",*extra,"-e","ssh -o BatchMode=yes",f"della:{REMOTE}/{exp}/",str(dest)+"/"],capture_output=True,text=True,check=True)
    checks.append(dict(group=exp,checksum_changes=p.stdout.strip()))
    assert not p.stdout.strip(),(exp,p.stdout)
(HERE/"data").mkdir(exist_ok=True)
(HERE/"data/pull-verification.json").write_text(json.dumps(checks,indent=1))
print("PULL VERIFIED; remote logs retained",flush=True)
