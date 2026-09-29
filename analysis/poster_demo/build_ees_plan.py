import collections,json,re
from pathlib import Path
ROOT=Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c/ees-gpoff/results/ees-lamcost5/ees/0")
PLAN=Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c/paired-gallery/plans/exp22c_ees-gpoff__ees-lamcost5_ees_0.json")
plan=json.loads(PLAN.read_text())
events=collections.defaultdict(list)
for line in (ROOT/"practice_events.jsonl").open():
 e=json.loads(line)
 if "entry" in e and re.match(r"^\d+  ",e["entry"]) and "interaction complete" not in e["entry"].lower(): events[e["cycle"]].append(e)
resets=next(r["practice_resets"] for r in json.loads(PLAN.parent.parent.joinpath("manifest.json").read_text())["runs"] if r["id"]==PLAN.stem)
for c,cycle in enumerate(plan["practice"]):
 robot=iter(cycle["actions"]);ordered=[]
 for event in events[c]:
  if "HUMAN RESET" in event["entry"]:
   ordered.append(dict(skill="ask_for_reset_cube_bin_only",label="Human reset",block=None,ticks=0,success=True))
  else:
   action=next(robot)
   assert action["skill"] in event["entry"],(c,action,event)
   ordered.append(action)
 ordered.extend(robot)
 missing=resets[c]-sum(a["block"] is None for a in ordered)
 assert missing in (0,1),(c,missing)
 if missing:ordered.append(dict(skill="ask_for_reset_cube_bin_only",label="Human reset",block=None,ticks=0,success=True))
 cycle["actions"]=ordered
 for i,a in enumerate(ordered):a["decision"]=i
assert sum(a["block"] is None for c in plan["practice"][24:27] for a in c["actions"])==18
Path(__file__).with_name("ees-plan.json").write_text(json.dumps(plan))
print("EES event ordering and reset totals reconciled50/50cycles")
