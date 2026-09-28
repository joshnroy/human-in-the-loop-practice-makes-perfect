"""Validated EXP-22c/23c collection; reuse EXP-22b streaming skill scan."""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/josh/Documents/repos/research/hitl-pmp-results")
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
(OUT / "scan").mkdir(exist_ok=True)
(OUT / "dec").mkdir(exist_ok=True)
def validate_ees(run, scan, stats):
    """Prior x22b/eescheck.py accounting check, plus independent practice-event parity."""
    counted = Counter()
    last = {}
    for skill in scan["skills"]:
        if skill["eval"] is None and skill["cycle"] >= 0:
            counted[(skill["cycle"], skill["name"])] += 1
            last[skill["cycle"]] = skill["name"]
    if not (run / "practice_events.jsonl").exists():
        issues = []
        terminal_unobserved = []
        omissions = Counter()
        for cycle, outcomes in enumerate(stats["practice_outcomes_per_cycle"]):
            capped = cycle < len(stats["practice_session_ends"]) and stats["practice_session_ends"][cycle]["reason"] == "session_action_cap"
            delta = {name: counted[(cycle,name)] - outcomes.get(name, {}).get("num_attempts", 0) for name in ("PickCube", "MoveToTossLocationAndToss")}
            total = sum(delta.values())
            valid = all(v == 0 for v in delta.values()) if not capped else (total in (0,1) and all(v in (0,1) for v in delta.values()) and (total == 0 or delta.get(last.get(cycle),0) == 1))
            if not valid:
                issues.append([cycle,capped,delta,last.get(cycle)])
            if capped and total == 0:
                terminal_unobserved.append(cycle)
            omissions.update(delta)
        return {"independent_events_available": False, "practice_event_parity": None, "cap_accounting_matches": None, "cap_compatibility_valid": not issues, "stats_omitted_capped_actions": dict(omissions), "terminal_action_unobserved_cycles": terminal_unobserved, "mismatches": issues}
    events = Counter()
    for line in (run / "practice_events.jsonl").open():
        event = json.loads(line)
        match = re.match(r"\d+\s+(?:random: |informed: )?([A-Za-z]+)\(", event.get("entry", ""))
        if match:
            events[(event["cycle"], match.group(1))] += 1
        if re.match(r"\d+\s", event.get("entry", "")):
            last[event["cycle"]] = match.group(1) if match else "RESET"
    names = ("PickCube", "MoveToTossLocationAndToss")
    parity = all(counted[k] == events[k] for k in set(counted) | set(events) if k[1] in names)
    omissions = Counter()
    mismatches = []
    for cycle, outcomes in enumerate(stats["practice_outcomes_per_cycle"]):
        capped = cycle < len(stats["practice_session_ends"]) and stats["practice_session_ends"][cycle]["reason"] == "session_action_cap"
        for name in names:
            delta = int(capped and last.get(cycle) == name)
            omissions[name] += delta
            expected = counted[(cycle,name)] - delta
            actual = outcomes.get(name, {}).get("num_attempts", 0)
            if expected != actual:
                mismatches.append([cycle,name,expected,actual])
    return {"independent_events_available": True, "practice_event_parity": parity, "cap_accounting_matches": not mismatches, "stats_omitted_capped_actions": dict(omissions), "mismatches": mismatches}

rows = []
validation = []
for src, pattern in [("ws", "exp22c/*/results/*/*/*/stats.json"), ("della", "exp22c-della/*/results/*/*/*/stats.json"), ("della23", "exp23c-della/results/*/*/*/stats.json")]:
    for f in sorted(ROOT.glob(pattern)):
        run = f.parent
        rel = run.relative_to(ROOT)
        tag = str(rel).replace("/results/", "__").replace("/", "_")
        st = json.loads(f.read_text())
        timing_file = run / "timing.json"
        tm = json.loads(timing_file.read_text()) if timing_file.exists() else {}
        cfgfile = json.loads((run / "config_snapshot.json").read_text())
        cfg = cfgfile["args"]
        cycles = int(cfg["num_cycles"])
        assert cfgfile["git_commit"].startswith("76732331"), (str(run), cfgfile["git_commit"])
        assert cfgfile["kindergarden_commit"].startswith("f0d554b"), (str(run), cfgfile["kindergarden_commit"])
        assert cfgfile["kinder_models_commit"].startswith("427ad6c"), (str(run), cfgfile["kinder_models_commit"])
        good = len(st["evaluations"]) == cycles + 1 and len(st["practice_session_ends"]) == cycles
        if not tm or tm.get("returncode") != 0 or tm.get("succeeded") is False:
            good = False
        if not good:
            validation.append({"run": str(run), "complete": False, "evaluations": len(st["evaluations"]), "timing": tm})
            continue
        scan = OUT / "scan" / (tag + ".json")
        if not scan.exists() or scan.stat().st_mtime < f.stat().st_mtime:
            subprocess.run([sys.executable, str(HERE / "scan.py"), str(run), str(scan)], check=True)
        scan_data = json.loads(scan.read_text())
        sc = scan_data["summary"]
        ees_accounting = validate_ees(run, scan_data, st) if run.parent.name == "ees" else None
        counts_valid = ((ees_accounting["practice_event_parity"] and ees_accounting["cap_accounting_matches"]) if ees_accounting["independent_events_available"] else ees_accounting["cap_compatibility_valid"]) if ees_accounting else (sc["pick_count_matches_stats"] and sc["toss_count_matches_stats"])
        checks = sc["eval_label_mismatch"] == 0 and counts_valid
        validation.append({"run": str(run), "complete": True, "scan_valid": checks, "summary": sc, "ees_accounting": ees_accounting})
        if not checks:
            raise RuntimeError("Practice/eval scan validation failed: " + str(run))
        method = run.parent.name
        arm = run.parent.parent.name
        group = rel.parts[1] if src != "della23" else "gpoff"
        ev = [e[1] for e in st["evaluations"]]
        assert all(e[2] == 10 for e in st["evaluations"]), str(run)
        model = cfg["pomdp_competence_model"] if method == "pomdp" else "ees"
        knob = float(cfg["pomdp_linear_cost_lambda"] if method == "pomdp" else cfg["human_reset_practice_cost"])
        resets = st["num_human_interventions_recorded"]
        row = dict(src=src, group=group, arm=arm, method=method, model=model, seed=int(run.name),
                   reset=str(cfg.get("human_reset")).lower() == "true", gp=float(cfg.get("goal_pursuit_horizon") or 0) > 0,
                   cost=float(cfg["human_reset_practice_cost"]), knob=knob, evals=ev, last5=sum(ev[-5:])/5,
                   last5_solved=sum(ev[-5:]), last5_total=50, peak=max(ev), n_evals=len(ev), resets=resets,
                   human_cost=st["summed_human_cost_recorded"], practice_actions=sc["practice_actions"],
                   tosses=sc["practice_tosses"], picks=sc["practice_picks"], refused=sc["practice_refused"],
                   eval_picks=sc["eval_picks"], eval_refused=sc["eval_refused"], longest=sc["longest_streak"],
                   elapsed=tm.get("elapsed_seconds"), start=tm.get("start_time"), end=tm.get("end_time"),
                   run=str(run), tag=tag, scan_valid=checks, ees_accounting=ees_accounting, commit=cfgfile.get("git_commit"), kinder_commit=cfgfile.get("kindergarden_commit"), kinder_models_commit=cfgfile.get("kinder_models_commit"))
        rows.append(row)
        print("COLLECT", src, group, arm, run.name, f"{row['last5']}/10", f"{row['refused']}/{row['picks']}", flush=True)
(OUT / "rows.json").write_text(json.dumps(rows, indent=1))
(OUT / "validation.json").write_text(json.dumps(validation, indent=1))
print("DONE", dict(Counter(r["src"] for r in rows)), flush=True)
