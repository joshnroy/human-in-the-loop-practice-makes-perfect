"""Same root-PickCube availability classification as EXP-23b, current vocabulary, Della-only."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
rows = json.loads((OUT / "rows.json").read_text())
selected = [r for r in rows if r["src"] == "della23" or (r["src"] == "della" and r["group"] == "gpoff" and r["arm"] == "local_trend-lam3e-4")]
out = []
for r in selected:
    path = OUT / "dec" / ("dec-" + r["tag"] + ".json")
    if not path.exists():
        subprocess.run([sys.executable, str(HERE / "dec.py"), r["run"], r["tag"], str(OUT / "dec")], check=True)
    dec = json.loads(path.read_text())
    counts, dest = Counter(), Counter()
    for e in dec.values():
        if not str(e.get("dispatch", "")).startswith("ask_for_reset"):
            continue
        assert "root" in e and "atoms" in e, (r["run"], e)
        atoms = set(e["atoms"])
        root_pick = any(o.startswith("PickCube(") for o in e["root"])
        if root_pick:
            counts["optional"] += 1
        elif "BinOnGround(bin_0)" not in atoms:
            counts["needed: bin not on ground"] += 1
        elif "CubeAtSide(cube_0,cuboid_barrier,opposite_side)" in atoms:
            counts["needed: far-side cube"] += 1
        elif any(a.startswith("Holding(") for a in atoms):
            counts["needed: holding cube (pick unavailable)"] += 1
        elif not any(a.startswith("PickPlannable(") for a in atoms):
            counts["needed: pick planner found no plan"] += 1
        else:
            counts["needed: other pick precondition"] += 1
        dest[e.get("destination", "unknown")] += 1
    n = sum(counts.values())
    assert n == r["resets"], (r["run"], n, r["resets"])
    rec = dict(cost=r["cost"], seed=r["seed"], src=r["src"], tag=r["tag"], resets_stats=r["resets"], resets_dec=n,
               split=dict(counts), dest=dict(dest), last5=r["last5"], peak=r["peak"], evals=r["evals"],
               refused=r["refused"], picks=r["picks"])
    out.append(rec)
    print("EXP23", r["cost"], r["seed"], f"{counts['optional']}/{n}", dict(counts), dict(dest), flush=True)
assert len(out) == 9, f"Expected9/9 EXP23 runs, got{len(out)}/9"
assert {(r["cost"],r["seed"]) for r in out} == {(c,s) for c in (1,5,20) for s in range(3)}
(OUT / "exp23.json").write_text(json.dumps(out, indent=1))
