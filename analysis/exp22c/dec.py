"""dec.py <run_dir> <tag>: stream pomdp_decisions.jsonl line by line; per decision keep
compact atoms, chosen action, root action values, pick-branch probabilities, outcome."""
import json, re, sys
from collections import Counter

RUN, TAG = sys.argv[1], sys.argv[2]
OUT = sys.argv[3].rstrip("/") + "/"
objpat = re.compile(r"Object\(name='([^']+)'")


def compact(a):
    m = re.match(r"predicate=Predicate\(name='([A-Za-z]+)'", a)
    name = m.group(1) if m else a[:40]
    objs = a.split(" objects=", 1)[1] if " objects=" in a else ""
    return name + "(" + ",".join(objpat.findall(objs)) + ")"


def an(a):
    if isinstance(a, dict):
        return a["skill"]["name"] + "(" + ",".join(o["name"] for o in a["objects"]) + ")"
    return str(a)[:80]


dec = {}
with open(RUN + "/pomdp_decisions.jsonl") as f:
    for line in f:
        m = re.search(r'"event": "([a-z_]+)"', line[:120])
        ev = m.group(1) if m else None
        if ev not in ("decision", "outcome", "dispatch"):
            continue
        d = json.loads(line)
        e = dec.setdefault(d["decision"], {"cycle": d["cycle"]})
        if ev == "decision":
            e["action"] = an(d["action"])
            e["destination"] = d["action"]["objects"][-1]["name"] if isinstance(d["action"], dict) and d["action"].get("objects") else None
            e["value"] = d["value"]
            e["atoms"] = sorted(compact(a) for a in d["atoms"])
            e["competences"] = d.get("competences")
            e["costs"] = d.get("estimated_costs")
            e["remaining"] = d.get("remaining_practice_actions")
            root = {}
            for s in d.get("search", []):
                if s.get("node") != 0:
                    continue
                if s["event"] == "action_value":
                    root.setdefault(an(s["action"]), {})["Q"] = s["value"]
                elif s["event"] == "branch":
                    root.setdefault(an(s["action"]), {}).setdefault("branches", []).append(
                        [round(s["probability"], 5), s["successor_value"], s["sampled_cost"]])
                elif s["event"] == "stop_value":
                    root["STOP"] = {"Q": s["value"]}
                elif s["event"] == "choice":
                    e["reason"] = s.get("reason")
            e["root"] = root
        elif ev == "dispatch":
            e["dispatch"] = d["skill"]
        else:
            e["success"] = d["success"]
        del d
json.dump(dec, open(OUT + f"dec-{TAG}.json", "w"))
c = Counter((e.get("dispatch"), e.get("success"), "InBin(cube_0,bin_0)" in e.get("atoms", [])) for e in dec.values())
print(TAG, len(dec), c.most_common())
