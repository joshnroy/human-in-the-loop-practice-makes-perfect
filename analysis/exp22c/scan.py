"""scan.py <run_dir> <out_json>: stream one run's state log (r22/scan.py method).
Index skills with tick byte offsets + a content hash (timestamps stripped) for Della diffs,
label eval vs practice by matching each eval episode's step count in episode_traces.jsonl to
the skill records just before its last timestamp, assign practice skills to cycles, count
refused practice picks (PickCube with 0 ticks) and streaks, and dump frozen state per streak."""
import bisect, hashlib, json, math, sys
from collections import Counter
from datetime import datetime

RUN, OUT = sys.argv[1], sys.argv[2]


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def tilt(q):
    qw, qx, qy, qz = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))


skills = []
h = None
with open(RUN + "/tossing3d_state_log.jsonl", "rb") as f:
    off = 0
    for raw in f:
        lo = off
        off += len(raw)
        if raw.startswith(b'{"kind": "tick"'):
            s = skills[-1]
            s["ticks"] += 1
            if s["first"] is None:
                s["first"] = lo
            s["last"] = lo
            h.update(raw.split(b', "timestamp"')[0])
            continue
        d = json.loads(raw)
        if d.get("kind") == "skill":
            if skills:
                skills[-1]["hash"] = h.hexdigest()[:16]
            h = hashlib.md5(json.dumps([d["name"], d["objects"], d["params"], d.get("toss_direction_deg")]).encode())
            skills.append({"name": d["name"], "objects": d["objects"], "params": d["params"],
                           "t": ts(d["timestamp"]), "ticks": 0, "first": None, "last": None, "off": lo})
if skills:
    skills[-1]["hash"] = h.hexdigest()[:16]
times = [s["t"] for s in skills]
eps = {}
with open(RUN + "/episode_traces.jsonl") as f:
    for line in f:
        t = json.loads(line)
        e = eps.setdefault((t["checkpoint"], t["task_index"]), [0, 0.0, []])
        e[0] += 1
        e[1] = max(e[1], ts(t["timestamp"]))
        e[2].append(t["action_label"].split("(")[0])
mism = 0
for (c, ti), (n, bt, labels) in eps.items():
    j = bisect.bisect_right(times, bt) - 1
    for k, i in enumerate(range(j - n + 1, j + 1)):
        skills[i]["eval"] = [c, ti, k]
        lab = labels[k]
        nm = skills[i]["name"]
        if not (nm == lab or (nm.startswith("unknown") and lab not in ("PickCube", "MoveToTossLocationAndToss"))):
            mism += 1
# cycle: eval skills carry checkpoint c; practice skills after checkpoint c's eval belong to cycle c
cyc = -1
for s in skills:
    if "eval" in s:
        cyc = s["eval"][0]
        s["cycle"] = cyc
    else:
        s["cycle"] = cyc


def st_at(off):
    with open(RUN + "/tossing3d_state_log.jsonl", "rb") as f:
        f.seek(off)
        return json.loads(f.readline())


def summ(line):
    st = line["state"]
    b, c, r = st["bin_0"], st["cube_0"], st["robot"]
    out = {"cube": [round(v, 3) for v in c[:3]], "cube_tilt": round(tilt(c[3:7]), 1),
           "bin": [round(v, 3) for v in b[:3]], "bin_tilt": round(tilt(b[3:7]), 2),
           "robot": [round(r[0], 3), round(r[1], 3), round(math.degrees(r[2]), 1)]}
    for k in ("atoms", "abstract_atoms"):
        if k in line:
            out["atoms"] = line[k]
    return out


streaks = []
cur = None
last_tick = None
for i, s in enumerate(skills):
    if "eval" in s:
        continue
    refused = s["name"] == "PickCube" and s["ticks"] == 0
    if refused:
        if cur is None:
            cur = {"start": i, "n": 0, "prev": last_tick, "cycle0": s["cycle"]}
        cur["n"] += 1
        cur["end"] = i
        cur["cycle1"] = s["cycle"]
    elif cur is not None:
        cur["broken_by"] = [i, s["name"], s["ticks"]]
        streaks.append(cur)
        cur = None
    if s["ticks"]:
        last_tick = i
if cur:
    cur["broken_by"] = None
    streaks.append(cur)
for sk in sorted(streaks, key=lambda x: -x["n"])[:5]:
    p = sk["prev"]
    sk["prev_name"] = skills[p]["name"] if p is not None else None
    sk["prev_eval"] = skills[p].get("eval") if p is not None else None
    sk["state"] = summ(st_at(skills[p]["last"])) if p is not None else None
pr = [s for s in skills if "eval" not in s]
ev = [s for s in skills if "eval" in s]
st = json.load(open(RUN + "/stats.json"))
stats_att = Counter()
for c in st["practice_outcomes_per_cycle"]:
    for k, v in c.items():
        stats_att[k] += v["num_attempts"]
summary = {"eval_label_mismatch": mism, "eval_steps": sum(e[0] for e in eps.values()),
           "n_skills": len(skills),
           "practice_picks": sum(s["name"] == "PickCube" for s in pr),
           "practice_refused": sum(s["name"] == "PickCube" and s["ticks"] == 0 for s in pr),
           "practice_tosses": sum(s["name"] == "MoveToTossLocationAndToss" for s in pr),
           "practice_tosses_ticked": sum(s["name"] == "MoveToTossLocationAndToss" and s["ticks"] > 0 for s in pr),
           "practice_actions": len(pr),
           "eval_picks": sum(s["name"] == "PickCube" for s in ev),
           "eval_refused": sum(s["name"] == "PickCube" and s["ticks"] == 0 for s in ev),
           "practice_names": Counter((s["name"], s["ticks"] > 0) for s in pr).most_common(),
           "stats_attempts": dict(stats_att),
           "n_streaks": len(streaks), "longest_streak": max([s["n"] for s in streaks], default=0),
           "streak_len_top": sorted([s["n"] for s in streaks], reverse=True)[:10]}
summary["pick_count_matches_stats"] = summary["practice_picks"] == stats_att.get("PickCube", 0)
summary["toss_count_matches_stats"] = summary["practice_tosses"] == stats_att.get("MoveToTossLocationAndToss", 0)
slim = [{k: s.get(k) for k in ("name", "ticks", "first", "last", "off", "cycle", "eval", "hash", "params", "objects")}
        for s in skills]
json.dump({"run": RUN, "skills": slim, "streaks": streaks, "summary": summary}, open(OUT, "w"))
print(RUN, json.dumps(summary, default=str))
