"""Control policy for the trash / recycling robot domain.

Two parts:

1. A symbolic controller.  The task structure is fixed: pick an item out of the
   pile, carry it to the matching bin (emptying the bin with the button first if
   it is full), throw it in.  Room-to-room navigation is BFS over the
   ``CanMoveRoom`` atoms, so one-way doors (``blocks_right``) are respected.

2. An online learner for the single continuous parameter of ``ThrowTrash`` /
   ``ThrowRecycling``.  The meaning of that parameter is unknown, so we treat it
   as an unknown function of the bin's ``throw_distance`` (d) and the held
   object's ``weight`` (w).  We enumerate a family of candidate forms, watch
   whether the object actually landed in the bin on the following step, and lock
   onto whatever works.  State persists across calls (module level), so the
   learning carries over between episodes of one evaluation.
"""

import math
import re
from collections import deque

_ATOM_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\(([^)]*)\)")


def _parse_atom(text):
    """'Pred(a, b)' -> ('Pred', ('a', 'b'))."""
    match = _ATOM_RE.match(text.strip())
    if match is None:
        return text.strip(), ()
    args = tuple(a.strip() for a in match.group(2).split(",") if a.strip())
    return match.group(1), args


# ---------------------------------------------------------------------------
# Throw-parameter learner
# ---------------------------------------------------------------------------

# Candidate functional forms for the throw parameter, given bin distance d and
# object weight w.  Ordered by prior plausibility: the parameter most likely
# just names the distance to throw.
_FAMILIES = (
    lambda d, w: d,
    lambda d, w: d * w,
    lambda d, w: d / w,
    lambda d, w: d * math.sqrt(w),
    lambda d, w: d / math.sqrt(w),
    lambda d, w: d + w,
    lambda d, w: d - w,
    lambda d, w: math.sqrt(d),
    lambda d, w: d * d,
    lambda d, w: math.sqrt(d * w),
    lambda d, w: math.sqrt(d / w),
    lambda d, w: w,
    lambda d, w: 1.0,
)

_MEM = {"prev": None, "learn": {}}


def _family_values(d, w):
    out = []
    for fam in _FAMILIES:
        try:
            val = float(fam(d, w))
        except Exception:
            val = None
        if val is None or not math.isfinite(val):
            val = None
        out.append(val)
    return out


def _explore_list(d, w):
    """Deterministic ordered list of parameter guesses to sweep through."""
    fam_vals = _family_values(d, w)
    vals = [v for v in fam_vals if v is not None]
    for scale in (0.5, 2.0, 0.25, 4.0, 1.5, 0.75, 0.1, 10.0):
        vals.extend(scale * v for v in fam_vals if v is not None)
    vals.extend((0.0, 0.2, 1.0, 3.0, 30.0))
    seen = set()
    out = []
    for val in vals:
        key = round(val, 6)
        if key in seen or not math.isfinite(val):
            continue
        seen.add(key)
        out.append(val)
    return out


def _model_predict(successes, d, w):
    """Best-fitting family: the one whose p / f(d, w) ratio is most constant."""
    best = None
    best_score = None
    for f_i, fam in enumerate(_FAMILIES):
        ratios = []
        ok = True
        for (s_d, s_w, s_p) in successes:
            try:
                val = float(fam(s_d, s_w))
            except Exception:
                ok = False
                break
            if not math.isfinite(val) or abs(val) < 1e-9:
                ok = False
                break
            ratios.append(s_p / val)
        if not ok:
            continue
        try:
            cur = float(fam(d, w))
        except Exception:
            continue
        if not math.isfinite(cur):
            continue
        median = sorted(ratios)[len(ratios) // 2]
        if abs(median) < 1e-12:
            continue
        score = ((max(ratios) - min(ratios)) / abs(median), f_i)
        if best_score is None or score < best_score:
            best_score = score
            best = median * cur
    return best


def _learner(key):
    return _MEM["learn"].setdefault(
        key, {"idx": 0, "succ": [], "fail": set(), "model_fails": 0}
    )


def _choose_param(key, d, w):
    """Return (param, explore_index_or_None)."""
    mem = _learner(key)
    dw = (round(d, 4), round(w, 4))

    # Something already known to work for exactly this situation.
    for (s_d, s_w, s_p) in reversed(mem["succ"]):
        if (round(s_d, 4), round(s_w, 4)) == dw:
            return s_p, None

    if mem["succ"] and mem["model_fails"] < 4:
        pred = _model_predict(mem["succ"], d, w)
        if pred is not None and dw + (round(pred, 4),) not in mem["fail"]:
            return pred, None

    candidates = _explore_list(d, w)
    i = mem["idx"]
    while i < len(candidates) and dw + (round(candidates[i], 4),) in mem["fail"]:
        i += 1
    if i < len(candidates):
        return candidates[i], i
    # Exhausted the structured guesses: deterministic fallback sweep.
    n = mem["idx"] - len(candidates)
    return 0.05 * (((n * 7) % 100) + 1) * max(abs(d), 1.0), mem["idx"]


def _record_outcome(prev, atom_set, room_of_robot):
    mem = _learner(prev["key"])
    if prev["target"] in atom_set:
        mem["succ"].append((prev["d"], prev["w"], prev["p"]))
        if len(mem["succ"]) > 40:
            del mem["succ"][0]
        mem["model_fails"] = 0
        return
    # Only judge a failure if we are still in the same episode/room; otherwise
    # the environment may simply have reset under us.
    if room_of_robot is not None and room_of_robot != prev["room"]:
        return
    if prev["hold"] not in atom_set and prev["target"] not in atom_set:
        # Object gone but not in the bin (dropped/lost), or a reset we cannot
        # distinguish.  Treat as a failure but do not blacklist the value.
        if prev["explore_idx"] is not None:
            mem["idx"] = max(mem["idx"], prev["explore_idx"] + 1)
        else:
            mem["model_fails"] += 1
        return
    mem["fail"].add((round(prev["d"], 4), round(prev["w"], 4), round(prev["p"], 4)))
    if prev["explore_idx"] is not None:
        mem["idx"] = max(mem["idx"], prev["explore_idx"] + 1)
    else:
        mem["model_fails"] += 1


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def _next_hop(graph, start, target):
    if start is None or target is None or start == target:
        return None
    parent = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, ()):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == target:
                cur = nxt
                while parent[cur] != start:
                    cur = parent[cur]
                return cur
            queue.append(nxt)
    return None


def _distances(graph, start):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, ()):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def policy(observation):
    try:
        return _policy(observation)
    except Exception:
        skill = observation["skills"][0]
        return {"skill_index": 0, "params": [0.0] * int(skill["param_dim"])}


def _result(skills, i, param=None):
    dim = int(skills[i]["param_dim"])
    if dim == 0:
        params = []
    elif param is None:
        params = [0.0] * dim
    else:
        params = [float(param)] + [0.0] * (dim - 1)
    return {"skill_index": int(i), "params": params}


def _policy(observation):
    skills = observation["skills"]
    objects = {o["name"]: o for o in observation["objects"]}
    atom_strs = observation["atoms"]

    by_pred = {}
    atom_set = set()
    for text in atom_strs:
        name, args = _parse_atom(text)
        atom_set.add((name, args))
        by_pred.setdefault(name, []).append(args)

    goal = [_parse_atom(g) for g in observation["goal"]]
    unsat = [g for g in goal if g not in atom_set]

    robot = next(
        (n for n, o in objects.items() if o["type"] == "robot"), None
    )
    room_of_robot = None
    for args in by_pred.get("RobotInRoom", ()):
        if args[0] == robot:
            room_of_robot = args[1]
            break

    # Grade the previous throw, if any.
    prev = _MEM["prev"]
    _MEM["prev"] = None
    if prev is not None:
        _record_outcome(prev, atom_set, room_of_robot)

    # Room connectivity.
    graph = {}
    for args in by_pred.get("CanMoveRoom", ()):
        graph.setdefault(args[0], set()).add(args[1])
    graph = {k: sorted(v) for k, v in graph.items()}

    # Static locations.
    def _room_of(pred, obj):
        for args in by_pred.get(pred, ()):
            if args[0] == obj:
                return args[1]
        return None

    def _first_of_type(type_name):
        for name, obj in sorted(objects.items()):
            if obj["type"] == type_name:
                return name
        return None

    pile = _first_of_type("pile")
    pile_room = _room_of("PileInRoom", pile) if pile else None

    def _move_toward(target_room):
        hop = _next_hop(graph, room_of_robot, target_room)
        if hop is not None:
            for i, sk in enumerate(skills):
                if sk["name"] == "MoveRoom" and sk["objects"][-1] == hop:
                    return _result(skills, i)
        return None

    def _skill_named(name, match=None):
        for i, sk in enumerate(skills):
            if sk["name"] != name:
                continue
            if match is None or match(sk["objects"]):
                return i
        return None

    def _any_named(name):
        return _skill_named(name)

    # ---- goal already met: do something harmless -------------------------
    if not unsat:
        i = _any_named("MoveRoom")
        if i is not None:
            return _result(skills, i)
        return _result(skills, 0)

    # ---- what are we holding? -------------------------------------------
    held_trash = None
    held_rec = None
    for args in by_pred.get("HoldingTrash", ()):
        if args[0] == robot:
            held_trash = args[1]
    for args in by_pred.get("HoldingRecycling", ()):
        if args[0] == robot:
            held_rec = args[1]

    def _dispose(item, is_trash):
        """Carry `item` to its bin (emptying the bin first if needed)."""
        if is_trash:
            bin_type, in_bin, empty_pred = "trash_bin", "TrashInBin", "TrashBinEmpty"
            bin_room_pred, throw = "TrashBinInRoom", "ThrowTrash"
            button_pred, press = "TrashButtonInRoom", "PressTrash"
            hold_pred = "HoldingTrash"
        else:
            bin_type, in_bin = "recycling_bin", "RecyclingInBin"
            empty_pred = "RecyclingBinEmpty"
            bin_room_pred, throw = "RecyclingBinInRoom", "ThrowRecycling"
            button_pred, press = "RecyclingButtonInRoom", "PressRecycling"
            hold_pred = "HoldingRecycling"

        target_bin = None
        for (name, args) in unsat:
            if name == in_bin and args[0] == item:
                target_bin = args[1]
                break
        if target_bin is None:
            target_bin = _first_of_type(bin_type)
        if target_bin is None:
            return None

        # Bin full: press the matching button first.
        if (empty_pred, (target_bin,)) not in atom_set:
            i = _skill_named(press, lambda o: len(o) > 3 and o[3] == target_bin)
            if i is None:
                i = _any_named(press)
            if i is not None:
                return _result(skills, i)
            button_rooms = [a[1] for a in by_pred.get(button_pred, ())]
            if button_rooms:
                dist = _distances(graph, room_of_robot)
                button_rooms.sort(key=lambda r: (dist.get(r, 10 ** 6), r))
                act = _move_toward(button_rooms[0])
                if act is not None:
                    return act

        bin_room = _room_of(bin_room_pred, target_bin)
        if bin_room is not None and bin_room != room_of_robot:
            act = _move_toward(bin_room)
            if act is not None:
                return act

        i = _skill_named(
            throw, lambda o: o[1] == item and o[2] == target_bin
        )
        if i is None:
            i = _skill_named(throw, lambda o: o[1] == item)
        if i is None:
            i = _any_named(throw)
        if i is None:
            return None

        bin_obj = objects.get(target_bin, {})
        item_obj = objects.get(item, {})
        d = float(bin_obj.get("features", {}).get("throw_distance", 1.0))
        w = float(item_obj.get("features", {}).get("weight", 1.0))
        if not math.isfinite(d):
            d = 1.0
        if not math.isfinite(w) or abs(w) < 1e-6:
            w = 1e-6
        param, explore_idx = _choose_param(throw, d, w)
        if not math.isfinite(param):
            param = d
        _MEM["prev"] = {
            "key": throw,
            "d": d,
            "w": w,
            "p": float(param),
            "explore_idx": explore_idx,
            "room": room_of_robot,
            "target": (in_bin, (item, target_bin)),
            "hold": (hold_pred, (robot, item)),
        }
        return _result(skills, i, param)

    if held_trash is not None:
        act = _dispose(held_trash, True)
        if act is not None:
            return act
    if held_rec is not None:
        act = _dispose(held_rec, False)
        if act is not None:
            return act

    # ---- hand empty: decide what to fetch next --------------------------
    wanted = []  # (is_trash, item, bin)
    for (name, args) in unsat:
        if name == "TrashInBin":
            wanted.append((True, args[0], args[1]))
        elif name == "RecyclingInBin":
            wanted.append((False, args[0], args[1]))

    if wanted:
        ref = pile_room if pile_room is not None else room_of_robot
        dist = _distances(graph, ref) if ref is not None else {}

        def _cost(entry):
            is_trash, _item, bin_name = entry
            pred = "TrashBinInRoom" if is_trash else "RecyclingBinInRoom"
            empty = "TrashBinEmpty" if is_trash else "RecyclingBinEmpty"
            room = _room_of(pred, bin_name)
            full_penalty = 0 if (empty, (bin_name,)) in atom_set else 100
            return (full_penalty + dist.get(room, 10 ** 5), 0 if is_trash else 1)

        wanted.sort(key=_cost)
        is_trash, item, target_bin = wanted[0]

        # Empty the destination bin before the trip to the pile.
        empty_pred = "TrashBinEmpty" if is_trash else "RecyclingBinEmpty"
        if (empty_pred, (target_bin,)) not in atom_set:
            press = "PressTrash" if is_trash else "PressRecycling"
            button_pred = (
                "TrashButtonInRoom" if is_trash else "RecyclingButtonInRoom"
            )
            i = _skill_named(press, lambda o: len(o) > 3 and o[3] == target_bin)
            if i is None:
                i = _any_named(press)
            if i is not None:
                return _result(skills, i)
            button_rooms = [a[1] for a in by_pred.get(button_pred, ())]
            if button_rooms:
                d_here = _distances(graph, room_of_robot)
                button_rooms.sort(key=lambda r: (d_here.get(r, 10 ** 6), r))
                act = _move_toward(button_rooms[0])
                if act is not None:
                    return act

        pick = "PickupTrash" if is_trash else "PickupRecycling"
        i = _skill_named(pick, lambda o: o[1] == item)
        if i is None:
            i = _any_named(pick)
        if i is not None:
            return _result(skills, i)
        if pile_room is not None:
            act = _move_toward(pile_room)
            if act is not None:
                return act

    # ---- other goal shapes ----------------------------------------------
    for (name, args) in unsat:
        if name == "TrashBinEmpty":
            i = _skill_named("PressTrash", lambda o: len(o) > 3 and o[3] == args[0])
            if i is None:
                i = _any_named("PressTrash")
            if i is not None:
                return _result(skills, i)
            rooms = [a[1] for a in by_pred.get("TrashButtonInRoom", ())]
            if rooms:
                act = _move_toward(rooms[0])
                if act is not None:
                    return act
        elif name == "RecyclingBinEmpty":
            i = _skill_named(
                "PressRecycling", lambda o: len(o) > 3 and o[3] == args[0]
            )
            if i is None:
                i = _any_named("PressRecycling")
            if i is not None:
                return _result(skills, i)
            rooms = [a[1] for a in by_pred.get("RecyclingButtonInRoom", ())]
            if rooms:
                act = _move_toward(rooms[0])
                if act is not None:
                    return act
        elif name == "RobotInRoom" and args[0] == robot:
            act = _move_toward(args[1])
            if act is not None:
                return act

    i = _any_named("MoveRoom")
    if i is not None:
        return _result(skills, i)
    return _result(skills, 0)
