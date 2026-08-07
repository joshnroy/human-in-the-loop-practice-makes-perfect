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

# Hypotheses for the throw parameter, given bin distance d and object weight w.
#
# Practice evidence (d = 2.0 throughout): p = 0.6424 succeeded at w = 1.212,
# while 0.7071@1.045, 0.7456@0.509, 0.7594@1.154, 0.8561@0.733, 0.9578@1.090
# and 1.4233@0.847 all failed.  p = 0.53 * w reproduces the success exactly and
# makes every failure an overshoot of 24% or more, so the parameter reads as a
# throw effort whose required value grows with the object's weight.  The
# alternatives that also fit the single success -- a weight-independent
# constant, and a sqrt(w) law -- are kept as backups, and each model class is
# swept over a range of coefficients in case the leading fit is off.
#
# Distance enters as a linear factor (d / 2), which is a no-op for the bins seen
# so far (both had throw_distance = 2.0) but scales sensibly if it ever varies.
_ORDER = (
    ("w", 0.53), ("c", 0.6424), ("s", 0.5835),
    ("w", 0.50), ("w", 0.56), ("c", 0.60), ("w", 0.47), ("w", 0.59),
    ("c", 0.45), ("w", 0.44), ("w", 0.62), ("s", 0.62), ("w", 0.41),
    ("w", 0.65), ("c", 0.30), ("w", 0.38), ("w", 0.68), ("s", 0.54),
    ("w", 0.35), ("w", 0.71), ("c", 0.15), ("w", 0.32), ("w", 0.74),
    ("w", 0.29), ("w", 0.77), ("c", 0.68), ("w", 0.26), ("w", 0.80),
    ("w", 0.23), ("w", 0.85), ("c", 0.05), ("w", 0.20), ("w", 0.90),
    ("w", 0.16), ("w", 0.95), ("c", 0.80), ("w", 0.12), ("w", 1.00),
    ("c", 1.00), ("w", 1.10), ("w", 0.08), ("c", 1.20), ("w", 1.25),
    # Long shots, in case a test bin sits well outside the practice regime.
    ("c", 1.50), ("w", 1.50), ("c", 2.00), ("w", 2.00), ("c", 0.02),
)

_MEM = {"prev": None, "learn": {}, "sig": None, "last_action": None}


def _hyp_value(kind, theta, d, w):
    scale = d / 2.0 if math.isfinite(d) and d > 0 else 1.0
    if kind == "w":
        return theta * w * scale
    if kind == "s":
        return theta * math.sqrt(w) * scale
    return theta * scale


def _learner(key):
    return _MEM["learn"].setdefault(
        key, {"score": [0] * len(_ORDER), "succ": [], "tries": [0] * len(_ORDER)}
    )


def _choose_param(key, d, w):
    """Return (param, hypothesis_index)."""
    mem = _learner(key)

    # Exactly this situation already solved once: reuse the known-good value.
    for (s_d, s_w, s_p) in reversed(mem["succ"]):
        if abs(s_d - d) < 1e-6 and abs(s_w - w) < 1e-6:
            return s_p, None

    best = 0
    best_score = None
    for i, _ in enumerate(_ORDER):
        # Prefer proven hypotheses; among untried ones keep the listed order.
        score = (mem["score"][i], -i)
        if best_score is None or score > best_score:
            best_score = score
            best = i
    kind, theta = _ORDER[best]
    val = _hyp_value(kind, theta, d, w)
    if not math.isfinite(val):
        val = theta
    mem["tries"][best] += 1
    return val, best


def _record_outcome(prev, atom_set):
    mem = _learner(prev["key"])
    idx = prev["hyp"]
    if prev["target"] in atom_set:
        mem["succ"].append((prev["d"], prev["w"], prev["p"]))
        if len(mem["succ"]) > 40:
            del mem["succ"][0]
        if idx is not None:
            mem["score"][idx] += 1
    elif idx is not None:
        mem["score"][idx] -= 1


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


def _signature(observation):
    """Fingerprint of the full state, symbolic and continuous."""
    try:
        feats = tuple(
            (o["name"], tuple(sorted(o.get("features", {}).items())))
            for o in sorted(observation["objects"], key=lambda o: o["name"])
        )
        return (tuple(sorted(observation["atoms"])), feats)
    except Exception:
        return None


def _pile_pickups(objects):
    for obj in objects.values():
        if obj["type"] == "pile":
            val = obj.get("features", {}).get("num_pickups")
            return None if val is None else float(val)
    return None


def _result(skills, i, param=None):
    dim = int(skills[i]["param_dim"])
    if dim == 0:
        params = []
    elif param is None:
        params = [0.0] * dim
    else:
        params = [float(param)] + [0.0] * (dim - 1)
    action = {"skill_index": int(i), "params": params}
    _MEM["last_action"] = action
    return action


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

    # A repeated call on an unchanged observation is the harness asking again,
    # not the environment having stepped: replay the cached answer and learn
    # nothing from it.  (Practice showed this happening ~15 times in a row on
    # one state, which otherwise burns a fresh guess on every repeat.)
    sig = _signature(observation)
    if sig is not None and sig == _MEM["sig"] and _MEM["last_action"] is not None:
        return _MEM["last_action"]
    _MEM["sig"] = sig

    # Grade the previous throw, if any.
    prev = _MEM["prev"]
    _MEM["prev"] = None
    if prev is not None:
        pickups = _pile_pickups(objects)
        reset = (
            pickups is not None
            and prev["pickups"] is not None
            and pickups < prev["pickups"]
        )
        if not reset:
            _record_outcome(prev, atom_set)

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
        param, hyp = _choose_param(throw, d, w)
        if not math.isfinite(param):
            param = 0.53 * w
        _MEM["prev"] = {
            "key": throw,
            "d": d,
            "w": w,
            "p": float(param),
            "hyp": hyp,
            "pickups": _pile_pickups(objects),
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
