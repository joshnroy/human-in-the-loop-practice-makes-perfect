"""Control policy for the trash/recycling robot domain.

Two parts:

1. A symbolic controller (navigate / pick / throw / press) driven by the atoms.
2. An active search for the one continuous parameter in the domain, the single
   number taken by ThrowTrash / ThrowRecycling.  Its meaning is unknown, so the
   policy sweeps an ordered list of candidate expressions in the observable
   features (the bin's ``throw_distance`` and the item's ``weight``), advancing
   only when a candidate is *confirmed* to have failed inside an episode, and
   pinning any candidate that is observed to work.
"""

import math
from collections import deque

__all__ = ["policy"]


# ------------------------------------------------- throw-parameter candidates --

_G = 9.81


def _candidates(d, w):
    """Ordered candidate values for a throw parameter.

    ``d`` is the target bin's throw_distance, ``w`` the held item's weight.
    Index i means the same *expression* in every episode, so a cursor into this
    list is a stable hypothesis id.  The head of the list holds the most
    plausible physical readings; the tail is a coarse absolute sweep in case the
    parameter is unrelated to the features.
    """
    sw = w if abs(w) > 1e-6 else 1e-6
    sd = d if d > 0.0 else 0.0
    rd = math.sqrt(sd)
    out = [
        d * w,                    # 0  force scaled by mass
        rd * abs(sw),             # 1  impulse for a projectile of that mass
        d / sw,                   # 2
        math.sqrt(sd * abs(sw)),  # 3  energy-like
        0.0,                      # 4  nominal / offset-style parameter
        1.0,                      # 5
        0.5,                      # 6
        w,                        # 7
        d + w,                    # 8
        2.0 * d,                  # 9
        0.5 * d,                  # 10
        rd,                       # 11
        rd / sw,                  # 12
        d * d,                     # 13
        0.25,                     # 14
        0.75,                     # 15
        0.1,                      # 16
        2.0,                      # 17
        5.0,                      # 18
        math.sqrt(_G * sd),       # 19 launch speed for range d
        abs(sw) * math.sqrt(_G * sd),   # 20
        d * math.sqrt(abs(sw)),   # 21
        d / (1.0 + w),            # 22
        d * (1.0 + w),            # 23
        d - w,                    # 24
        -d,                       # 25
        10.0,                     # 26
        0.2,                      # 27
        0.3,                      # 28
        0.4,                      # 29
        0.6,                      # 30
        0.7,                      # 31
        0.8,                      # 32
        0.9,                      # 33
        1.25,                     # 34
        1.5,                      # 35
        1.75,                     # 36
        2.5,                      # 37
        3.0,                      # 38
        3.5,                      # 39
        4.0,                      # 40
        6.0,                      # 41
        7.0,                      # 42
        8.0,                      # 43
        12.0,                     # 44
        15.0,                     # 45
        20.0,                     # 46
        25.0,                     # 47
        50.0,                     # 48
        100.0,                    # 49
        -1.0,                     # 50
        -0.5,                     # 51
        -2.0,                     # 52
        math.pi / 4.0,            # 53 angle-style parameters
        math.pi / 6.0,            # 54
        math.pi / 3.0,            # 55
        math.pi / 2.0,            # 56
        0.05,                     # 57
        d * 0.25,                 # 58
        d * 4.0,                  # 59
        d * w * w,                # 60
        d * d / sw,               # 61
        sd / _G,                  # 62
        math.sqrt(2.0 * sd / _G),  # 63 time of flight
        1.0 / sw,                 # 64
        abs(w) * 2.0,             # 65
        abs(w) * 0.5,             # 66
        d + 1.0,                  # 67
        d - 1.0,                  # 68
        d * 1.5,                  # 69
        d * 0.75,                 # 70
        d * 10.0,                 # 71
        0.15,                     # 72
        0.35,                     # 73
        0.45,                     # 74
        0.55,                     # 75
        0.65,                     # 76
        0.85,                     # 77
        0.95,                     # 78
        1.1,                      # 79
    ]
    return out


def _value(idx, d, w):
    cands = _candidates(d, w)
    v = cands[idx % len(cands)]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(-1e6, min(1e6, v))


# ------------------------------------------------------------ search memory ----

_S = {
    "cursor": 0,      # next hypothesis to test
    "pin": {},        # throw skill name -> hypothesis confirmed to work for it
    "good": None,     # most recent hypothesis that worked for any throw skill
    "pending": None,  # throw awaiting its outcome
    "ambig": {},      # hypothesis id -> times its outcome was unobservable
    "sig": None,      # signature of the current task instance
}


def _task_sig(feats):
    """Signature of the randomized, episode-static features."""
    out = []
    for name in sorted(feats):
        f = feats[name]
        for k in ("throw_distance", "weight_seed"):
            if k in f:
                try:
                    out.append((name, k, round(float(f[k]), 6)))
                except (TypeError, ValueError):
                    pass
    return tuple(out)


# ---------------------------------------------------------------- parsing ----

def _parse_atom(s):
    """'Pred(a, b)' -> ('Pred', ['a', 'b'])."""
    i = s.find("(")
    if i < 0:
        return s, []
    name = s[:i]
    inner = s[i + 1:s.rfind(")")].strip()
    if not inner:
        return name, []
    return name, [a.strip() for a in inner.split(",")]


def _index(atom_strings):
    out = {}
    for s in atom_strings:
        name, args = _parse_atom(s)
        out.setdefault(name, []).append(tuple(args))
    return out


def _first(idx, pred):
    lst = idx.get(pred)
    return lst[0] if lst else None


# ------------------------------------------------------------- navigation ----

def _next_hop(idx, start, goal_rooms):
    """First room along a shortest CanMoveRoom path; None if none is needed."""
    if start in goal_rooms:
        return None
    succ = {}
    for a, b in idx.get("CanMoveRoom", []):
        succ.setdefault(a, []).append(b)
    for vs in succ.values():
        vs.sort()
    prev = {start: None}
    q = deque([start])
    found = None
    while q:
        cur = q.popleft()
        if cur in goal_rooms:
            found = cur
            break
        for nxt in succ.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if found is None:
        return None
    node = found
    while prev[node] != start:
        node = prev[node]
        if node is None:  # pragma: no cover - defensive
            return None
    return node


# ----------------------------------------------------------------- skills ----

def _find(skills, name, **slots):
    """First skill with this name whose positional object slots match."""
    want = [(int(k[1:]), v) for k, v in slots.items() if v is not None]
    for sk in skills:
        if sk["name"] != name:
            continue
        objs = sk["objects"]
        ok = True
        for pos, val in want:
            if pos >= len(objs) or objs[pos] != val:
                ok = False
                break
        if ok:
            return sk
    return None


def _plain(sk):
    n = int(sk.get("param_dim", 0) or 0)
    return {"skill_index": int(sk["index"]), "params": [0.0] * n}


def _fallback(observation):
    skills = observation["skills"]
    for pref in ("MoveRoom", "PickupTrash", "PickupRecycling"):
        for sk in skills:
            if sk["name"] == pref:
                return _plain(sk)
    return _plain(skills[0])


# ----------------------------------------------------------------- policy ----

def policy(observation):
    skills = observation["skills"]
    atom_set = set(observation["atoms"])
    feats = {o["name"]: dict(o.get("features", {})) for o in observation["objects"]}
    types = {o["name"]: o["type"] for o in observation["objects"]}
    idx = _index(observation["atoms"])
    goal = list(observation["goal"])
    missing = [g for g in goal if g not in atom_set]

    rir = _first(idx, "RobotInRoom")
    if rir is None:
        return _fallback(observation)
    robot, cur_room = rir[0], rir[1]

    # -- resolve the outcome of the previous throw ----------------------------
    sig = _task_sig(feats)
    new_task = _S["sig"] is not None and sig != _S["sig"]
    _S["sig"] = sig
    pend = _S.pop("pending", None)
    _S["pending"] = None
    if pend is not None:
        skill, hyp = pend["skill"], pend["idx"]
        if pend["atom"] in atom_set:
            _S["pin"][skill] = hyp           # it worked: keep using it
            _S["good"] = hyp
        elif not new_task and cur_room == pend["room"]:
            if _S["pin"].get(skill) == hyp:  # confirmed failure in-episode
                del _S["pin"][skill]
            if _S["good"] == hyp:
                _S["good"] = None
            _S["cursor"] = hyp + 1
        else:
            # The episode ended before the outcome was visible -- most likely
            # because that throw solved the task.  Re-test the same hypothesis.
            n = _S["ambig"].get(hyp, 0) + 1
            _S["ambig"][hyp] = n
            if n >= 6 and not _S["pin"]:
                _S["cursor"] = hyp + 1
            else:
                _S["cursor"] = hyp

    def hypothesis(skill):
        h = _S["pin"].get(skill)
        if h is None:
            h = _S["good"]
        return _S["cursor"] if h is None else h

    # -- world layout --------------------------------------------------------
    room_of = {}
    for pred in ("TrashBinInRoom", "RecyclingBinInRoom", "TrashButtonInRoom",
                 "RecyclingButtonInRoom", "PileInRoom"):
        for args in idx.get(pred, []):
            room_of.setdefault(args[0], set()).add(args[1])

    def rooms_with(pred):
        return {args[1] for args in idx.get(pred, [])}

    def go(goal_rooms):
        if not goal_rooms:
            return None
        hop = _next_hop(idx, cur_room, set(goal_rooms))
        if hop is None:
            return None
        sk = _find(skills, "MoveRoom", a2=hop) or _find(skills, "MoveRoom")
        return _plain(sk) if sk else None

    def empty_bin(bin_name, kind):
        if kind == "trash":
            btn_pred, press = "TrashButtonInRoom", "PressTrash"
        else:
            btn_pred, press = "RecyclingButtonInRoom", "PressRecycling"
        sk = _find(skills, press, a3=bin_name) or _find(skills, press)
        if sk is not None:
            return _plain(sk)
        return go(rooms_with(btn_pred))

    def throw(item, bin_name, kind):
        """Emit a throw, recording which hypothesis is being tested."""
        name = "ThrowTrash" if kind == "trash" else "ThrowRecycling"
        pred = "TrashInBin" if kind == "trash" else "RecyclingInBin"
        sk = (_find(skills, name, a1=item, a2=bin_name)
              or _find(skills, name, a2=bin_name)
              or _find(skills, name))
        if sk is None:
            return None
        d = float(feats.get(bin_name, {}).get("throw_distance", 1.0))
        w = float(feats.get(item, {}).get("weight", 1.0))
        h = hypothesis(name)
        n = int(sk.get("param_dim", 0) or 0)
        params = [_value(h, d, w)] + [0.0] * max(0, n - 1)
        _S["pending"] = {"idx": h, "room": cur_room, "skill": name,
                         "atom": "%s(%s, %s)" % (pred, item, bin_name)}
        return {"skill_index": int(sk["index"]), "params": params[:n]}

    def deliver(item, bin_name, kind):
        if kind == "trash":
            in_room, empty_pred = "TrashBinInRoom", "TrashBinEmpty"
        else:
            in_room, empty_pred = "RecyclingBinInRoom", "RecyclingBinEmpty"
        bin_rooms = room_of.get(bin_name) or rooms_with(in_room)
        if cur_room not in bin_rooms:
            act = go(bin_rooms)
            if act is not None:
                return act
        if (bin_name,) not in idx.get(empty_pred, []):
            act = empty_bin(bin_name, kind)
            if act is not None:
                return act
        return throw(item, bin_name, kind)

    def pick(item, kind):
        name = "PickupTrash" if kind == "trash" else "PickupRecycling"
        sk = _find(skills, name, a1=item) or _find(skills, name)
        if sk is not None:
            return _plain(sk)
        return go(rooms_with("PileInRoom"))

    def bin_for(item, kind):
        pred = "TrashInBin" if kind == "trash" else "RecyclingInBin"
        for g in goal:
            n, a = _parse_atom(g)
            if n == pred and len(a) == 2 and a[0] == item:
                return a[1]
        in_room = "TrashBinInRoom" if kind == "trash" else "RecyclingBinInRoom"
        for args in idx.get(in_room, []):
            return args[0]
        btype = "trash_bin" if kind == "trash" else "recycling_bin"
        for name, t in sorted(types.items()):
            if t == btype:
                return name
        return None

    # -- 1) holding something: put it in its bin (also frees the hand) -------
    held = None
    for args in idx.get("HoldingTrash", []):
        if args[0] == robot:
            held = (args[1], "trash")
    if held is None:
        for args in idx.get("HoldingRecycling", []):
            if args[0] == robot:
                held = (args[1], "recycling")
    if held is not None:
        item, kind = held
        bin_name = bin_for(item, kind)
        if bin_name is not None:
            act = deliver(item, bin_name, kind)
            if act is not None:
                return act

    # -- 2) hand empty: chase the missing goal atoms -------------------------
    for g in missing:
        n, a = _parse_atom(g)
        if n == "TrashInBin" and len(a) == 2:
            act = pick(a[0], "trash")
            if act is not None:
                return act
        elif n == "RecyclingInBin" and len(a) == 2:
            act = pick(a[0], "recycling")
            if act is not None:
                return act

    for g in missing:
        n, a = _parse_atom(g)
        if n == "TrashBinEmpty" and len(a) == 1:
            act = empty_bin(a[0], "trash")
            if act is not None:
                return act
        elif n == "RecyclingBinEmpty" and len(a) == 1:
            act = empty_bin(a[0], "recycling")
            if act is not None:
                return act
        elif n == "RobotInRoom" and len(a) == 2:
            act = go({a[1]})
            if act is not None:
                return act

    return _fallback(observation)
