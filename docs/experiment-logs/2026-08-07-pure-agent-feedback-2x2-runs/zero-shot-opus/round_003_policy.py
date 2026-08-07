"""Control policy for the trash/recycling robot domain.

Three parts:

1. A symbolic controller (navigate / pick / throw / press) driven by the atoms.
2. An active search for the domain's one continuous parameter, the number taken
   by ThrowTrash / ThrowRecycling.  Its meaning is undocumented, so the policy
   sweeps an ordered list of candidate expressions in the observable features
   (the bin's ``throw_distance``, the item's ``weight``), skips candidates
   already known to fail, and pins any candidate observed to work.
3. A recovery behaviour: when no productive action exists the policy presses a
   blocking button if it can, else patrols least-recently-visited rooms.  It
   must never sit and oscillate between two rooms, because a stranded episode
   spends its whole horizon learning nothing.

The sweep index is derived from an *observable* counter (how many pickups have
happened this episode) so that it advances within an episode even if module
state does not survive between episodes.
"""

import math
from collections import deque

__all__ = ["policy"]


# ------------------------------------------------- throw-parameter candidates --

_G = 9.81


def _candidates(d, w):
    """Ordered candidate values for a throw parameter.

    ``d`` is the target bin's throw_distance, ``w`` the held item's weight.
    Index i names the same *expression* in every episode, so an index is a
    stable hypothesis id.  The head mixes hypothesis *classes* -- feature
    products, nominal constants, a very large value (in case success is
    "throw hard enough") and zero (in case it is "throw gently") -- so that a
    single attempt-rich episode samples several classes.  ``d`` alone is absent:
    it was already falsified 19/19.
    """
    sw = w if abs(w) > 1e-6 else 1e-6
    sd = d if d > 0.0 else 0.0
    rd = math.sqrt(sd)
    return [
        0.0,                            # 0  nominal: zero aim error / offset
        1.0,                            # 1  nominal unit
        10.0,                           # 2  large: any "throw hard enough" rule
        d * w,                          # 3  force scaled by mass
        math.sqrt(sd * abs(sw)),        # 4  energy-like
        d + w,                          # 5
        d / sw,                         # 6
        0.5,                            # 7
        w,                              # 8
        50.0,                           # 9
        2.0 * d,                        # 10
        0.5 * d,                        # 11
        d / 10.0,                       # 12 distance in other units
        rd * abs(sw),                   # 13 impulse for that mass
        rd,                             # 14
        0.25,                           # 15
        0.75,                           # 16
        0.1,                            # 17
        5.0,                            # 18
        d * d,                          # 19
        math.sqrt(_G * sd),             # 20 launch speed for range d
        abs(sw) * math.sqrt(_G * sd),   # 21
        d * math.sqrt(abs(sw)),         # 22
        d / (1.0 + w),                  # 23
        d * (1.0 + w),                  # 24
        d - w,                          # 25
        rd / sw,                        # 26
        -d,                             # 27
        2.0,                            # 28
        3.0,                            # 29
        0.2,                            # 30
        0.3,                            # 31
        0.4,                            # 32
        0.6,                            # 33
        0.7,                            # 34
        0.8,                            # 35
        0.9,                            # 36
        1.25,                           # 37
        1.5,                            # 38
        1.75,                           # 39
        2.5,                            # 40
        3.5,                            # 41
        4.0,                            # 42
        6.0,                            # 43
        8.0,                            # 44
        12.0,                           # 45
        15.0,                           # 46
        20.0,                           # 47
        25.0,                           # 48
        100.0,                          # 49
        1000.0,                         # 50
        -1.0,                           # 51
        -0.5,                           # 52
        -2.0,                           # 53
        math.pi / 4.0,                  # 54 angle-style parameters
        math.pi / 6.0,                  # 55
        math.pi / 3.0,                  # 56
        math.pi / 2.0,                  # 57
        0.05,                           # 58
        d * 0.25,                       # 59
        d * 4.0,                        # 60
        d * w * w,                      # 61
        d * d / sw,                     # 62
        sd / _G,                        # 63
        math.sqrt(2.0 * sd / _G),       # 64 time of flight
        1.0 / sw,                       # 65
        abs(w) * 2.0,                   # 66
        abs(w) * 0.5,                   # 67
        d + 1.0,                        # 68
        d - 1.0,                        # 69
        d * 1.5,                        # 70
        d * 0.75,                       # 71
        d * 10.0,                       # 72
        0.15,                           # 73
        0.35,                           # 74
        0.45,                           # 75
        0.55,                           # 76
        0.65,                           # 77
        0.85,                           # 78
        0.95,                           # 79
        1.1,                            # 80
    ]


_NCAND = len(_candidates(1.0, 1.0))


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
    "pin": {},        # throw skill name -> hypothesis confirmed to work for it
    "good": None,     # most recent hypothesis that worked for any throw skill
    "failed": set(),  # hypotheses falsified in some episode
    "pending": None,  # throw awaiting its outcome
    "ambig": {},      # hypothesis -> times its outcome was unobservable
    "sig": None,      # signature of the current task instance
    "room": None,     # robot's room on the previous step
    "pk": None,       # pickup counter on the previous step
    "tries": 0,       # throws made this episode
    "pk0": None,      # pile pickup counter at the start of this episode
    "visits": {},     # room -> times entered this episode (patrol memory)
    "dry": set(),     # pile rooms that offered no pickup this episode
    "badvals": set(), # parameter values already falsified for this instance
}


def _task_sig(feats):
    """Signature of the randomized but episode-static features."""
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


def _pickup_counter(feats):
    for name, f in feats.items():
        if "num_pickups" in f:
            try:
                return float(f["num_pickups"])
            except (TypeError, ValueError):
                return None
    return None


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
        return _plain(skills[0])
    robot, cur_room = rir[0], rir[1]

    # -- episode bookkeeping -------------------------------------------------
    sig = _task_sig(feats)
    pk = _pickup_counter(feats)
    reset = False
    if _S["sig"] is not None:
        if sig != _S["sig"]:
            reset = True                      # a different task instance
            _S["badvals"] = set()
        elif pk is not None and _S["pk"] is not None and pk > _S["pk"] + 0.5:
            reset = True                      # pickups are never regained
        elif (_S["room"] is not None and _S["room"] != cur_room
              and (_S["room"], cur_room) not in idx.get("CanMoveRoom", [])):
            reset = True                      # the robot teleported
    if reset:
        _S["tries"] = 0
        _S["visits"] = {}
        _S["dry"] = set()
        _S["pk0"] = pk
    if _S["sig"] is None or _S["pk0"] is None:
        _S["pk0"] = pk
    _S["sig"] = sig
    _S["pk"] = pk
    _S["room"] = cur_room

    # -- resolve the outcome of the previous throw ---------------------------
    pend = _S["pending"]
    _S["pending"] = None
    if pend is not None:
        skill, hyp = pend["skill"], pend["idx"]
        if pend["atom"] in atom_set:
            _S["pin"][skill] = hyp           # it worked: keep using it
            _S["good"] = hyp
            _S["failed"].discard(hyp)
        elif not reset and cur_room == pend["room"]:
            _S["failed"].add(hyp)            # confirmed failure in-episode
            if pend.get("val") is not None:
                _S["badvals"].add(round(pend["val"], 6))
            if _S["pin"].get(skill) == hyp:
                del _S["pin"][skill]
            if _S["good"] == hyp:
                _S["good"] = None
        else:
            # The episode ended before the outcome was visible -- most likely
            # because that throw solved the task.  Do not falsify it; re-test.
            n = _S["ambig"].get(hyp, 0) + 1
            _S["ambig"][hyp] = n
            if n >= 6 and not _S["pin"]:
                _S["failed"].add(hyp)

    # How many attempts have already happened this episode?  Prefer the
    # observable pickup counter, so the sweep still advances if module state
    # was not carried over from the previous episode.
    seen = _S["tries"]
    if pk is not None and _S["pk0"] is not None:
        # Each throw is preceded by exactly one pickup, so the number of throws
        # already made is one less than the number of pickups.
        seen = max(seen, int(round(abs(pk - _S["pk0"]))) - 1)
    seen = max(0, seen)

    def hypothesis(skill, d, w):
        h = _S["pin"].get(skill)
        if h is None:
            h = _S["good"]
        if h is not None:
            return h
        h = seen % _NCAND
        first = None
        for _ in range(_NCAND):           # skip falsified hypotheses and values
            if h not in _S["failed"]:
                if round(_value(h, d, w), 6) not in _S["badvals"]:
                    return h
                if first is None:
                    first = h
            h = (h + 1) % _NCAND
        return h if first is None else first

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
        sk = _find(skills, "MoveRoom", a2=hop)
        if sk is None:
            return None
        _S["visits"][hop] = _S["visits"].get(hop, 0) + 1
        return _plain(sk)

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
        h = hypothesis(name, d, w)
        n = int(sk.get("param_dim", 0) or 0)
        val = _value(h, d, w)
        params = [val] + [0.0] * max(0, n - 1)
        _S["tries"] += 1
        _S["pending"] = {"idx": h, "room": cur_room, "skill": name, "val": val,
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

    hand_empty = (robot,) in idx.get("HandEmpty", [])

    def pick(item, kind):
        name = "PickupTrash" if kind == "trash" else "PickupRecycling"
        sk = _find(skills, name, a1=item) or _find(skills, name)
        if sk is not None:
            return _plain(sk)
        if not hand_empty:
            return None       # every pickup needs HandEmpty; the pile is moot
        pile_rooms = rooms_with("PileInRoom")
        if cur_room in pile_rooms:
            # Standing at the pile, hand empty, yet no pickup is applicable:
            # this pile is spent.  Never walk back to it this episode.
            _S["dry"].add(cur_room)
        return go(pile_rooms - _S["dry"])

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

    def recover():
        """No productive action: unblock a bin, else patrol without oscillating."""
        # A full bin blocks every future throw into it; pressing is free unless
        # it would undo a goal atom we already achieved.
        for press, empty_pred, in_pred in (
                ("PressTrash", "TrashBinEmpty", "TrashInBin"),
                ("PressRecycling", "RecyclingBinEmpty", "RecyclingInBin")):
            for sk in skills:
                if sk["name"] != press:
                    continue
                objs = sk["objects"]
                bin_name = objs[3] if len(objs) > 3 else None
                if bin_name is None or (bin_name,) in idx.get(empty_pred, []):
                    continue
                undoes = False
                for g in goal:
                    n, a = _parse_atom(g)
                    if (n == in_pred and len(a) == 2 and a[1] == bin_name
                            and g in atom_set):
                        undoes = True
                if not undoes:
                    return _plain(sk)
        # Head for the nearest room where something could become possible: a
        # button for a bin that is still full, or a pile to pick up from.
        targets = set()
        for btn_pred, empty_pred, btype in (
                ("TrashButtonInRoom", "TrashBinEmpty", "trash_bin"),
                ("RecyclingButtonInRoom", "RecyclingBinEmpty", "recycling_bin")):
            full = [n for n, t in types.items()
                    if t == btype and (n,) not in idx.get(empty_pred, [])]
            if full:
                targets |= rooms_with(btn_pred)
        if hand_empty:
            targets |= rooms_with("PileInRoom") - _S["dry"]
        act = go(targets - {cur_room})
        if act is not None:
            return act
        # Otherwise sweep the whole map, least-visited room first, so the robot
        # cannot ping-pong between two rooms for the rest of the episode.
        rooms = sorted((n for n, t in types.items() if t == "room"),
                       key=lambda n: (_S["visits"].get(n, 0), n))
        for r in rooms:
            if r == cur_room:
                continue
            act = go({r})
            if act is not None:
                return act
        for sk in skills:
            if int(sk.get("param_dim", 0) or 0) == 0:
                return _plain(sk)
        return _plain(skills[0])

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

    return recover()
