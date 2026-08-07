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

Candidates are ranked by how often they have been seen to fail, so a single
failure never permanently retires a value that has evidence behind it -- a throw
may well be noisy.  Any note about the *current* episode (a spent pile, room
visit counts) expires, so stale evidence cannot veto an action for a whole run.
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
    stable hypothesis id.  The head is d*w, which is the value a throw was
    observed to succeed with, followed by its numeric neighbours in case the
    real target is merely close to it; bare constants sit at the back, having
    been falsified in practice.
    """
    sw = w if abs(w) > 1e-6 else 1e-6
    sd = d if d > 0.0 else 0.0
    rd = math.sqrt(sd)
    return [
        d * w,                          # 0  CONFIRMED: force scaled by mass
        math.sqrt(sd * abs(sw)),        # 1  the other reading of that success
        d * w * 1.05,                   # 2  near-neighbours, in case the target
        d * w * 0.95,                   # 3  is close to but not exactly d*w
        d * w + 0.1,                    # 4
        d * w - 0.1,                    # 5
        d * w * 1.2,                    # 6
        d * w * 0.8,                    # 7
        d * w + 0.5,                    # 8
        d * w * 1.5,                    # 9
        d * w * 0.5,                    # 10
        d * w * 2.0,                    # 11
        d + w,                          # 12
        d / sw,                         # 13
        rd * abs(sw),                   # 14 impulse for that mass
        d * (1.0 + w),                  # 15
        d / (1.0 + w),                  # 16
        d * math.sqrt(abs(sw)),         # 17
        w,                              # 18
        2.0 * d,                        # 19
        0.5 * d,                        # 20
        d / 10.0,                       # 21
        d * w * w,                      # 22
        d * d * w,                      # 23
        math.sqrt(_G * sd) * abs(sw),   # 24
        rd,                             # 25
        d * d,                          # 26
        math.sqrt(_G * sd),             # 27
        d - w,                          # 28
        rd / sw,                        # 29
        d * 0.25,                       # 30
        d * 0.75,                       # 31
        d * 1.5,                        # 32
        d * 4.0,                        # 33
        d + 1.0,                        # 34
        d - 1.0,                        # 35
        d,                              # 36 (falsified 19/19 in period 1)
        0.5,                            # 37
        0.25,                           # 38
        0.75,                           # 39
        0.1,                            # 40
        2.0,                            # 41
        5.0,                            # 42
        3.0,                            # 43
        1.5,                            # 44
        0.2,                            # 45
        0.3,                            # 46
        0.4,                            # 47
        0.6,                            # 48
        0.7,                            # 49
        0.8,                            # 50
        0.9,                            # 51
        1.25,                           # 52
        1.75,                           # 53
        2.5,                            # 54
        3.5,                            # 55
        4.0,                            # 56
        6.0,                            # 57
        8.0,                            # 58
        12.0,                           # 59
        15.0,                           # 60
        20.0,                           # 61
        25.0,                           # 62
        50.0,                           # 63
        100.0,                          # 64
        0.05,                           # 65
        0.15,                           # 66
        0.35,                           # 67
        0.45,                           # 68
        0.55,                           # 69
        0.65,                           # 70
        0.85,                           # 71
        0.95,                           # 72
        1.1,                            # 73
        1.0,                            # 74 (falsified in period 5)
        10.0,                           # 75 (falsified in period 5)
        0.0,                            # 76 (falsified in period 5)
        -d * w,                         # 77
        -1.0,                           # 78
        1.0 / sw,                       # 79
        sd / _G,                        # 80
        math.sqrt(2.0 * sd / _G),       # 81
        math.pi / 4.0,                  # 82
    ]


_NCAND = len(_candidates(1.0, 1.0))

# Retry allowance for the candidates with evidence behind them.
_BONUS = {0: 2, 1: 1}


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
    "fails": {},      # hypothesis -> times it has been seen to fail
    "pending": None,  # throw awaiting its outcome
    "ambig": {},      # hypothesis -> times its outcome was unobservable
    "sig": None,      # signature of the current task instance
    "room": None,     # robot's room on the previous step
    "pk": None,       # pickup counter on the previous step
    "tries": 0,       # throws made this episode
    "pk0": None,      # pile pickup counter at the start of this episode
    "visits": {},     # room -> times entered this episode (patrol memory)
    "dry": {},        # pile room -> call index when it offered no pickup
    "calls": 0,       # monotone call counter, used to expire "dry" entries
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


_DRY_TTL = 40


def _dry_rooms():
    """Pile rooms recently seen to offer no pickup (the note expires)."""
    now = _S["calls"]
    return {r for r, t in _S["dry"].items() if now - t < _DRY_TTL}


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

    _S["calls"] += 1
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
        _S["dry"] = {}
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
            _S["fails"][hyp] = 0
        elif not reset and cur_room == pend["room"]:
            _S["fails"][hyp] = _S["fails"].get(hyp, 0) + 1
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
                _S["fails"][hyp] = _S["fails"].get(hyp, 0) + 1

    # How many attempts have already happened this episode?  Prefer the
    # observable pickup counter, so the sweep still advances if module state
    # was not carried over from the previous episode.
    # The falsified sets are updated as soon as a throw resolves, so scanning
    # from index 0 always lands on the first hypothesis not yet ruled out --
    # within an episode and across episodes alike.

    def hypothesis(skill, d, w):
        h = _S["pin"].get(skill)
        if h is None:
            h = _S["good"]
        if h is not None:
            return h
        # Fewest failures wins, list order breaks ties.  The two front-runners
        # carry a bonus, so a value with real evidence behind it is retried a
        # few times before the sweep moves on.
        best = None
        for i in range(_NCAND):
            if i >= 2 and round(_value(i, d, w), 6) in _S["badvals"]:
                continue                  # already ruled out this episode
            score = _S["fails"].get(i, 0) - _BONUS.get(i, 0)
            if best is None or (score, i) < best[0]:
                best = ((score, i), i)
        return 0 if best is None else best[1]

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
        # Empty the bin *before* walking to it: a full bin blocks the throw, and
        # heading for the bin first makes the robot shuttle between the bin room
        # and the button room without ever pressing.
        if (bin_name,) not in idx.get(empty_pred, []):
            act = empty_bin(bin_name, kind)
            if act is not None:
                return act
        bin_rooms = room_of.get(bin_name) or rooms_with(in_room)
        if cur_room not in bin_rooms:
            act = go(bin_rooms)
            if act is not None:
                return act
        return throw(item, bin_name, kind)

    hand_empty = (robot,) in idx.get("HandEmpty", [])

    def pick(item, kind):
        name = "PickupTrash" if kind == "trash" else "PickupRecycling"
        sk = _find(skills, name, a1=item) or _find(skills, name)
        if sk is not None:
            _S["dry"].clear()
            return _plain(sk)
        if not hand_empty:
            return None       # every pickup needs HandEmpty; the pile is moot
        pile_rooms = rooms_with("PileInRoom")
        if cur_room in pile_rooms:
            # Standing at the pile, hand empty, yet no pickup is applicable:
            # this pile looks spent.  Recorded with a timestamp so the note
            # expires -- a hard veto here would strand the whole episode.
            _S["dry"][cur_room] = _S["calls"]
        return go(pile_rooms - _dry_rooms())

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
            targets |= rooms_with("PileInRoom") - _dry_rooms()
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
