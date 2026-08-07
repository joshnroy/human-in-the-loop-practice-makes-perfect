"""Control policy for the trash/recycling robot domain.

Strategy: a small symbolic controller.

  * Parse the atoms into a relational index and read the raw continuous
    features of every object.
  * Figure out which goal atoms are still missing and pick one subtask.
  * Achieve the subtask with the shortest legal skill sequence: navigate over
    the CanMoveRoom graph, pick up from the pile, throw into the bin, press a
    button when a bin needs emptying.

The only continuous parameter in the domain is the single number taken by
ThrowTrash / ThrowRecycling.  The bins expose a ``throw_distance`` feature, so
the throw parameter is emitted as that distance (see _throw_param).
"""

from collections import deque

__all__ = ["policy"]


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
    """Map predicate name -> list of argument tuples."""
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
    """First room to move to along a shortest CanMoveRoom path.

    Returns None if already at a goal room or no path exists.
    """
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
    """First skill with the given name whose object slots match.

    ``slots`` keys are positional indices spelled ``a0``, ``a1``, ... and the
    values are object names (or None for "don't care").
    """
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


def _throw_param(feats, bin_name):
    """Continuous parameter for a throw skill: the bin's throw distance."""
    f = feats.get(bin_name, {})
    for key in ("throw_distance", "distance"):
        if key in f:
            return float(f[key])
    return 1.0


def _act(sk, feats, bin_name=None):
    n = int(sk.get("param_dim", 0) or 0)
    if n <= 0:
        return {"skill_index": int(sk["index"]), "params": []}
    if bin_name is None:
        for o in sk["objects"]:
            if "bin" in o:
                bin_name = o
                break
    p = _throw_param(feats, bin_name) if bin_name else 0.0
    params = [float(p)] + [0.0] * (n - 1)
    return {"skill_index": int(sk["index"]), "params": params[:n]}


def _fallback(observation, feats):
    skills = observation["skills"]
    # Prefer a no-op-ish move over a destructive button press.
    for pref in ("MoveRoom", "PickupTrash", "PickupRecycling"):
        for sk in skills:
            if sk["name"] == pref:
                return _act(sk, feats)
    return _act(skills[0], feats)


# ----------------------------------------------------------------- policy ----

def policy(observation):
    skills = observation["skills"]
    feats = {o["name"]: dict(o.get("features", {})) for o in observation["objects"]}
    types = {o["name"]: o["type"] for o in observation["objects"]}
    idx = _index(observation["atoms"])
    goal = list(observation["goal"])
    missing = [g for g in goal if g not in observation["atoms"]]

    rir = _first(idx, "RobotInRoom")
    if rir is None:
        return _fallback(observation, feats)
    robot, cur_room = rir[0], rir[1]

    # Where things are.
    room_of = {}
    for pred in ("TrashBinInRoom", "RecyclingBinInRoom", "TrashButtonInRoom",
                 "RecyclingButtonInRoom", "PileInRoom"):
        for args in idx.get(pred, []):
            room_of.setdefault(args[0], set()).add(args[1])

    def rooms_with(pred):
        return {args[1] for args in idx.get(pred, [])}

    def go(goal_rooms):
        """Move one step toward any room in goal_rooms; None if impossible."""
        if not goal_rooms:
            return None
        hop = _next_hop(idx, cur_room, set(goal_rooms))
        if hop is None:
            return None
        sk = _find(skills, "MoveRoom", a2=hop)
        if sk is None:
            sk = _find(skills, "MoveRoom")
        return _act(sk, feats) if sk else None

    def empty_bin(bin_name, kind):
        """Press the matching button so bin_name becomes empty."""
        if kind == "trash":
            btn_pred, press, bin_slot = "TrashButtonInRoom", "PressTrash", "a3"
        else:
            btn_pred, press, bin_slot = ("RecyclingButtonInRoom",
                                         "PressRecycling", "a3")
        sk = _find(skills, press, **{bin_slot: bin_name})
        if sk is None:
            sk = _find(skills, press)
        if sk is not None:
            return _act(sk, feats)
        return go(rooms_with(btn_pred))

    def deliver(item, bin_name, kind):
        """Robot holds `item`; put it into bin_name."""
        if kind == "trash":
            in_room, empty_pred, throw = ("TrashBinInRoom", "TrashBinEmpty",
                                          "ThrowTrash")
        else:
            in_room, empty_pred, throw = ("RecyclingBinInRoom",
                                          "RecyclingBinEmpty",
                                          "ThrowRecycling")
        bin_rooms = room_of.get(bin_name) or rooms_with(in_room)
        if cur_room not in bin_rooms:
            act = go(bin_rooms)
            if act is not None:
                return act
        if (bin_name,) not in idx.get(empty_pred, []):
            act = empty_bin(bin_name, kind)
            if act is not None:
                return act
        sk = _find(skills, throw, a1=item, a2=bin_name)
        if sk is None:
            sk = _find(skills, throw, a2=bin_name)
        if sk is None:
            sk = _find(skills, throw)
        if sk is not None:
            return _act(sk, feats, bin_name=bin_name)
        return None

    def pick(item, kind):
        """Hand is empty; grab `item` from the pile."""
        name = "PickupTrash" if kind == "trash" else "PickupRecycling"
        sk = _find(skills, name, a1=item)
        if sk is None:
            sk = _find(skills, name)
        if sk is not None:
            return _act(sk, feats)
        pile_rooms = set()
        for args in idx.get("PileInRoom", []):
            pile_rooms.add(args[1])
        return go(pile_rooms)

    def bin_for(item, kind):
        """Goal bin for item, else any bin of the right type."""
        pred = "TrashInBin" if kind == "trash" else "RecyclingInBin"
        btype = "trash_bin" if kind == "trash" else "recycling_bin"
        for g in goal:
            n, a = _parse_atom(g)
            if n == pred and len(a) == 2 and a[0] == item:
                return a[1]
        in_room = "TrashBinInRoom" if kind == "trash" else "RecyclingBinInRoom"
        for args in idx.get(in_room, []):
            return args[0]
        for name, t in types.items():
            if t == btype:
                return name
        return None

    # 1) Holding something -> get rid of it usefully (frees the hand either way).
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
        pred = "TrashInBin" if kind == "trash" else "RecyclingInBin"
        bin_name = bin_for(item, kind)
        if bin_name is not None and "%s(%s, %s)" % (pred, item, bin_name) in missing:
            act = deliver(item, bin_name, kind)
            if act is not None:
                return act
        # Not needed for the goal, but we must free the hand to continue.
        if bin_name is not None:
            act = deliver(item, bin_name, kind)
            if act is not None:
                return act

    # 2) Hand empty: work on the missing goal atoms, in-bin goals first.
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
        elif n == "HandEmpty":
            pass  # handled above by delivering whatever is held

    return _fallback(observation, feats)
