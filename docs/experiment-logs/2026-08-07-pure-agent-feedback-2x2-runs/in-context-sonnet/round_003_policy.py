"""Reactive control policy for the trash/recycling sorting robot domain.

Strategy (no real planner, just greedy reactive rules re-derived every step
from the current observation, since the environment is fully re-observed
each call):

  1. If the robot is holding an item that some unsatisfied goal wants placed
     in a specific bin:
       - if already in that bin's room and the bin is empty -> throw it,
       - if already in that bin's room but the bin is occupied -> press the
         matching button to empty the bin first,
       - otherwise -> move one step closer to the bin's room.
  2. Otherwise (hand empty), pick the first still-needed trash/recycling
     item:
       - if a pickup skill for it is legal right now -> take it,
       - otherwise -> move one step closer to the nearest room with a pile.
  3. If nothing is applicable (goal already satisfied, or something
     unexpected), fall back to a harmless legal skill.

The only continuous parameter used by any skill is ThrowTrash/ThrowRecycling's
single float. Its meaning is not documented. Evidence so far (from practice):
  - params == the bin's "throw_distance" feature (a constant 2.0 for every
    bin seen): failed 19/19, across item weights ranging ~0.51-1.48.
  - params == the thrown item's own "weight" feature (~1.0-1.4 for
    recycling items, bin throw_distance still 2.0): failed 2/2.
So neither the bin's stated distance nor the item's weight alone is the
right value; the true relationship is unknown. A first attempt at an online
search kept a call counter in module-level state, but that turned out to be
unreliable (the observed practice log showed parameter values that did not
match what a freshly-reset counter should have produced, suggesting the
harness does not call policy() exactly once per logged execution). Instead,
we derive the search index from something actually visible in the
observation and reproducible across calls: the shared pile's "num_pickups"
feature, which practice confirms increments by exactly one each time an
item is newly picked up (it went 1..19 across 19 pickup/throw cycles in one
trash episode) and resets to 1 at the start of each fresh episode. Using
`num_pickups - 1` as an index into a fixed numeric sweep means every new
throw attempt into a bin -- across an episode with many pickup/throw
cycles, or one per episode when only a single attempt is possible -- tries
the next untested value, with no dependence on internal call counts.
"""

from collections import deque


def _throw_param_candidates():
    # throw_distance (2.0) and weight (~0.5-1.5) both failed as standalone
    # guesses, so sweep broadly and finely around and beyond that range
    # rather than trusting either feature specifically.
    fine = [round(i * 0.1, 3) for i in range(0, 31)]  # 0.0 .. 3.0
    mid = [round(i * 0.5, 3) for i in range(6, 21)]  # 3.0 .. 10.0
    coarse = [12.0, 15.0, 20.0, 30.0, 50.0, -1.0, -2.0, -5.0]
    seen = set()
    out = []
    for v in fine + mid + coarse:
        v = round(float(v), 3)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


_THROW_PARAM_CANDIDATES = _throw_param_candidates()


def _pile_num_pickups(objects):
    for o in objects:
        if o.get("type") == "pile":
            return o.get("features", {}).get("num_pickups", 1.0)
    return 1.0


def _throw_param_for(num_pickups):
    idx = int(round(num_pickups)) - 1
    if idx < 0:
        idx = 0
    return _THROW_PARAM_CANDIDATES[idx % len(_THROW_PARAM_CANDIDATES)]


def _parse_atom(atom):
    name, rest = atom.split("(", 1)
    rest = rest[:-1]
    args = [a.strip() for a in rest.split(",")] if rest else []
    return name, args


def _bfs_next_step(start, targets, edges):
    if start in targets:
        return None
    prev = {start: None}
    q = deque([start])
    dest = None
    while q:
        cur = q.popleft()
        if cur in targets:
            dest = cur
            break
        for nxt in edges.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if dest is None:
        return None
    node = dest
    while prev[node] is not None and prev[prev[node]] is not None:
        node = prev[node]
    return node if prev[node] is None and node != start else node


def _find_skill(skills, pred):
    for s in skills:
        if pred(s):
            return s
    return None


def _zeros(n):
    return [0.0] * n


def policy(observation):
    goal = observation["goal"]
    atoms = observation["atoms"]
    objects = observation["objects"]
    skills = observation["skills"]

    atom_set = set(atoms)
    atoms_by_pred = {}
    for a in atoms:
        name, args = _parse_atom(a)
        atoms_by_pred.setdefault(name, []).append(args)

    obj_features = {o["name"]: o.get("features", {}) for o in objects}
    throw_param = _throw_param_for(_pile_num_pickups(objects))

    def fallback():
        s = skills[0]
        return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}

    if not skills:
        return {"skill_index": 0, "params": []}

    # Identify the (single) robot and its room.
    robot = None
    robot_room = None
    for args in atoms_by_pred.get("RobotInRoom", []):
        robot, robot_room = args[0], args[1]
        break
    if robot is None:
        for o in objects:
            if o["type"] == "robot":
                robot = o["name"]
                break

    # Movement graph via CanMoveRoom.
    edges = {}
    for args in atoms_by_pred.get("CanMoveRoom", []):
        edges.setdefault(args[0], []).append(args[1])

    unsatisfied = [g for g in goal if g not in atom_set]

    holding_trash = None
    for args in atoms_by_pred.get("HoldingTrash", []):
        if args[0] == robot:
            holding_trash = args[1]
            break
    holding_recycling = None
    for args in atoms_by_pred.get("HoldingRecycling", []):
        if args[0] == robot:
            holding_recycling = args[1]
            break

    # ---- Case: holding trash -> deliver it. ----
    if holding_trash is not None:
        t = holding_trash
        target_bin = None
        for g in unsatisfied:
            name, args = _parse_atom(g)
            if name == "TrashInBin" and args[0] == t:
                target_bin = args[1]
                break
        if target_bin is None:
            for g in goal:
                name, args = _parse_atom(g)
                if name == "TrashInBin" and args[0] == t:
                    target_bin = args[1]
                    break

        if target_bin is not None:
            bin_room = None
            for args in atoms_by_pred.get("TrashBinInRoom", []):
                if args[0] == target_bin:
                    bin_room = args[1]
                    break

            if bin_room is not None and robot_room == bin_room:
                bin_empty = "TrashBinEmpty({})".format(target_bin) in atom_set
                if bin_empty:
                    s = _find_skill(
                        skills,
                        lambda s: s["name"] == "ThrowTrash"
                        and len(s["objects"]) >= 3
                        and s["objects"][1] == t
                        and s["objects"][2] == target_bin,
                    )
                    if s is not None:
                        params = _zeros(s["param_dim"])
                        if params:
                            params[0] = float(throw_param)
                        return {"skill_index": s["index"], "params": params}
                else:
                    occupant = None
                    for args in atoms_by_pred.get("TrashInBin", []):
                        if args[1] == target_bin:
                            occupant = args[0]
                            break
                    s = None
                    if occupant is not None:
                        s = _find_skill(
                            skills,
                            lambda s: s["name"] == "PressTrash"
                            and len(s["objects"]) >= 5
                            and s["objects"][3] == target_bin
                            and s["objects"][4] == occupant,
                        )
                    if s is None:
                        s = _find_skill(
                            skills,
                            lambda s: s["name"] == "PressTrash"
                            and len(s["objects"]) >= 4
                            and s["objects"][3] == target_bin,
                        )
                    if s is not None:
                        return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
            elif bin_room is not None and robot_room is not None:
                nxt = _bfs_next_step(robot_room, {bin_room}, edges)
                if nxt is not None:
                    s = _find_skill(
                        skills,
                        lambda s: s["name"] == "MoveRoom"
                        and len(s["objects"]) >= 3
                        and s["objects"][1] == robot_room
                        and s["objects"][2] == nxt,
                    )
                    if s is not None:
                        return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        return fallback()

    # ---- Case: holding recycling -> deliver it. ----
    if holding_recycling is not None:
        r = holding_recycling
        target_bin = None
        for g in unsatisfied:
            name, args = _parse_atom(g)
            if name == "RecyclingInBin" and args[0] == r:
                target_bin = args[1]
                break
        if target_bin is None:
            for g in goal:
                name, args = _parse_atom(g)
                if name == "RecyclingInBin" and args[0] == r:
                    target_bin = args[1]
                    break

        if target_bin is not None:
            bin_room = None
            for args in atoms_by_pred.get("RecyclingBinInRoom", []):
                if args[0] == target_bin:
                    bin_room = args[1]
                    break

            if bin_room is not None and robot_room == bin_room:
                bin_empty = "RecyclingBinEmpty({})".format(target_bin) in atom_set
                if bin_empty:
                    s = _find_skill(
                        skills,
                        lambda s: s["name"] == "ThrowRecycling"
                        and len(s["objects"]) >= 3
                        and s["objects"][1] == r
                        and s["objects"][2] == target_bin,
                    )
                    if s is not None:
                        params = _zeros(s["param_dim"])
                        if params:
                            params[0] = float(throw_param)
                        return {"skill_index": s["index"], "params": params}
                else:
                    occupant = None
                    for args in atoms_by_pred.get("RecyclingInBin", []):
                        if args[1] == target_bin:
                            occupant = args[0]
                            break
                    s = None
                    if occupant is not None:
                        s = _find_skill(
                            skills,
                            lambda s: s["name"] == "PressRecycling"
                            and len(s["objects"]) >= 5
                            and s["objects"][3] == target_bin
                            and s["objects"][4] == occupant,
                        )
                    if s is None:
                        s = _find_skill(
                            skills,
                            lambda s: s["name"] == "PressRecycling"
                            and len(s["objects"]) >= 4
                            and s["objects"][3] == target_bin,
                        )
                    if s is not None:
                        return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
            elif bin_room is not None and robot_room is not None:
                nxt = _bfs_next_step(robot_room, {bin_room}, edges)
                if nxt is not None:
                    s = _find_skill(
                        skills,
                        lambda s: s["name"] == "MoveRoom"
                        and len(s["objects"]) >= 3
                        and s["objects"][1] == robot_room
                        and s["objects"][2] == nxt,
                    )
                    if s is not None:
                        return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        return fallback()

    # ---- Case: hand empty -> pick up something still needed. ----
    needed_trash = []
    needed_recycling = []
    for g in unsatisfied:
        name, args = _parse_atom(g)
        if name == "TrashInBin":
            needed_trash.append(args[0])
        elif name == "RecyclingInBin":
            needed_recycling.append(args[0])

    if needed_trash:
        t = needed_trash[0]
        s = _find_skill(
            skills,
            lambda s: s["name"] == "PickupTrash"
            and len(s["objects"]) >= 2
            and s["objects"][1] == t,
        )
        if s is not None:
            return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        pile_rooms = {args[1] for args in atoms_by_pred.get("PileInRoom", [])}
        if robot_room is not None and pile_rooms:
            nxt = _bfs_next_step(robot_room, pile_rooms, edges)
            if nxt is not None:
                s = _find_skill(
                    skills,
                    lambda s: s["name"] == "MoveRoom"
                    and len(s["objects"]) >= 3
                    and s["objects"][1] == robot_room
                    and s["objects"][2] == nxt,
                )
                if s is not None:
                    return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        return fallback()

    if needed_recycling:
        r = needed_recycling[0]
        s = _find_skill(
            skills,
            lambda s: s["name"] == "PickupRecycling"
            and len(s["objects"]) >= 2
            and s["objects"][1] == r,
        )
        if s is not None:
            return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        pile_rooms = {args[1] for args in atoms_by_pred.get("PileInRoom", [])}
        if robot_room is not None and pile_rooms:
            nxt = _bfs_next_step(robot_room, pile_rooms, edges)
            if nxt is not None:
                s = _find_skill(
                    skills,
                    lambda s: s["name"] == "MoveRoom"
                    and len(s["objects"]) >= 3
                    and s["objects"][1] == robot_room
                    and s["objects"][2] == nxt,
                )
                if s is not None:
                    return {"skill_index": s["index"], "params": _zeros(s["param_dim"])}
        return fallback()

    return fallback()
