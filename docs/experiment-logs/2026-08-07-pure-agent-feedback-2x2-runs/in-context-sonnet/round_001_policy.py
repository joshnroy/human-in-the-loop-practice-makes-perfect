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
single float. Its meaning is not documented, and setting it to the target
bin's "throw_distance" feature turned out to always fail (0/19 in practice,
regardless of the thrown item's weight). Since the policy is called every
environment step and the environment re-exposes the *same* state (still
holding the item, bin still empty) after a failed throw, we can treat this
as a small online search: remember, across calls, which parameter values
have already been tried for the current (item, bin, item-weight) situation,
and sweep through a list of candidates -- smart guesses derived from the
object features first, then a fine-grained numeric sweep -- advancing to the
next candidate each time we find ourselves needing to throw the same item
into the same bin again (a sign the previous attempt did not succeed).
"""

from collections import deque

# Persists across policy() calls within a single evaluation process, so a
# failed throw with one parameter value leads to a different value being
# tried the next time the same (item, bin, weight) situation is seen.
_THROW_PARAM_STATE = {}


def _throw_param_candidates(weight, throw_distance):
    # `throw_distance` alone (the bin's own feature, e.g. 2.0) was tried as
    # the very first hypothesis and failed 19/19 times in practice across a
    # wide range of item weights (0.5-1.5) -- so it is *not* the right value
    # and is de-prioritized below in favor of weight-dependent guesses,
    # which remain consistent with that evidence (a fixed 2.0 could easily
    # miss a per-item, weight-dependent target every time).
    smart = [
        weight,
        0.0,
        max(throw_distance - weight, 0.0),
        throw_distance * weight,
        1.0,
        (throw_distance / weight) if weight else throw_distance,
        0.5,
        weight * 2.0,
        throw_distance + weight,
        throw_distance,
    ]
    sweep = [round(i * 0.25, 3) for i in range(0, 41)]  # 0.0 .. 10.0
    seen = set()
    out = []
    for v in smart + sweep:
        v = round(float(v), 3)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _next_throw_param(kind, item, target_bin, weight, throw_distance):
    key = (kind, item, target_bin, round(float(weight), 4))
    idx = _THROW_PARAM_STATE.get(key, 0)
    candidates = _throw_param_candidates(weight, throw_distance)
    value = candidates[idx % len(candidates)]
    _THROW_PARAM_STATE[key] = idx + 1
    return value


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
                        dist = obj_features.get(target_bin, {}).get(
                            "throw_distance", 0.0
                        )
                        weight = obj_features.get(t, {}).get("weight", 0.0)
                        params = _zeros(s["param_dim"])
                        if params:
                            params[0] = float(
                                _next_throw_param(
                                    "trash", t, target_bin, weight, dist
                                )
                            )
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
                        dist = obj_features.get(target_bin, {}).get(
                            "throw_distance", 0.0
                        )
                        weight = obj_features.get(r, {}).get("weight", 0.0)
                        params = _zeros(s["param_dim"])
                        if params:
                            params[0] = float(
                                _next_throw_param(
                                    "recycling", r, target_bin, weight, dist
                                )
                            )
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
