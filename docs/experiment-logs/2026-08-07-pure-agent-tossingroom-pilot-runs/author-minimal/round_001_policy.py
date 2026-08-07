import re
from collections import deque


def _parse_atom(atom):
    m = re.match(r"(\w+)\(([^)]*)\)", atom)
    if not m:
        return atom, []
    name = m.group(1)
    inner = m.group(2).strip()
    args = [a.strip() for a in inner.split(",")] if inner else []
    return name, args


def policy(observation):
    atoms = observation["atoms"]
    goal = observation["goal"]
    skills = observation["skills"]
    objects = {o["name"]: o for o in observation["objects"]}

    atom_set = set(atoms)
    missing = [g for g in goal if g not in atom_set]

    by_pred = {}
    for a in atoms:
        name, args = _parse_atom(a)
        by_pred.setdefault(name, []).append(args)

    robot_name = None
    robot_room = None
    for args in by_pred.get("RobotInRoom", []):
        robot_name, robot_room = args[0], args[1]
        break

    missing_trash = []
    missing_recycling = []
    for g in missing:
        name, args = _parse_atom(g)
        if name == "TrashInBin":
            missing_trash.append((args[0], args[1]))
        elif name == "RecyclingInBin":
            missing_recycling.append((args[0], args[1]))

    missing_trash_objs = {t for t, b in missing_trash}
    missing_recycling_objs = {r for r, b in missing_recycling}

    def find_skill(name, filt):
        for s in skills:
            if s["name"] == name and filt(s):
                return s
        return None

    def throw_distance_param(bin_name):
        feats = objects.get(bin_name, {}).get("features", {})
        return float(feats.get("throw_distance", 0.0))

    holding_trash = {a[1] for a in by_pred.get("HoldingTrash", [])}
    holding_recycling = {a[1] for a in by_pred.get("HoldingRecycling", [])}

    # 1. Throw held items that satisfy a missing goal.
    for t, b in missing_trash:
        if t in holding_trash:
            s = find_skill(
                "ThrowTrash",
                lambda s, t=t, b=b: s["objects"][1] == t and s["objects"][2] == b,
            )
            if s:
                return {"skill_index": s["index"], "params": [throw_distance_param(b)]}

    for r, b in missing_recycling:
        if r in holding_recycling:
            s = find_skill(
                "ThrowRecycling",
                lambda s, r=r, b=b: s["objects"][1] == r and s["objects"][2] == b,
            )
            if s:
                return {"skill_index": s["index"], "params": [throw_distance_param(b)]}

    # 2. If the bin we need is currently full, press its button to empty it.
    for t, b in missing_trash:
        bin_empty = any(a[0] == b for a in by_pred.get("TrashBinEmpty", []))
        if not bin_empty:
            s = find_skill("PressTrash", lambda s, b=b: s["objects"][3] == b)
            if s:
                return {"skill_index": s["index"], "params": []}

    for r, b in missing_recycling:
        bin_empty = any(a[0] == b for a in by_pred.get("RecyclingBinEmpty", []))
        if not bin_empty:
            s = find_skill("PressRecycling", lambda s, b=b: s["objects"][3] == b)
            if s:
                return {"skill_index": s["index"], "params": []}

    # 3. If hand is empty, pick up something the goal still needs.
    hand_empty = any(a[0] == robot_name for a in by_pred.get("HandEmpty", []))
    if hand_empty:
        for s in skills:
            if s["name"] == "PickupTrash" and s["objects"][1] in missing_trash_objs:
                return {"skill_index": s["index"], "params": []}
        for s in skills:
            if s["name"] == "PickupRecycling" and s["objects"][1] in missing_recycling_objs:
                return {"skill_index": s["index"], "params": []}

    # 4. Otherwise, move toward the room that helps most.
    target_room = None
    if holding_trash:
        for t, b in missing_trash:
            if t in holding_trash:
                for a in by_pred.get("TrashBinInRoom", []):
                    if a[0] == b:
                        target_room = a[1]
                        break
            if target_room:
                break
    elif holding_recycling:
        for r, b in missing_recycling:
            if r in holding_recycling:
                for a in by_pred.get("RecyclingBinInRoom", []):
                    if a[0] == b:
                        target_room = a[1]
                        break
            if target_room:
                break
    elif hand_empty and (missing_trash_objs or missing_recycling_objs):
        for a in by_pred.get("PileInRoom", []):
            target_room = a[1]
            break

    if target_room and robot_room and target_room != robot_room:
        graph = {}
        for a in by_pred.get("CanMoveRoom", []):
            graph.setdefault(a[0], []).append(a[1])

        visited = {robot_room}
        parent = {}
        q = deque([robot_room])
        found = False
        while q:
            cur = q.popleft()
            if cur == target_room:
                found = True
                break
            for nxt in graph.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = cur
                    q.append(nxt)

        if found:
            path = [target_room]
            while path[-1] != robot_room:
                path.append(parent[path[-1]])
            path.reverse()
            if len(path) > 1:
                next_room = path[1]
                for s in skills:
                    if s["name"] == "MoveRoom" and s["objects"][2] == next_room:
                        return {"skill_index": s["index"], "params": []}

    # 5. Fallback: take whatever progress is available, deterministically.
    for s in skills:
        if s["name"] in ("ThrowTrash", "ThrowRecycling"):
            b = s["objects"][2]
            return {"skill_index": s["index"], "params": [throw_distance_param(b)]}
    for s in skills:
        if s["name"] in ("PickupTrash", "PickupRecycling"):
            return {"skill_index": s["index"], "params": []}
    for s in skills:
        if s["name"] == "MoveRoom":
            return {"skill_index": s["index"], "params": []}

    s0 = skills[0]
    return {"skill_index": s0["index"], "params": [0.0] * s0["param_dim"]}
