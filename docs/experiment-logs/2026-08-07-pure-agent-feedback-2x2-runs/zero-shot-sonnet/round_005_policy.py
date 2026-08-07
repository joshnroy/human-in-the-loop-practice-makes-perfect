"""Greedy planner for the trash/recycling robot domain."""
from collections import deque


def parse_atom(atom):
    i = atom.index('(')
    pred = atom[:i]
    inner = atom[i + 1:-1]
    args = tuple(a.strip() for a in inner.split(',')) if inner else tuple()
    return pred, args


EFFECT_SLOTS = {
    'PickupTrash': lambda o: (
        {f"HoldingTrash({o[0]}, {o[1]})"},
        {f"HandEmpty({o[0]})"},
    ),
    'PickupRecycling': lambda o: (
        {f"HoldingRecycling({o[0]}, {o[1]})"},
        {f"HandEmpty({o[0]})"},
    ),
    'MoveRoom': lambda o: (
        {f"RobotInRoom({o[0]}, {o[2]})"},
        {f"RobotInRoom({o[0]}, {o[1]})"},
    ),
    'ThrowTrash': lambda o: (
        {f"HandEmpty({o[0]})", f"TrashInBin({o[1]}, {o[2]})"},
        {f"HoldingTrash({o[0]}, {o[1]})", f"TrashBinEmpty({o[2]})"},
    ),
    'ThrowRecycling': lambda o: (
        {f"HandEmpty({o[0]})", f"RecyclingInBin({o[1]}, {o[2]})"},
        {f"HoldingRecycling({o[0]}, {o[1]})", f"RecyclingBinEmpty({o[2]})"},
    ),
    'PressTrash': lambda o: (
        {f"TrashBinEmpty({o[3]})"},
        {f"TrashInBin({o[4]}, {o[3]})"},
    ),
    'PressRecycling': lambda o: (
        {f"RecyclingBinEmpty({o[3]})"},
        {f"RecyclingInBin({o[4]}, {o[3]})"},
    ),
}


def compute_effects(skill):
    fn = EFFECT_SLOTS.get(skill['name'])
    if fn is None:
        return set(), set()
    return fn(skill['objects'])


def get_params(skill, objects_by_name):
    n = skill['param_dim']
    if n == 0:
        return []
    # Four prior attempts -- raw throw_distance, throw_distance * weight, 0.0,
    # and 1.0 -- have now all failed on every single throw (0/38, 0/2 totals).
    # Both endpoints of a plausible normalized [0, 1] range failing identically
    # rules out "min effort" and "max effort" alike, and argues against a
    # state-dependent target too (two very different state-derived formulas
    # both missed uniformly). Bisect toward the middle of the range next.
    return [0.5] * n


def find_first(pred, atoms_set):
    for a in atoms_set:
        p, args = parse_atom(a)
        if p == pred:
            return args
    return None


def find_arg1_by_arg0(pred, arg0, atoms_set):
    for a in atoms_set:
        p, args = parse_atom(a)
        if p == pred and len(args) > 1 and args[0] == arg0:
            return args[1]
    return None


def bfs_next(start, goal_room, atoms_set):
    if start == goal_room:
        return start
    adj = {}
    for a in atoms_set:
        p, args = parse_atom(a)
        if p == 'CanMoveRoom' and len(args) == 2:
            adj.setdefault(args[0], []).append(args[1])
    parent = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur == goal_room:
            break
        for nxt in adj.get(cur, []):
            if nxt not in parent:
                parent[nxt] = cur
                dq.append(nxt)
    if goal_room not in parent:
        return None
    node = goal_room
    path = [node]
    while parent[node] is not None:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path[1] if len(path) >= 2 else None


def room_from_feature(obj_name, objects_by_name):
    obj = objects_by_name.get(obj_name)
    if obj is None:
        return None
    idx = obj.get('features', {}).get('room')
    if idx is None:
        return None
    for name, o in objects_by_name.items():
        if o.get('type') == 'room' and o.get('features', {}).get('index') == idx:
            return name
    return None


def parsed_unmet(unmet):
    """Split unmet goal atoms into (pred, item_obj, bin_obj) tuples we know how
    to act on, tagged with the relevant predicate names."""
    parsed = []
    for g in unmet:
        pred, args = parse_atom(g)
        if pred == 'TrashInBin' and len(args) == 2:
            parsed.append((args[0], args[1], 'HoldingTrash', 'PickupTrash',
                            'TrashBinInRoom'))
        elif pred == 'RecyclingInBin' and len(args) == 2:
            parsed.append((args[0], args[1], 'HoldingRecycling', 'PickupRecycling',
                            'RecyclingBinInRoom'))
    return parsed


def move_toward(pool_by_name, current_room, target_room, atoms_set):
    if target_room is None or current_room is None or target_room == current_room:
        return None
    next_room = bfs_next(current_room, target_room, atoms_set)
    if next_room is None:
        return None
    for s in pool_by_name.get('MoveRoom', []):
        if s['objects'][1] == current_room and s['objects'][2] == next_room:
            return s
    return None


def subgoal_choice(pool, unmet, atoms_set, robot_name, objects_by_name):
    pool_by_name = {}
    for s in pool:
        pool_by_name.setdefault(s['name'], []).append(s)

    current_room = find_arg1_by_arg0('RobotInRoom', robot_name, atoms_set)
    entries = parsed_unmet(unmet)

    # Pass 1: finish delivering whatever the robot is already holding before
    # getting pulled toward a different unmet goal (avoids abandoning a
    # carried item mid-delivery to chase the other item's pickup).
    for item_obj, bin_obj, hold_pred, _pickup_name, bin_room_pred in entries:
        hold_atom = f"{hold_pred}({robot_name}, {item_obj})"
        if hold_atom not in atoms_set:
            continue
        target_room = find_arg1_by_arg0(bin_room_pred, bin_obj, atoms_set)
        if target_room is None:
            target_room = room_from_feature(bin_obj, objects_by_name)
        mv = move_toward(pool_by_name, current_room, target_room, atoms_set)
        if mv is not None:
            return mv

    # Pass 2: nothing currently held is actionable; go pick something up.
    for item_obj, _bin_obj, hold_pred, pickup_name, _bin_room_pred in entries:
        hold_atom = f"{hold_pred}({robot_name}, {item_obj})"
        if hold_atom in atoms_set:
            continue
        cands = [s for s in pool_by_name.get(pickup_name, [])
                 if s['objects'][1] == item_obj]
        if cands:
            return cands[0]
        pile_info = find_first('PileInRoom', atoms_set)
        target_room = pile_info[1] if pile_info else room_from_feature(item_obj, objects_by_name)
        mv = move_toward(pool_by_name, current_room, target_room, atoms_set)
        if mv is not None:
            return mv

    return None


def fallback_choice(pool):
    for name in ('ThrowTrash', 'ThrowRecycling', 'PickupTrash', 'PickupRecycling', 'MoveRoom'):
        cands = [s for s in pool if s['name'] == name]
        if cands:
            return min(cands, key=lambda s: s['index'])
    non_press = [s for s in pool if not s['name'].startswith('Press')]
    if non_press:
        return min(non_press, key=lambda s: s['index'])
    return min(pool, key=lambda s: s['index'])


def policy(observation):
    atoms = set(observation['atoms'])
    goal = list(observation['goal'])
    skills = sorted(observation['skills'], key=lambda s: s['index'])
    objects_by_name = {o['name']: o for o in observation['objects']}

    robot_name = None
    for o in observation['objects']:
        if o['type'] == 'robot':
            robot_name = o['name']
            break
    if robot_name is None or not any(
            a.startswith('RobotInRoom(') and a.split('(')[1].split(',')[0].strip() == robot_name
            for a in atoms):
        for a in atoms:
            p, args = parse_atom(a)
            if p == 'RobotInRoom' and args:
                robot_name = args[0]
                break

    unmet = [g for g in goal if g not in atoms]
    satisfied_goal = set(goal) - set(unmet)

    effects = {s['index']: compute_effects(s) for s in skills}

    def loss(s):
        _, dels = effects[s['index']]
        return len(dels & satisfied_goal)

    pool = [s for s in skills if loss(s) == 0]
    if not pool:
        pool = skills

    chosen = None
    if unmet:
        unmet_set = set(unmet)
        best = max(pool, key=lambda s: len(effects[s['index']][0] & unmet_set))
        if len(effects[best['index']][0] & unmet_set) > 0:
            chosen = best
        elif robot_name is not None:
            chosen = subgoal_choice(pool, unmet, atoms, robot_name, objects_by_name)

    if chosen is None:
        chosen = fallback_choice(pool)

    params = get_params(chosen, objects_by_name)
    return {"skill_index": chosen['index'], "params": params}
