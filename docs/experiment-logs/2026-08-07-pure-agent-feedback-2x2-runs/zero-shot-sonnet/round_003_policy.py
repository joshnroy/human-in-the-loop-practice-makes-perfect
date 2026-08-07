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
    # Three prior attempts -- raw throw_distance, throw_distance * weight, and
    # 0.0 -- all failed on every single throw across three rounds (0/19, 0/19,
    # 0/2). Wildly different magnitudes all failing completely rules out both
    # "match the raw feature value" and "small delta near zero". That pattern
    # is consistent with a normalized [0, 1] parameter space where 0.0 is the
    # empty/no-throw end of the range; try the opposite end, full strength.
    return [1.0] * n


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


def subgoal_choice(pool, unmet, atoms_set, robot_name):
    pool_by_name = {}
    for s in pool:
        pool_by_name.setdefault(s['name'], []).append(s)

    current_room = find_arg1_by_arg0('RobotInRoom', robot_name, atoms_set)

    for g in unmet:
        pred, args = parse_atom(g)
        if pred == 'TrashInBin':
            item_obj, bin_obj = args
            hold_atom = f"HoldingTrash({robot_name}, {item_obj})"
            pickup_name, throw_name, in_bin_pred, bin_room_pred = (
                'PickupTrash', 'ThrowTrash', 'TrashInBin', 'TrashBinInRoom')
        elif pred == 'RecyclingInBin':
            item_obj, bin_obj = args
            hold_atom = f"HoldingRecycling({robot_name}, {item_obj})"
            pickup_name, throw_name, in_bin_pred, bin_room_pred = (
                'PickupRecycling', 'ThrowRecycling', 'RecyclingInBin', 'RecyclingBinInRoom')
        else:
            continue

        if hold_atom in atoms_set:
            target_room = find_arg1_by_arg0(bin_room_pred, bin_obj, atoms_set)
        else:
            cands = [s for s in pool_by_name.get(pickup_name, [])
                     if s['objects'][1] == item_obj]
            if cands:
                return cands[0]
            if pool_by_name.get(pickup_name):
                return pool_by_name[pickup_name][0]
            pile_info = find_first('PileInRoom', atoms_set)
            target_room = pile_info[1] if pile_info else None

        if target_room is None or current_room is None or target_room == current_room:
            continue

        next_room = bfs_next(current_room, target_room, atoms_set)
        if next_room is None:
            continue
        move_cands = [s for s in pool_by_name.get('MoveRoom', [])
                      if s['objects'][1] == current_room and s['objects'][2] == next_room]
        if move_cands:
            return move_cands[0]

    return None


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
            chosen = subgoal_choice(pool, unmet, atoms, robot_name)

    if chosen is None:
        chosen = min(pool, key=lambda s: s['index'])

    params = get_params(chosen, objects_by_name)
    return {"skill_index": chosen['index'], "params": params}
