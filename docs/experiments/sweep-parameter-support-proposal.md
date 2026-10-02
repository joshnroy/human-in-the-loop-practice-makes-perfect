# Proposed learnable Sweep parameter supports — awaiting Josh

The pinned kinder-baselines commit427ad6cd fixes all stock sampler bounds to points (`dynamic3d/utils.py:78–83`): OpenDrawer distance0.8m, PickWiper0.7m, Sweep0.55m, all rotation−π. Therefore forwarding two parameter slots is insufficient: the inherited samplers cannot learn a different proposal.

Separate fixed-controller diagnostics exposed this before any first-seed experiment batch: OpenDrawer0/24, PickWiper24/24, Sweep8/13 completed attempts (13/24 contexts before interruption). Each skill used1/1 distinct parameter vector. These are not valid random-competence calibration estimates and must not populate a frozen learning manifest. Raw plans, attempts, states and replay records remain under `scratchpad/sweep-integration/random-calibration-*`; the diagnosis is `degenerate-support-diagnosis.json`.

## Concrete proposed change

| Skill | Proposed uniform distance | Proposed uniform rotation |
|---|---:|---:|
| OpenDrawer |[0.65,0.95]m|[−13π/12,−11π/12]rad|
| PickWiper |[0.55,0.85]m|[−13π/12,−11π/12]rad|
| Sweep |[0.40,0.70]m|[−13π/12,−11π/12]rad|

This is ±0.15m and ±15° around the existing nominal approach. It is a proposed scientific sampling-support choice, not an upstream advertised range or validated success region. Store it in the local experiment manifest/provider; do not mutate lab repositories or global upstream constants. Apply the same supports to EES and ModelB, all costs, all seeds, and evaluation samplers. Keep environment distribution, skill controllers, goal predicate, action limits and collision/planning checks unchanged. Invalid trajectories remain observed controller failures; no resampling until a skill succeeds and no outcome-based seed filtering.

## Geometric provenance and limits

`dynamic3d/utils.py:453` computes the base pose at distance r from the target, at angle `target_yaw+rotation`, facing the target. OpenDrawer uses the wiper reference shifted+0.3m in y (`sweep3D/parameterized_skills.py:151–160`); PickWiper uses the wiper pose (`:521–525`); Sweep uses cube0 position with zero reference yaw (`:835–845`). The same controller then applies its existing base planner/world bounds [−2.5,2.5]m and arm planning checks. The proposed neighborhood is finite, positive-radius, and within the world box for declared valid start poses. This does **not** prove collision-free reachability or successful contact, especially after practice changes object geometry.

The saved OpenDrawer diagnostics finish with drawer displacement0.00000m,0.09517m,0.12012m for seeds0,1,2, below the current0.15m symbolic-open criterion. This is observed failure/partial progress, not grounds to lower the criterion to inflate success. Parameter learning must be validated physically after the support is approved.

## Checks after approval, before launch

1. Reject missing/degenerate support during learning preflight; verify nonidentical proposals and exact bounds.
2. Use a fixed predeclared geometric feasibility/calibration set, preserving every failure and denominator. Do not tune intervals based on first-seed experiment results.
3. Freeze the chosen intervals and random-exploration competence estimates before scoring the first-seed experiment batch. If the fixed set shows the controller cannot perform its task, diagnose controller readiness instead of changing the goal or forcing success.
4. Retain the other launch gates: valid starts, physical recovery, matching human/robot reset conditions, exact adequate-depth search, source/pin provenance and measured reset accounting.

No new support is implemented or measured yet. Josh's approval is required for these numerical ranges.
