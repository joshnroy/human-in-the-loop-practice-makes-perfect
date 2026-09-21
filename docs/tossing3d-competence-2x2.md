# Tossing3D competence experiment

The experiment crosses two competence models with two inference engines. All four
arms use the same simulator, task seed, planner, action budget and cost objective.

| Model | Particle filter | Fixed grid |
| --- | --- | --- |
| A: global learning curve | `global_curve-particle` | `global_curve-grid` |
| B: local competence and learning rate | `local_trend-particle` | `local_trend-grid` |

## Probabilistic models

Both models observe skill success/failure using a Bernoulli likelihood with latent
competence C. Learning rate has no separately observed target: subsequent outcomes
reweight the latent learning hypotheses. Epsilon-random exploration tosses train the
sampler but are excluded from learned-controller competence evidence, as before.

Model A uses cumulative training examples m and static parameters phi:

```text
mu(m) = phi_0 + (phi_1 - phi_0) * (1 - exp(-phi_2 * m))
C_m | phi ~ Beta(1 + mu(m) * (kappa - 2),
                 1 + (1 - mu(m)) * (kappa - 2))
```

Here mu is the Beta mode, not its mean. The reported learning rate is the derivative
of the conditional mean: `(kappa - 2) / kappa * (phi_1 - phi_0) * phi_2 * exp(-phi_2*m)`.
The shared discrete prior has phi_0 and phi_1 in `{0, .1, ..., 1}` subject to
phi_0 <= phi_1, phi_2 in `{.02, .05, .1, .2, .4, .8}`, and kappa in `{6, 24, 96}`.
All 1,188 valid parameter combinations have equal prior probability. The grid
integrates conditional Beta mass over 25 competence bins; particles sample the
same parameter prior and continuous competence.

Model B uses the following transition after n new examples:

```text
C'   = clip(C + n * eta + Normal(0, .03), 0, 1)
eta' = clip(.9**n * eta + Normal(0, .005), 0, .15)
```

Its prior is C ~ Beta(2, 1), eta ~ capped HalfNormal(.05), independently. Both
engines include the clipping mass at the boundaries. The grid uses 25 competence
bins and 16 learning-rate bins with CDF-integrated prior and transition masses.
The particle filter uses 1,024 particles and systematic resampling below 50% ESS.
Within each model, particles and the grid approximate the same prior. Comparing
models also compares their different prior assumptions.

An actual cycle with zero examples advances diagnostic bookkeeping but leaves the
latent distribution unchanged. Hypothetical zero-example refits are pure identity
operations. Forecasts never consume the live inference random stream.

## Costs and smoothing

The cost distribution retains the existing particle filter, prior, Student-t
observation likelihood and resampler in every arm, including the fixed-grid arms.
Ordinary robot skills still cost 1; both reset skills retain the configured reset
cost of 5. The planner retains
the existing linear objective with lambda = .0003. The experiment does not change
how physical outcomes or practice actions are charged.

Cycle-end filtered posteriors are saved before the training transition. Particle
smoothing traces recorded resampling ancestry backwards, carrying terminal weights.
Grid smoothing applies the model's backward transition and subsequent S/F
likelihoods. These retrospective summaries are written to `pomdp_decisions.jsonl`
and never replace online beliefs or feed planning/evaluation.

Particle smoothing can lose early ancestors, and static Model A parameters can
lose support after resampling. Logs retain ESS, unique ancestors, resampling counts
and surviving phi configurations. Abrupt reversals can expose material particle
approximation error. Tiny particle populations can exhaust compatible support;
the configured population is 1,024 and zero-mass posteriors fail explicitly.

## Practice planning and simulator consistency

The initial pilot exposed reset-heavy planning and sparse toss practice. The
current implementation corrects the shared planner and symbolic dynamics before
repeating the same four inference arms. The original pilot remains a separate
measurement at source revision `68dd6a9797fc651922991f4aa3b888d8f7249d1c`.

The practice loop supplies its actual remaining action slots before each decision.
Determinized search respects that bound as well as its unchanged 100-iteration
compute limit. Cache keys distinguish remaining budgets. Logs include the effective
action horizon and selected path depth; the older generic `horizon` argument alone
did not impose a bound on this solver.

Deployment value is integrated exactly over the represented independent skill
distributions after the existing pending-example forecast. Repeated attempts use
mixed moments of the same latent competence, rather than powers of its mean.
This removes fresh Monte Carlo error from objective comparisons without changing
the deployment policy, cost distribution, hard-budget alternative or linear lambda.
Particle approximation and forecast uncertainty still exist.

Both reset mechanisms now delete `Holding` and conditionally add `ClosedEmpty`
when a cube was held; an already open gripper remains open. The classical and
belief-space models agree on these effects. Completed resets have success
probability one, matching the simulator API: errors abort execution rather than
produce a completed failed reset. Hypothetical resets change atoms and accrued
cost only. Actual reset completion still records the same success, cost observation,
training count and cycle refit as before.

A reset whose modeled successor atoms equal its incoming atoms is omitted from
practice search: it consumes a nonnegative cost and an action slot without changing
the represented deployment value or future dynamics. Recovery resets remain
available. This dominance applies within the symbolic abstraction; any benefit of
resampling physical poses that leaves those atoms unchanged is outside this model.

Failed robot skills need not leave the world unchanged. Their symbolic effect
distribution is estimated from actual failed practice executions, separately for
each grounded skill, exact incoming atom set, and epsilon-random versus policy
sampling mode. Empirical frequencies split the existing failure probability; they
add no competence observation, learning-rate target or cost charge. Counts persist
across cycles and stay fixed during each search. Unseen contexts retain the previous
identity-effect approximation, which can be optimistic. No outcomes from the
original pilot initialize this model. Before/after atoms and count snapshots are
logged for independent replay.

## Running the four arms

From a clean checkout with the committed submodule pins and environment installed:

```bash
scripts/with_env.sh python -m scripts.tossing3d_competence_2x2 \
  --results-root /absolute/path/to/new/results-directory
```

Use the repository's memory-limited simulator execution convention on workstations.
The launcher defaults to two concurrent workers, one paired seed (0), ten practice
cycles per arm, ten evaluation tasks per sweep, and eleven evaluation sweeps per
arm. Practice and evaluation use the barrier layout, canonical seed 125, no free
practice resets, and twenty actions per cycle. The determinized A* planner uses
100 search iterations and observation-probability weight .001. The long-range
throw domain is 1.25–2.6 m, 115–420 degrees/s and 400–840 ms release time.

The launcher uses `scripts.run_sweep.SweepRunner` and records commands, the source
commit, environment paths, source diff, per-run `timing.json`, separate process
logs and exit statuses. Commit new source files before a measured
run: the source diff does not include untracked files. Each arm also writes its
configuration snapshot, progress, sampler draws, physical state log, decisions,
evaluation episode traces and final statistics. A successful process exit alone is
insufficient: check ten refit
and smoothing events, eleven evaluation sweeps, and numerical diagnostics.

Validate the completed matrix and export CSV/JSON summaries plus PNG/PDF figures:

```bash
scripts/with_env.sh python scripts/summarize_tossing3d_competence_2x2.py \
  --results-root /absolute/path/to/results-directory
```

The summary command streams the large belief logs and checks cycle completion,
dispatch accounting, sampler counts and retrospective smoothing before plotting.

One seed per arm is a controlled pilot, not an estimate of variation across seeds.
The matched initial tasks do not imply identical training trajectories: model and
inference differences can lead the planner to choose different actions.
