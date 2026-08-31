# Implemented specialization versus the formulation

Source: the Formulation Notes for Josh PDF, sections 1–5, and
understanding/pomdp_formulation.py. This describes the online-belief implementation,
not the older fixed Pick/Open prototype.

## Preserved structure

The state includes physical state, cumulative cost, and belief. STOP competes with
applicable skills. Chance branches update the belief before recursion, accumulate
cost, and weight successor values by predictive probability. Real execution takes
one skill and replans. Independent per-skill beliefs are an allowed factorization
in PDF section 2.4.

## Fixed or simplified

- Only Tossing3D's eight hand-written symbolic phases and three robot skills plus
  human cube/bin reset are modeled. This is not a generic KINDER environment plugin.
  Search applicability is narrower than the raw skill initiation sets.
- Robot competence is one scalar per skill, not state-conditioned. Detailed poses
  influence the real controller but not the POMDP competence likelihood.
- Each robot's discrete prior is uniform over p={0,.25,.5,.75,1} and eta={0,.1}.
  The saturating improvement law is fixed, depends only on example count, and
  ignores which states or outcomes make learning more effective.
- Human reset competence is fixed at 1. Costs are configured deterministic
  constants, all 1 in this experiment. Theta contains no cost latent; timing logs
  are not fed into cost estimation.
- Random-toss exploration has fixed modeled success .25 and epsilon .5. Random
  toss outcomes can supply training rows but are not greedy-policy competence
  observations. The actual uninformative sampler can differ from that idealized
  mixture. The PDF explicitly identifies exploration/deployment mismatch as open.
- Success/failure drives robot competence inference, rather than the full
  transition-and-cost likelihood. This is the PDF's simplified Bernoulli case,
  but relies on the hand-written success/failure successor model being adequate.
- Learning-curve advancement is deferred to session boundaries. Pick/Open remain
  parameter-free controllers; inferred improvement is a forecast, not a real
  controller update. The PDF allows batched updates via pending data counters.
  The user accepted this timing for these experiments.
- During the first session, equal-p hypotheses with different eta have equal
  likelihoods. These ten independent single-session seeds cannot establish eta=0;
  competence updates immediately, but eta identification requires later-session
  observations. At p=1, all rates already predict zero further gain.
- Deployment utility is a four-skill, goal-success surrogate from a fixed READY
  state. It does not call the actual Fast Downward task-distribution planner for
  each theta sample. Human reset is excluded from modeled deployment. This is a
  choice of g, not the general optimal frozen-MDP solve suggested by the PDF and
  pseudocode. Deployment resource cost and discounting are not included.
- Hard budget B=150 implements the PDF's hard-budget G by excluding deterministic
  over-budget actions. Linear cost penalty is inactive. Cost accumulates across
  replans within a session; each new session starts at C=0 with a fresh B.
  This chooses a separate practice objective per session; the PDF leaves
  multi-session scheduling unspecified. Previously B was shared across cycles.
  That change does not affect the single-session experiments described here.
- H=10 is truncated receding-horizon planning. With unit costs and B=150, the PDF's
  initial effective horizon bound is 150. Consequently this configuration does
  not inherit the PDF's exact optimality guarantee.
- STOP value is estimated with 100 joint-theta samples per unique search state.
  Successor branches are enumerated, not sampled trajectories. Fresh searches
  resample; caching is local to a search. Noisy values and repeated maximization
  can change choices, so online replanning need not match a fixed offline tree.
- One practice session is capped at 150 harness iterations. If it executes all
  150 paid skills there is no extra iteration to select STOP and export a STOP
  tree. Before/after evaluation is 10 tasks each and does not train the learner.

## Differences from the literal pseudocode

The pseudocode compares a running partial chance sum inside the successor loop
and uses <=, so ties replace STOP. The implementation compares only after summing
every successor and uses strict >, retaining STOP on ties. This follows PDF
Algorithm 1 lines 12–14 and fixes the partial-sum issue for negative values.

The implementation adds explicit model injection, finite-value/probability
assertions, a per-search functools cache, floating-point costs, diagnostics,
and runtime sample configuration. The pseudocode's NUM_SAMPLES=1 is a placeholder;
the experiment uses 100.

The physical-state protocol carries a situated successor that also embeds the
updated belief; update_belief_state retrieves it. This is an implementation
convenience, not evidence that physical observations reveal the hidden theta.

## Computational and reporting limits

Full diagnostics retain each state's theta samples and serialized branches in
memory until the decision completes, in addition to the search cache. H=100
memory failures therefore measure both search expansion and diagnostic overhead.
Video charts and HTML trees read the existing trace without drawing more samples.

These runs are behavior/implementation validation. Ten seeds are replications of
one H=10/B=150 configuration, not a parameter sweep, and no comparative or
statistical performance claim is made before analyzing their results.
