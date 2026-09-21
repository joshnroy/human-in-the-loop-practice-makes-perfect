# Online skill beliefs

The current Tossing3D method selects the A/B competence models and particle/grid
inference described in [the competence experiment](tossing3d-competence-2x2.md).
Its execution-cost filter is independent and unchanged. The older standalone
particle model, retained for numerical consumers, has joint competence p,
learning rate eta, and execution cost c with initial marginals

- p ~ Beta(10, 1), with mean 10/11 and support [0, 1];
- eta ~ Uniform(0, 1); and
- c ~ 20 Beta(1, 9), with mean 2 and support [0, 20].

The marginals are initialized independently with randomized stratified samples
and equal weights, then retained as joint particles so observations may create
posterior dependence. The original EES formulation specifies Beta(10, 1) for
competence; it does not introduce latent learning-rate or cost priors. The eta
and cost priors are therefore implementation choices for this belief-space
extension. Human-reset competence and cost are inferred like every other skill,
not held fixed.

A policy success reweights each hypothesis by p; a failure by 1-p.
The practice model forecasts the next session's competence as
clip(p + eta n, 0, 1) after n practice examples. It does not force a positive eta.
Across session boundaries, stationary outcomes can favor eta=0. Within a
session, competence evidence updates immediately, but improvement is deferred
to the boundary, matching the existing toss refit schedule.

PickCube, OpenGripper, and reset controllers receive S/F and cost observations but
no learning credit: no fitting step can change those controllers. Toss retains
every sampler-training row, including epsilon-random examples, while excluding
epsilon-random outcomes from policy-competence evidence. Evaluation outcomes are
not training data.

The toss forecast carries observed and last-fitted positive/negative label counts.
With no data or one label class, the real sampler bypasses epsilon and chooses a
uniform candidate; hypothetical execution uses that same mode. One-class refits
cannot change the policy and therefore preserve the competence posterior. Their
examples remain retained and become learning credit at the first mixed-class
refit. Further refits use only new examples. This preserves the value of reaching
the first informative fit without awarding improvement to an unchanged policy.
Within a practice cycle, hypothetical new labels affect deployment after fitting,
not the sampling mode of the next practice action.

Mixed-class fitting is necessary but not sufficient for a discriminating sampler:
a particular candidate batch can still have tied scores. The symbolic forecast
does not model those candidate-specific ties. It does suppress the epsilon branch
when that branch is provably impossible under the one-class shortcut.

The particle approximation is a modeling assumption, not a guarantee of
calibration. Constant performance does not prove an exactly zero learning rate;
in particular, learning rate is unidentifiable at competence 1, where all rates
predict zero further improvement. Tests cover stationary and always-successful
observations rather than hardcoding a skill's resulting estimate.

Tossing3D search integrates deployment value over the represented competence
posteriors directly, including failed OpenGripper outcomes, without drawing fresh
theta samples for each stopping value. Expectimax averages every modeled practice
outcome and its configured observation penalty; determinized search instead
follows one sampled successor per state/action edge. The human reset cost is still
charged normally.

## Validation limits

During the first session, hypotheses with equal competence and different learning
rates have identical likelihoods. Their relative learning-rate weights therefore
cannot change until observations arrive after a session-boundary update. Multiple
independent one-session seeds validate competence updates and execution, not
identification of a zero learning rate.

Each practice session starts with accumulated cost C=0. Every robot and reset
skill supplies its own cost. Linear G subtracts lambda C from expected deployment
value; hard-budget G rejects C above B. Replanning does not reset C. Learned
samplers, data, and beliefs persist across sessions; evaluation does not reset C.

Earlier fixed-pick/open experiment results and videos describe the previous
model and are not validation of these online beliefs.
