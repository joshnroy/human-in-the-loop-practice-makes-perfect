# Online skill beliefs

Every modeled skill has its own joint particle posterior over competence p,
learning rate eta, and execution cost c. Particle marginals begin as

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

PickCube and OpenGripper count each observed practice attempt as an example.
Their controllers remain parameter-free: advancing their hypothesized learning
curve is a prediction to test against subsequent observations, not a real
controller update. Toss retains its sampler-training-row accounting and does
not treat random exploration outcomes as greedy-policy competence evidence.
Evaluation outcomes are not training data.

The particle approximation is a modeling assumption, not a guarantee of
calibration. Constant performance does not prove an exactly zero learning rate;
in particular, learning rate is unidentifiable at competence 1, where all rates
predict zero further improvement. Tests cover stationary and always-successful
observations rather than hardcoding a skill's resulting estimate.

Search samples a joint theta across all modeled robot and reset skills. Both practice
transitions and deployment evaluation use these estimates, including failed
OpenGripper outcomes. The human reset cost is still charged normally.

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
