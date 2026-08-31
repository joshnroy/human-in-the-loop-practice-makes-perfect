# Online skill beliefs

Every robot skill has an independent posterior over competence p and learning
rate eta. The shared discrete prior is uniform over p in {0, .25, .5, .75, 1}
and eta in {0, .1}. Endpoints let data support always-successful or
always-failing stationary skills; they are not fixed assumptions about any
particular robot skill. Only human reset retains a known success probability.

A policy success reweights each hypothesis by p; a failure by 1-p.
The practice model forecasts the next session's competence as
1 - (1-p)(1-eta)^n after n practice examples. It does not force a positive eta.
Across session boundaries, stationary outcomes can favor eta=0. Within a
session, competence evidence updates immediately, but improvement is deferred
to the boundary, matching the existing toss refit schedule.

PickCube and OpenGripper count each observed practice attempt as an example.
Their controllers remain parameter-free: advancing their hypothesized learning
curve is a prediction to test against subsequent observations, not a real
controller update. Toss retains its sampler-training-row accounting and does
not treat random exploration outcomes as greedy-policy competence evidence.
Evaluation outcomes are not training data.

The finite hypothesis family is a modeling assumption, not a guarantee of
calibration. Constant performance does not prove an exactly zero learning rate;
in particular, learning rate is unidentifiable at competence 1, where all rates
predict zero further improvement. Tests cover stationary and always-successful
observations rather than hardcoding a skill's resulting estimate.

Search samples a joint theta across the three robot skills. Both practice
transitions and deployment evaluation use these estimates, including failed
OpenGripper outcomes. The human reset cost is still charged normally.

## Validation limits

During the first session, hypotheses with equal competence and different learning
rates have identical likelihoods. Their relative learning-rate weights therefore
cannot change until observations arrive after a session-boundary update. Multiple
independent one-session seeds validate competence updates and execution, not
identification of a zero learning rate.

The accumulated practice cost is not reset at session boundaries. A hard budget
currently applies to the entire training run; it is not replenished each cycle.

Earlier fixed-pick/open experiment results and videos describe the previous
model and are not validation of these online beliefs.
