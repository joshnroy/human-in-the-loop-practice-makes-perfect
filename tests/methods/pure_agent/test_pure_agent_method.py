import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.method.types import Policy, Skill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.pure_agent.agent_backend import ScriptedAgentBackend
from hitl_pmp.methods.pure_agent.prompts import PromptArm, PromptBuilder
from hitl_pmp.methods.pure_agent.pure_agent_method import PureAgentMethod

# A policy that always takes the first applicable ground skill, with zero parameters.
# Deliberately trivial: these tests pin the *plumbing* (the observation contract, the
# authoring cadence, replay determinism), not whether any authored policy is any good.
FIRST_SKILL = """
def policy(observation):
    skill = observation["skills"][0]
    return {"skill_index": 0, "params": [0.0] * skill["param_dim"]}
"""

# A policy that names a skill outside the applicable list, so it is observably NOT
# FIRST_SKILL from the emitted label alone. Used where a test has to tell one round's
# policy from the next: Light Switch's initial state has exactly one applicable ground
# skill, so "first" and "last" are the same action there and could not distinguish them.
DECLINES = """
def policy(observation):
    return {"skill_index": len(observation["skills"]), "params": []}
"""

# Loads, but raises the moment it is called: the "authored something that runs and then
# misbehaves" branch, which is distinct from failing to import at all.
RAISES_WHEN_CALLED = """
def policy(observation):
    raise ValueError("boom")
"""

# Does not even import.
DOES_NOT_IMPORT = """
this is not python
"""

# Imports, but has no `policy` name at all.
NO_POLICY_FUNCTION = """
def not_the_policy(observation):
    return {"skill_index": 0, "params": []}
"""


def _env() -> LightSwitchEnvironment:
    return LightSwitchEnvironment(grid_size=5)


def _method(
    *,
    env: LightSwitchEnvironment,
    sources: tuple[str | None, ...] = (FIRST_SKILL,),
    prompt_arm: PromptArm = PromptArm.MINIMAL,
    domain_description: str = "",
) -> PureAgentMethod:
    return PureAgentMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        backend=ScriptedAgentBackend(sources=sources),
        prompt_arm=prompt_arm,
        domain_description=domain_description,
    )


def test_authors_a_policy_lazily_on_first_use_and_then_selects_a_ground_skill() -> None:
    """No query fires at construction -- the first one fires when the harness first asks
    for a policy, which is the initial evaluation sweep."""
    env = _env()
    method = _method(env=env)
    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    assert backend.prompts_seen() == []

    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_task_policy(task=task)

    assert len(backend.prompts_seen()) == 1
    label = policy(env.get_current_state()).label
    assert any(
        label.startswith(name)
        for name in ("MoveRobot", "TurnOnLight", "TurnOffLight", "JumpToLight")
    )


def test_the_observation_hands_the_policy_the_goal_atoms_and_the_applicable_skills() -> None:
    """The contract the authored code is written against. Captured by a policy that
    stashes what it was given, so a change to the observation shape breaks here rather
    than silently changing what every authored policy sees."""
    env = _env()
    capture = """
seen = {}


def policy(observation):
    seen.clear()
    seen.update(observation)
    return {"skill_index": 0, "params": [0.0] * observation["skills"][0]["param_dim"]}
"""
    method = _method(env=env, sources=(capture,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)(env.get_current_state())

    observation = method.last_observation()
    assert observation is not None
    assert set(observation) == {"goal", "objects", "atoms", "skills"}
    assert observation["goal"] == sorted(
        f"{atom.predicate.name}({', '.join(obj.name for obj in atom.objects)})"
        for atom in task.goal.atoms
    )
    assert observation["skills"], "at least one skill is applicable in the initial state"
    first = observation["skills"][0]
    assert set(first) == {"index", "name", "objects", "param_dim"}
    assert first["index"] == 0
    robot = next(entry for entry in observation["objects"] if entry["name"] == "robot")
    assert set(robot) == {"name", "type", "features"}
    assert set(robot["features"]) == set(LightSwitchEnvironment.robot_type.feature_names)


def test_end_cycle_authors_a_revision_and_the_new_policy_takes_effect() -> None:
    """`end_cycle` is the seam: PracticeLoop calls it after each interaction period and
    before that cycle's evaluation sweep, which is exactly the notebook's
    revise-then-score cadence."""
    env = _env()
    method = _method(env=env, sources=(FIRST_SKILL, DECLINES))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)

    before = method.get_task_policy(task=task)(env.get_current_state()).label
    method.end_cycle()
    after = method.get_task_policy(task=task)(env.get_current_state()).label

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    assert len(backend.prompts_seen()) == 2
    assert before != after


def test_the_revision_prompt_carries_the_practice_outcomes_as_counts() -> None:
    """`practice_outcomes()` is the feedback payload, and it reaches the agent as x/y
    rather than as a bare rate -- a rate over three attempts and a rate over three
    hundred support very different revisions."""
    env = _env()
    method = _method(env=env, sources=(FIRST_SKILL, FIRST_SKILL))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)

    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(6):
        state = env.take_action(action=policy(state).action)
    method.end_cycle()

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    feedback = backend.prompts_seen()[1]
    outcomes = method.practice_outcomes()
    assert outcomes, "practice steps were taken, so something must have been tallied"
    for name, tally in outcomes.items():
        assert f"{name}: {tally.num_successes}/{tally.num_attempts}" in feedback


def test_replay_reproduces_the_authoring_run_and_never_queries_a_backend() -> None:
    """Josh's record-then-replay decision: authoring is nondeterministic and happens
    once; every measured run replays the recorded sources with no API call at all."""
    env = _env()
    authored = _method(env=env, sources=(FIRST_SKILL, DECLINES))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()

    def actions(*, method: PureAgentMethod) -> list[str]:
        method.env.set_state(state=task.initial_state)
        labels = [method.get_task_policy(task=task)(method.env.get_current_state()).label]
        method.end_cycle()
        labels.append(method.get_task_policy(task=task)(method.env.get_current_state()).label)
        return labels

    authored_labels = actions(method=authored)
    replay_env = _env()
    replayed = PureAgentMethod(
        env=replay_env,
        skill_provider=LightSwitchSkillProvider(env=replay_env),
        replay_sources=authored.authoring_transcript().policy_sources(),
    )
    assert replayed.backend is None
    assert actions(method=replayed) == authored_labels


def test_a_method_needs_exactly_one_of_a_backend_and_replay_sources() -> None:
    env = _env()
    provider = LightSwitchSkillProvider(env=env)
    with pytest.raises(ValueError, match="exactly one"):
        PureAgentMethod(env=env, skill_provider=provider)
    with pytest.raises(ValueError, match="exactly one"):
        PureAgentMethod(
            env=env,
            skill_provider=provider,
            backend=ScriptedAgentBackend(sources=(FIRST_SKILL,)),
            replay_sources=(FIRST_SKILL,),
        )


def test_replay_running_out_of_recorded_rounds_raises_rather_than_reusing_the_last() -> None:
    """Silently holding the last round's policy would turn a truncated artifact into a
    flat learning curve that looks like a converged one."""
    env = _env()
    method = PureAgentMethod(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        replay_sources=(FIRST_SKILL,),
    )
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)
    with pytest.raises(RuntimeError, match="recorded"):
        method.end_cycle()


@pytest.mark.parametrize("source", [DOES_NOT_IMPORT, NO_POLICY_FUNCTION, RAISES_WHEN_CALLED, None])
def test_an_unusable_authored_policy_emits_a_no_op_and_is_recorded(*, source: str | None) -> None:
    """Every way the agent can fail to deliver ends in the same observable place: a
    no-op action and a recorded error, never a crashed run."""
    env = _env()
    method = _method(env=env, sources=(source,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    labeled = method.get_task_policy(task=task)(env.get_current_state())
    assert "no-op" in labeled.label
    assert method.authoring_transcript().rounds[0].load_error is not None


def test_the_round_after_a_failure_is_prompted_with_the_error() -> None:
    env = _env()
    method = _method(env=env, sources=(DOES_NOT_IMPORT, FIRST_SKILL))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)(env.get_current_state())
    method.end_cycle()

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    assert "did not produce a usable policy" in backend.prompts_seen()[1]
    label = method.get_task_policy(task=task)(env.get_current_state()).label
    assert "no-op" not in label


def test_the_recovery_prompt_restates_the_whole_task() -> None:
    """A round only reaches the error branch after the previous query failed, and the
    commonest way it fails is being cut off mid-turn by its own budget cap -- after which
    the CLI's `--continue` carries no usable memory of what was asked. Measured: on the
    2026-08-07 Tossing Room pilot the described arm burned all three rounds and $1.9544
    because the recovery prompt said only "Fix `policy.py`", and the agent had nothing to
    go on. So the recovery prompt has to be self-contained."""
    env = _env()
    method = _method(env=env, sources=(DOES_NOT_IMPORT, FIRST_SKILL))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)(env.get_current_state())
    method.end_cycle()

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    initial, recovery = backend.prompts_seen()
    # Everything the first prompt established -- the contract and the whole symbolic
    # layer -- is present again, so the round can succeed with no conversation at all.
    assert "def policy(observation)" in recovery
    provider = LightSwitchSkillProvider(env=env)
    for skill in provider.skills():
        assert skill.name in recovery
    assert PromptBuilder.symbolic_layer(skill_provider=provider) in recovery
    assert PromptBuilder.symbolic_layer(skill_provider=provider) in initial


def test_the_feedback_prompt_is_not_padded_with_the_whole_task() -> None:
    """The other side of the rule above, and it is a rule rather than an oversight: the
    feedback branch is only reached after a round that ran to completion and left a
    loadable file, where `--continue` has been observed to work. Restating the domain
    there would pay for it on every successful round for a case that cannot arise."""
    env = _env()
    method = _method(env=env, sources=(FIRST_SKILL, FIRST_SKILL))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)(env.get_current_state())
    method.end_cycle()

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    feedback = backend.prompts_seen()[1]
    provider = LightSwitchSkillProvider(env=env)
    assert PromptBuilder.symbolic_layer(skill_provider=provider) not in feedback


def test_a_malformed_decision_is_a_no_op_rather_than_a_crash_or_a_random_draw() -> None:
    """An out-of-range skill index is the agent's mistake, and it has to stay
    deterministic: falling back to a random draw would put an RNG stream in the replay
    path and break the byte-stability the whole design turns on."""
    env = _env()
    out_of_range = """
def policy(observation):
    return {"skill_index": 999, "params": []}
"""
    method = _method(env=env, sources=(out_of_range,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    labeled = method.get_task_policy(task=task)(env.get_current_state())
    assert "no-op" in labeled.label
    assert method.num_malformed_decisions() == 1


def test_counts_every_decision_so_a_be_the_policy_variant_can_be_priced() -> None:
    """The by-product Josh asked for: a be-the-policy variant makes one API call per
    decision, so this count *is* the price of that arm."""
    env = _env()
    method = _method(env=env, sources=(FIRST_SKILL,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    policy = method.get_practice_policy(task=task)
    state = env.get_current_state()
    for _ in range(7):
        state = env.take_action(action=policy(state).action)
    assert method.num_decisions() == 7


class _DeadEndSkillProvider(LightSwitchSkillProvider):
    """A provider with no skills at all, so every state is a dead end. Subclassing the
    real one keeps every other method honest; only `skills` is emptied."""

    def skills(self) -> tuple[Skill, ...]:
        return ()


def test_practice_ends_early_on_a_dead_end_while_evaluation_emits_a_no_op() -> None:
    """The same phase split RandomSkillsMethod makes, and for the same reason: a
    dead-ended period would otherwise burn its whole remaining budget on no-ops while
    the transition axis kept charging for them."""
    env = _env()
    method = PureAgentMethod(
        env=env,
        skill_provider=_DeadEndSkillProvider(env=env),
        backend=ScriptedAgentBackend(sources=(FIRST_SKILL,)),
    )
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)

    assert "no-op" in method.get_task_policy(task=task)(env.get_current_state()).label
    with pytest.raises(InteractionComplete):
        method.get_practice_policy(task=task)(env.get_current_state())


def test_the_described_arm_adds_the_domain_account_and_the_minimal_arm_does_not() -> None:
    """The two prompt arms. At one seed these establish that the plumbing carries both;
    they are not a measurement of what the hint is worth."""
    env = _env()
    description = "A robot walks a row of cells and toggles a light."
    minimal = _method(env=env, domain_description=description)
    described = _method(env=env, prompt_arm=PromptArm.DESCRIBED, domain_description=description)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    minimal.get_task_policy(task=task)
    described.get_task_policy(task=task)

    minimal_backend, described_backend = minimal.backend, described.backend
    assert isinstance(minimal_backend, ScriptedAgentBackend)
    assert isinstance(described_backend, ScriptedAgentBackend)
    assert description not in minimal_backend.prompts_seen()[0]
    assert description in described_backend.prompts_seen()[0]


def test_the_described_arm_refuses_an_empty_description() -> None:
    """A `--pure-agent-prompt-arm described` run with no description file is the minimal
    arm wearing the other arm's label, which would silently pool the two."""
    env = _env()
    with pytest.raises(ValueError, match="description"):
        _method(env=env, prompt_arm=PromptArm.DESCRIBED, domain_description="")


def test_every_prompt_names_the_domain_s_own_skills_and_predicates() -> None:
    """Domain-agnostic through SkillProvider: nothing here knows what Light Switch is,
    yet the prompt describes it, so the same method runs on any --env."""
    env = _env()
    method = _method(env=env)
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)

    backend = method.backend
    assert isinstance(backend, ScriptedAgentBackend)
    prompt = backend.prompts_seen()[0]
    provider = LightSwitchSkillProvider(env=env)
    for skill in provider.skills():
        assert skill.name in prompt
    for predicate in provider.predicates():
        assert predicate.name in prompt
    for object_type in provider.types():
        assert object_type.name in prompt


# Walks right until a parameterized skill becomes applicable, then takes it forever.
# Light Switch's `JumpToLight` is `param_dim=1` and becomes applicable two moves in, so
# this reaches a *parameterized* decision under both phases -- which is what makes the
# firewall test below non-vacuous. `FIRST_SKILL` would oscillate between two `MoveRobot`
# groundings and never select a parameterized skill at all, so an empty transition log
# would prove nothing.
SEEKS_A_PARAMETERIZED_SKILL = """
def policy(observation):
    skills = observation["skills"]
    for skill in skills:
        if skill["param_dim"] > 0:
            return {"skill_index": skill["index"], "params": [0.3] * skill["param_dim"]}
    return {"skill_index": len(skills) - 1, "params": []}
"""


def _drive(*, env: LightSwitchEnvironment, policy: Policy, steps: int) -> None:
    state = env.get_current_state()
    for _ in range(steps):
        state = env.take_action(action=policy(state).action)


def test_practice_transitions_record_the_observation_the_choice_and_the_outcome() -> None:
    """The in-context arm's payload: what the policy saw, what it did, and what happened.

    The aggregate tallies this arm shipped with say `JumpToLight: 0/9` and nothing more --
    not which parameter was tried, not what the state was when it was tried. A relation
    between an observable feature and a working parameter is not recoverable from a
    ratio, which is the whole defect the 2026-08-07 pilot measured."""
    env = _env()
    method = _method(env=env, sources=(SEEKS_A_PARAMETERIZED_SKILL,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    _drive(env=env, policy=method.get_practice_policy(task=task), steps=6)

    transitions = method.practice_transitions()
    assert transitions, "a parameterized skill was executed, so something must be recorded"
    first = transitions[0]
    assert first.skill_name == "JumpToLight"
    assert first.params == (0.3,)
    # The observation is the one the policy was called with, verbatim -- so a recorded
    # transition is replayable against the authored file without the environment.
    assert set(first.observation) == {"goal", "objects", "atoms", "skills"}
    assert first.observation["skills"][first.skill_index]["name"] == "JumpToLight"
    assert first.achieved_add_effects is False, "0.3 is outside the tolerance window"


def test_evaluation_transitions_never_reach_the_practice_transition_log() -> None:
    """**The firewall.** A `Method` is never shown its own evaluation outcomes; that is
    what makes training on the test set structurally impossible here rather than merely
    discouraged, and it is the property this whole feature could most easily destroy.

    `choose_ground_skill` is shared by both phases, so recording there -- the obvious
    place, since that is where the observation and the choice both exist -- would silently
    log every evaluation decision and feed the test set straight back to the agent. The
    append lives on the practice path only, and this test fails if it ever moves.

    Paired with a positive control in the same test on purpose: an assertion that a log is
    empty is worthless unless the same code path demonstrably fills it, and the evaluation
    phase here selects the same parameterized skill practice did."""
    env = _env()
    method = _method(env=env, sources=(SEEKS_A_PARAMETERIZED_SKILL,))
    tasks = LightSwitchTasks(env=env, seed=0)
    practice_task = tasks.sample_train_task()

    env.set_state(state=practice_task.initial_state)
    _drive(env=env, policy=method.get_practice_policy(task=practice_task), steps=6)
    after_practice = method.practice_transitions()
    assert after_practice, "positive control: practice must fill the log"

    # A *test* task, driven through the evaluation entrypoint, for at least as many steps.
    test_task = tasks.sample_test_task()
    env.set_state(state=test_task.initial_state)
    _drive(env=env, policy=method.get_task_policy(task=test_task), steps=12)

    assert method.practice_transitions() == after_practice, (
        "an evaluation transition reached the practice log: this Method is now able to "
        "train on the test set"
    )


def test_the_transition_log_holds_only_parameterized_skills() -> None:
    """Volume, and the reason the dump is small enough to be a prompt at all.

    One Tossing Room seed runs ~15,000 practice transitions, which no prompt holds. Only a
    parameterized skill carries a continuous decision to learn anything about, so those are
    the ones kept -- ~20 records rather than ~15,000. What is lost is real and is covered
    by the aggregate tallies, which still report every skill: a `param_dim=0` skill can
    only be right or wrong about *when* it was selected, and the tally says how often."""
    env = _env()
    method = _method(env=env, sources=(SEEKS_A_PARAMETERIZED_SKILL,))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    _drive(env=env, policy=method.get_practice_policy(task=task), steps=6)

    outcomes = method.practice_outcomes()
    assert outcomes["MoveRobot"].num_attempts > 0, "unparameterized skills were executed"
    assert {transition.skill_name for transition in method.practice_transitions()} == {
        "JumpToLight"
    }
    assert all(transition.params for transition in method.practice_transitions())


def test_the_transcript_records_what_each_round_cost() -> None:
    env = _env()
    method = _method(env=env, sources=(FIRST_SKILL, DECLINES))
    task = LightSwitchTasks(env=env, seed=0).sample_train_task()
    env.set_state(state=task.initial_state)
    method.get_task_policy(task=task)
    method.end_cycle()

    transcript = method.authoring_transcript()
    assert len(transcript.rounds) == 2
    assert [round_.round_index for round_ in transcript.rounds] == [0, 1]
    assert transcript.total_cost_usd() == pytest.approx(0.0)
    assert transcript.policy_sources() == (FIRST_SKILL, DECLINES)
