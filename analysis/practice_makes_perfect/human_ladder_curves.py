"""Post-run analysis for the human-in-the-loop baseline ladder on Tossing Room: **does
being rescued by a human make reset-free practice work?**

**Background.** `--practice-reset-policy never` is the real-robot condition -- a robot
practising in a lab is not teleported to a fresh start every few minutes. Measured on
this domain, it is also badly damaged: Tossing Room's one-way ledge severs rooms 0-2
from the item pile in room 3, so a practice period that steps left once can never pick
anything up again, and under `never` that damage carries into every later period. The
reset-free A/B found the one-way reset-free arm spending its full 150 transitions per
cycle walking, managing 207 effective practice attempts pooled against the
scheduled-reset arm's 1191, with 85/100 cycles attempting not one.

A human is the sanctioned way out: `Problem.execute_human_command` is the only reset a
robot with irreversible actions is entitled to, and it is *charged*. This reads back
eight arms, all `--practice-reset-policy never`, that differ in whether and how one is
called.

**The eight arms, and which comparisons are clean.**

The help-seeking interface is a product of two orthogonal flags -- `--ask-for-help`
(*when* the robot asks; a method flag on `--method ees`) and `--human-reset-target`
(*what* the human does on arrival; a global flag, because it is a property of the human).
The four treated arms are the full 2x2 of that product, which is what makes the timing
axis and the target axis separately identifiable:

| arm                 | `--method`      | `--ask-for-help` | `--human-reset-target` | world   |
| ------------------- | --------------- | ---------------- | ---------------------- | ------- |
| `no-human`          | `ees`           | `never`          | --                     | one-way |
| `stuck-initial`     | `ees`           | `on-stuck`       | `task-initial`         | one-way |
| `stuck-random`      | `ees`           | `on-stuck`       | `random`               | one-way |
| `at-random-initial` | `ees`           | `at-random`      | `task-initial`         | one-way |
| `at-random-random`  | `ees`           | `at-random`      | `random`               | one-way |
| `two-way-ledge`     | `ees`           | `never`          | --                     | two-way |
| `skill-oracle`      | `skill-oracle`  | --               | --                     | one-way |
| `random-skills`     | `random-skills` | --               | --                     | one-way |

**Five arms are paired against each other and nothing else is.** `no-human` and the four
treated arms share `--method ees`, the same one-way world and every seed, so differences
between them isolate *the human* and `PairedTests.sign_flip` applies. The other three each
change a second thing:

* `two-way-ledge` changes the **world**, not the human -- it is the ceiling with
  irreversibility removed, so a gap against it prices what the one-way ledge costs, not
  what a human buys. Reported as a ceiling level, never sign-flipped against a human arm.
* `skill-oracle` changes the **Method** -- the privileged hand-authored solver, the
  ceiling on skill quality. It never practises at all (no `--num-cycles` flag exists for
  it), so it has a single evaluation checkpoint.
* `random-skills` changes the **Method**, so a gap between it and any EES arm is a gap in
  two things at once. Reported as a floor and differenced against nothing.

**`at-random` is the control for `on-stuck`, and it is EES-based on purpose.** It asks on
a schedule of the method's own -- Bernoulli(1/`--mean-steps-between-help-requests`) per
policy call, defaulting to one request per 150-step period -- with the timing carrying no
information about the robot's situation. Because it shares `--method ees` and its seeds
with the control, it is a legitimately paired comparison, which an earlier layout of this
ladder could not offer: it put random-timing rescue on `--method random-skills`, which
confounded it with the method and left it in no comparison at all.

**But the two triggers do NOT fire at the same rate, so the contrast is not "timing at
matched cost".** The `--mean-steps-between-help-requests 150` default was chosen to make
them comparable and does not: measured here, `on-stuck` spends about 3.4x the rescues
`at-random` does, because a stranded robot is stuck on many consecutive steps while the
Bernoulli schedule fires about once a period. So `on-stuck` minus `at-random` is a gap in
**timing and rate together**, and a reader who wants the timing effect alone should read
the per-rescue view in `render_interventions`' right panel instead, where the two arms'
cost-effectiveness can be compared directly. `print_report` prints both the gap and the
solves-per-rescue for exactly this reason. Equalising the rate would need
`--mean-steps-between-help-requests` retuned per arm and is not what was run.

**Non-learners are drawn as reference lines, not curves.** `skill-oracle` and
`random-skills` do not improve with practice -- the first never practises, the second acts
at random forever -- so a "curve" for either would invite a reader to see learning in
noise. They get a horizontal reference line at their pooled level, matching how
`tossing3d_ees_arms.py` and `ees.py` already draw the privileged oracle. Only the six EES
arms get curves. `two-way-ledge` is **not** in that list and is deliberately drawn as a
curve: it is EES and it genuinely learns, so flattening it would misreport it.

**The manipulation checks are not optional here.** `num_practice_resets` must be 0 in
every arm, or an arm labelled reset-free was quietly reset for free.
`num_human_interventions_recorded` must be 0 in the arms that wire **no reachable human at
all** (`no-human`, `two-way-ledge`, `skill-oracle`, `random-skills`) and is *measured*,
never assumed, in the four that ask -- because "the human did not help" and "the human was
never called" are completely different findings and only this number tells them apart.

A rescue **consumes its loop iteration**: `PracticeLoop`'s `except HumanHelpRequested`
branch `continue`s, which is what stops a method that asks on every call from spinning. So
an asking arm takes exactly one fewer online transition per rescue, and its final
checkpoint sits at `num_cycles * max_steps_per_interaction - interventions` rather than at
the round number. That is checked below rather than assumed, and it is the reason the
arms' x-axes do not all end on the same value.

**Statistics.** All arms ran the same fixed seeds, so every EES-to-EES comparison is
paired and the test is `PairedTests.sign_flip`, exact by enumerating its null in full --
no normal approximation, no continuity or tie correction, no scipy. Imported from
`paired_tests` rather than reimplemented; goal classification comes from
`goal_families.GoalFamilies` for the same reason.

**Per-seed spread is plotted, not only a mean.** At ten seeds a mean can describe none
of them -- the one-way reset-free arm is bimodal on this domain -- so every figure draws
one faint line per seed under the bold pooled curve, and the cross-arm figures carry a
per-seed strip so a single seed driving an arm is visible rather than averaged away.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never
runs a simulation or drives a `Method`). Each `--arm` points at the directory holding
that arm's `<seed>/stats.json`.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# The eight arms, in the order every table, legend and report below uses: the incumbent
# first, then the 2x2 of (when to ask) x (what the human does), then the two ceilings,
# then the floor. `no-human` is the arm every banked reset-free number was measured under.
_ARMS = (
    "no-human",
    "stuck-initial",
    "stuck-random",
    "at-random-initial",
    "at-random-random",
    "two-way-ledge",
    "skill-oracle",
    "random-skills",
)

# The arms that do not learn, and are therefore drawn as a horizontal reference line at
# their pooled level rather than as a curve. `skill-oracle` never practises at all;
# `random-skills` acts at random forever. Plotting either as a curve would invite a reader
# to read learning into noise. `two-way-ledge` is deliberately NOT here: it is EES and it
# does learn, so drawing it flat would misreport it -- checked, not assumed.
_REFERENCE_ARMS = ("skill-oracle", "random-skills")

# The arms with no reachable human at all, which must therefore record zero interventions.
# `no-human` and `two-way-ledge` pass `--ask-for-help never`, so EesMethod builds no
# help-seeking policy; `skill-oracle` and `random-skills` do not register the flag at all.
_ARMS_WITHOUT_A_HUMAN = ("no-human", "two-way-ledge", "skill-oracle", "random-skills")

# The composition the domain allocates for --num-test-tasks 30. Asserted per sweep,
# because a goal misfiled between families moves tasks between denominators invisibly.
_COMPOSITION = {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}
_NUM_TEST_TASKS = sum(_COMPOSITION.values())

# Every arm is `--practice-reset-policy never`, so a free harness reset anywhere means
# the arm is not what its name says.
_EXPECTED_PRACTICE_RESETS = 0

# humans/oracle.py's UnconditionalHumanOracle charges a flat 1.0 per rescue, so cost and
# count are proportional at v0. Checked rather than assumed: if they ever come apart, a
# different oracle was wired and every cost number below means something else.
_V0_INTERVENTION_COST = 1.0

# Colourblind-safe and distinguishable in greyscale; linestyle carries the trigger too,
# so an arm's identity never rests on hue alone.
_COLORS = {
    "no-human": "#0072B2",
    "stuck-initial": "#D55E00",
    "stuck-random": "#009E73",
    "at-random-initial": "#E69F00",
    "at-random-random": "#56B4E9",
    "two-way-ledge": "#CC79A7",
    "skill-oracle": "#000000",
    "random-skills": "#785EF0",
}
# Linestyle carries the TRIGGER axis so an arm's identity never rests on hue alone:
# `on-stuck` arms are dashed, `at-random` arms are dash-dotted, the control is solid.
_LINESTYLES = {
    "no-human": "-",
    "stuck-initial": "--",
    "stuck-random": (0, (5, 1)),
    "at-random-initial": "-.",
    "at-random-random": (0, (6, 2, 1, 2)),
    "two-way-ledge": (0, (3, 1)),
    "skill-oracle": "--",
    "random-skills": ":",
}

# Display names. DISPLAY ONLY: the keys, directory names and flag values stay as they
# are, because those are what the CLI accepts and what every config_snapshot.json
# records.
_LABELS = {
    "no-human": "EES, no human (control)",
    "stuck-initial": "EES, asks on stuck → task initial state",
    "stuck-random": "EES, asks on stuck → random state",
    "at-random-initial": "EES, asks at random → task initial state",
    "at-random-random": "EES, asks at random → random state",
    "two-way-ledge": "EES, two-way ledge (ceiling: no irreversibility)",
    "skill-oracle": "skill oracle (ceiling: skills)",
    "random-skills": "random skills (floor)",
}


class HumanLadderCurves:
    """A static-method container, never instantiated."""

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y`, never a bare percentage: the denominators here are small and uneven."""
        return f"{solved}/{total}"

    @staticmethod
    def load_arms(*, directories: dict[str, Path]) -> dict[str, dict]:
        """Every arm's per-seed data, with the checks first.

        All eight arms are required. A missing one is refused rather than worked around:
        with `no-human` absent there is no control to difference against, with a `stuck`
        arm absent the target comparison does not exist, and with a ceiling absent the
        remaining-gap arithmetic has no ceiling -- and a report that silently printed
        whatever comparison it could still make would read as a result."""
        missing = [arm for arm in _ARMS if arm not in directories]
        if missing:
            raise ValueError(
                f"missing arm(s): {', '.join(missing)}. This experiment is eight arms "
                "sharing one seed set; with one absent the paired comparisons it exists "
                "to make are not defined."
            )
        return {
            arm: HumanLadderCurves.load_arm(directory=directories[arm], arm=arm) for arm in _ARMS
        }

    @staticmethod
    def load_arm(*, directory: Path, arm: str) -> dict:
        """One arm: `{seed: {"transitions", "families", "overall", "interventions",
        "human_cost"}}`."""
        seeds = sorted(int(path.parent.name) for path in directory.glob("*/stats.json"))
        if not seeds:
            raise ValueError(f"no <seed>/stats.json under {directory}")
        loaded: dict[int, dict] = {}
        for seed in seeds:
            stats = json.loads((directory / str(seed) / "stats.json").read_text())
            HumanLadderCurves.check_manipulation(stats=stats, arm=arm, where=f"{directory}/{seed}")
            transitions = []
            families: dict[str, list[tuple[int, int]]] = {family: [] for family in _COMPOSITION}
            overall: list[tuple[int, int]] = []
            for breakdown in stats["breakdowns"]:
                transitions.append(breakdown["num_online_transitions"])
                counts = HumanLadderCurves.sweep_counts(outcomes=breakdown["outcomes"])
                composition = {family: total for family, (_, total) in counts.items()}
                if composition != _COMPOSITION:
                    raise ValueError(
                        f"{directory}/{seed}: sweep composition {composition} is not the "
                        f"domain's {_COMPOSITION}. A goal has been misfiled between "
                        "families, which moves tasks between denominators invisibly."
                    )
                for family, count in counts.items():
                    families[family].append(count)
                overall.append((sum(solved for solved, _ in counts.values()), _NUM_TEST_TASKS))
            loaded[seed] = {
                "transitions": transitions,
                "families": families,
                "overall": overall,
                "interventions": stats.get("num_human_interventions_recorded", 0),
                "human_cost": stats.get("summed_human_cost_recorded", 0.0),
            }
        return loaded

    @staticmethod
    def check_manipulation(*, stats: dict, arm: str, where: str) -> None:
        """The three things that make an arm's name true, checked before any number off
        it is used.

        Only the first two are hard requirements. Whether a `stuck` arm was ever
        rescued is deliberately NOT one: zero interventions there is a finding about the
        trigger, and rejecting the run would hide exactly the result worth reporting."""
        resets = stats.get("num_practice_resets")
        if resets != _EXPECTED_PRACTICE_RESETS:
            raise ValueError(
                f"{where}: num_practice_resets is {resets}, expected "
                f"{_EXPECTED_PRACTICE_RESETS}. Every arm here is "
                "--practice-reset-policy never, so this arm was quietly reset for free "
                "and its practice is not reset-free."
            )
        interventions = stats.get("num_human_interventions_recorded", 0)
        cost = stats.get("summed_human_cost_recorded", 0.0)
        if arm in _ARMS_WITHOUT_A_HUMAN and interventions:
            raise ValueError(
                f"{where}: the {arm} arm recorded {interventions} human interventions. "
                "It is the control and must never call one."
            )
        if abs(cost - interventions * _V0_INTERVENTION_COST) > 1e-9:
            raise ValueError(
                f"{where}: summed_human_cost_recorded is {cost} against {interventions} "
                f"interventions, which is not the flat {_V0_INTERVENTION_COST} the v0 "
                "oracle charges. A different HumanOracle was wired, so every cost "
                "number here means something else."
            )

    @staticmethod
    def sweep_counts(*, outcomes: list[dict]) -> dict[str, tuple[int, int]]:
        """One sweep's `(solved, total)` per family.

        Classification is `GoalFamilies.classify`, reused rather than recopied: it tests
        the `BinEmpty` predicate before the item names, because `Goal.describe()` renders
        EMPTY naming BOTH bins, so a naive "does it mention recycling?" test swallows it
        and silently reports 16 RECYCLING / 0 EMPTY."""
        solved: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for outcome in outcomes:
            family = GoalFamilies.classify(goal=outcome["goal"])
            total[family] += 1
            solved[family] += int(outcome["solved"])
        return {family: (solved[family], total[family]) for family in total}

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def entry(*, arm: dict, seed: int, family: str | None) -> list[tuple[int, int]]:
        """One seed's `(solved, total)` per checkpoint, overall or for one family."""
        return arm[seed]["overall"] if family is None else arm[seed]["families"][family]

    @staticmethod
    def pooled_curve(*, arm: dict, family: str | None) -> list[tuple[int, int]]:
        """The arm's curve pooled over seeds: solved and total both SUMMED, per
        checkpoint.

        Summed rather than averaged, so `x/300` at ten seeds means what it says. A mean
        of per-seed rates would silently reweight a seed that ran a different number of
        tasks."""
        seeds = sorted(arm)
        num_checkpoints = len(arm[seeds[0]]["transitions"])
        pooled = []
        for index in range(num_checkpoints):
            solved = 0
            total = 0
            for seed in seeds:
                entry = HumanLadderCurves.entry(arm=arm, seed=seed, family=family)
                solved += entry[index][0]
                total += entry[index][1]
            pooled.append((solved, total))
        return pooled

    @staticmethod
    def transitions(*, arm: dict) -> list[int]:
        """The pooled x axis: each checkpoint's MEAN transition count over the seeds.

        Seeds do not share a grid on any arm that asks for help, and that is correct
        behaviour rather than a defect. `PracticeLoop`'s `except HumanHelpRequested` branch
        `continue`s, so a granted rescue consumes its loop iteration and a rescued seed
        reaches every later checkpoint one transition earlier per rescue. Ten seeds rescued
        25-43 times therefore end 25-43 transitions short of the nominal
        `num_cycles * max_steps_per_interaction`, all differently.

        An earlier version of this function *raised* on exactly that, on the premise --
        stated in its own docstring -- that "a rescue is not charged as a transition". The
        reshape that moved the trigger method-side made that premise false, so the check
        now rejects only what is still a real error: a differing NUMBER of checkpoints,
        which means a seed ran a different number of cycles and no amount of averaging
        makes those curves commensurable.

        The mean is used for the pooled curve only. Per-seed lines are drawn at each
        seed's own transition counts, so the spread on the x axis stays visible rather
        than being averaged into the summary."""
        lengths = {len(arm[seed]["transitions"]) for seed in sorted(arm)}
        if len(lengths) != 1:
            raise ValueError(
                f"seeds disagree on the number of evaluation checkpoints ({sorted(lengths)}). "
                "That is a differing cycle count, not the one-transition-per-rescue "
                "shortfall, so these curves are not commensurable."
            )
        seeds = sorted(arm)
        return [
            round(sum(arm[seed]["transitions"][index] for seed in seeds) / len(seeds))
            for index in range(next(iter(lengths)))
        ]

    @staticmethod
    def final_per_seed(*, arm: dict, family: str | None) -> list[int]:
        """Each seed's final-checkpoint solved count, in seed order."""
        return [
            HumanLadderCurves.entry(arm=arm, seed=seed, family=family)[-1][0]
            for seed in sorted(arm)
        ]

    @staticmethod
    def paired_final_differences(
        *, arms: dict, treatment: str, control: str, family: str | None
    ) -> list[float]:
        """`treatment` minus `control` at the final checkpoint, **within a seed**.

        The arms share seeds, so this is paired data and an unpaired test would throw
        that structure away. Zero differences are KEPT rather than dropped: "9/10 seeds
        differ by exactly zero" is itself a headline and is invisible if ties are
        discarded."""
        left, right = arms[treatment], arms[control]
        seeds = sorted(set(left) & set(right))
        return [
            float(
                HumanLadderCurves.entry(arm=left, seed=seed, family=family)[-1][0]
                - HumanLadderCurves.entry(arm=right, seed=seed, family=family)[-1][0]
            )
            for seed in seeds
        ]

    @staticmethod
    def comparisons() -> tuple[tuple[str, str], ...]:
        """The (treatment, control) pairs that are legitimately paired: EES against EES,
        in the SAME one-way world, on shared seeds -- so the only thing that differs is
        the human.

        Three arms appear in none of them, each because it moves a second variable:
        `two-way-ledge` changes the world, and `skill-oracle` and `random-skills` each
        change the Method.

        The seven pairs are three groups. **Four against the control** ask "does being
        rescued at all help?", one per cell of the 2x2. **Two `on-stuck` minus
        `at-random` at a matched target** ask the sharper question -- does the *timing*
        of a rescue carry information, holding the rescue rate and what the human does
        fixed? **One `random` minus `task-initial` at a matched trigger** isolates the
        target axis. Every pair shares `--method ees`, the one-way world and all ten
        seeds, so the only thing that differs is the named factor."""
        return (
            ("stuck-initial", "no-human"),
            ("stuck-random", "no-human"),
            ("at-random-initial", "no-human"),
            ("at-random-random", "no-human"),
            ("stuck-initial", "at-random-initial"),
            ("stuck-random", "at-random-random"),
            ("stuck-random", "stuck-initial"),
        )

    @staticmethod
    def solves_per_rescue(*, arms: dict, treatment: str, control: str) -> float | None:
        """Extra tasks solved per human rescue spent: the gap divided by what it cost.

        The headline gap is not comparable across arms that rescue at different rates, and
        `on-stuck` and `at-random` measurably do -- about 3.4x apart on this domain. An arm
        can therefore post the larger gap while being the *worse* use of a human, and only
        this ratio makes that visible. Pooled over seeds rather than averaged per seed: a
        per-seed ratio is undefined for any seed that was never rescued, and dropping those
        seeds would quietly change the denominator.

        `None`, never a number, when the treatment spent nothing -- that is a division by
        zero, and reporting it as `inf` or `0` would both read as findings."""
        rescues = sum(arms[treatment][seed]["interventions"] for seed in sorted(arms[treatment]))
        if not rescues:
            return None
        gap = sum(
            HumanLadderCurves.paired_final_differences(
                arms=arms, treatment=treatment, control=control, family=None
            )
        )
        return gap / rescues

    # ------------------------------------------------------------------ the report

    @staticmethod
    def print_report(*, arms: dict) -> None:
        """Every number the write-up quotes, as `x/y`, re-derived here."""
        print("manipulation checks and how much human help each arm actually bought\n")
        for arm_name in _ARMS:
            arm = arms[arm_name]
            seeds = sorted(arm)
            interventions = [arm[seed]["interventions"] for seed in seeds]
            rescued_seeds = sum(1 for count in interventions if count)
            print(
                f"  {arm_name:>14}   {sum(interventions):>5} interventions pooled"
                f"   cost {sum(arm[seed]['human_cost'] for seed in seeds):>7.1f}"
                f"   {rescued_seeds}/{len(seeds)} seeds rescued at all"
                f"   per-seed {min(interventions)}-{max(interventions)}"
            )
        print("\n  Every arm reports num_practice_resets == 0 (checked on load).\n")

        print("final-checkpoint scores, pooled over seeds\n")
        for family in (None, "TRASH", "RECYCLING", "EMPTY"):
            name = "OVERALL" if family is None else family
            print(f"  {name}")
            for arm_name in _ARMS:
                final = HumanLadderCurves.pooled_curve(arm=arms[arm_name], family=family)[-1]
                per_seed = HumanLadderCurves.final_per_seed(arm=arms[arm_name], family=family)
                print(
                    f"    {arm_name:>14}  "
                    f"{HumanLadderCurves.format_count(solved=final[0], total=final[1]):>8}"
                    f"   per-seed {min(per_seed)}-{max(per_seed)}"
                )
            print()

        print("paired exact sign-flip tests, EES arms only (shared seeds)\n")
        for family in (None, "TRASH", "RECYCLING", "EMPTY"):
            name = "OVERALL" if family is None else family
            print(f"  {name}")
            for treatment, control in HumanLadderCurves.comparisons():
                differences = HumanLadderCurves.paired_final_differences(
                    arms=arms, treatment=treatment, control=control, family=family
                )
                test = PairedTests.sign_flip(differences=differences)
                better = sum(1 for d in differences if d > 0)
                worse = sum(1 for d in differences if d < 0)
                mde = PairedTests.minimum_detectable_effect(differences=differences)
                print(
                    f"    {treatment:>17} - {control:<17}"
                    f" gap {int(sum(differences)):>+5}"
                    f"   better {better}/{len(differences)}"
                    f"   worse {worse}/{len(differences)}"
                    f"   tied {test.num_zero_differences}/{len(differences)}"
                    f"   p = {test.p_value:.4g}   MDE {mde:.2f}"
                )
            print()

        HumanLadderCurves.print_cost_effectiveness(arms=arms)
        HumanLadderCurves.print_ceiling_gaps(arms=arms)

    @staticmethod
    def print_cost_effectiveness(*, arms: dict) -> None:
        """Each treated arm's gap over the control, priced by the rescues it spent.

        Printed next to the sign-flip table because the two answer different questions and
        the first is routinely misread as the second: a larger gap is not a better use of a
        human if it was bought with several times the help. This is the column that says
        so."""
        print("what each arm's gap over the control cost in human help\n")
        for arm_name in _ARMS:
            if arm_name in _REFERENCE_ARMS or arm_name in ("no-human", "two-way-ledge"):
                continue
            rescues = sum(arms[arm_name][seed]["interventions"] for seed in sorted(arms[arm_name]))
            gap = int(
                sum(
                    HumanLadderCurves.paired_final_differences(
                        arms=arms, treatment=arm_name, control="no-human", family=None
                    )
                )
            )
            ratio = HumanLadderCurves.solves_per_rescue(
                arms=arms, treatment=arm_name, control="no-human"
            )
            shown = "n/a (never rescued)" if ratio is None else f"{ratio:.3f}"
            print(
                f"  {arm_name:>17}   gap {gap:>+5} tasks   {rescues:>4} rescues"
                f"   {shown} extra solves per rescue"
            )
        print()

    @staticmethod
    def print_ceiling_gaps(*, arms: dict) -> None:
        """What is still missing between the best human arm and each ceiling.

        Descriptive arithmetic on pooled counts, NOT a test: the two ceilings each move a
        second variable (`two-way-ledge` the world, `skill-oracle` the Method), so a
        sign-flip against them would be answering a question this design cannot ask. It
        is printed because the size of the remaining gap is what the next experiment has
        to aim at, and because a reader who sees 220/300 next to 300/300 will do this
        subtraction anyway -- better it is done once, correctly, with its denominators
        visible."""
        print("remaining gap to each ceiling (descriptive, not a test)\n")
        best = max(
            (arm for arm in _ARMS if arm not in _REFERENCE_ARMS and arm != "two-way-ledge"),
            key=lambda name: HumanLadderCurves.pooled_curve(arm=arms[name], family=None)[-1][0],
        )
        best_solved, best_total = HumanLadderCurves.pooled_curve(arm=arms[best], family=None)[-1]
        print(
            f"  best human arm: {best} at "
            f"{HumanLadderCurves.format_count(solved=best_solved, total=best_total)}"
        )
        for ceiling in ("two-way-ledge", "skill-oracle"):
            solved, total = HumanLadderCurves.pooled_curve(arm=arms[ceiling], family=None)[-1]
            print(
                f"    vs {ceiling:>14} "
                f"{HumanLadderCurves.format_count(solved=solved, total=total):>8}"
                f"   gap {solved - best_solved:>+5}/{total}"
            )
        print()

    # ------------------------------------------------------------------ the figures

    @staticmethod
    def render_arm(*, arms: dict, arm_name: str, output: Path, title: str) -> None:
        """One arm's own figure: its learning curve, and what its rescues bought it.

        **Left** is the curve, one faint line per seed under the bold pooled mean --
        because a mean over ten seeds can describe none of them when an arm is bimodal,
        which this domain's reset-free arm is.

        **Right** plots each seed's final score against how many times that seed was
        rescued. It is the panel that separates "the human did not help" from "the human
        was never called": for `no-human` it collapses to a column at zero, which is the
        honest picture of that arm's spread, and for a `stuck` arm a flat scatter says
        rescues bought nothing while a sloped one says they did."""
        arm = arms[arm_name]
        color = _COLORS[arm_name]
        fig, (ax_curve, ax_rescue) = plt.subplots(1, 2, figsize=(12.6, 5.0))
        xs = HumanLadderCurves.transitions(arm=arm)
        seeds = sorted(arm)
        pooled = HumanLadderCurves.pooled_curve(arm=arm, family=None)
        scale = _NUM_TEST_TASKS / pooled[-1][1]
        label = (
            f"{_LABELS[arm_name]} — "
            f"{HumanLadderCurves.format_count(solved=pooled[-1][0], total=pooled[-1][1])}"
        )
        if arm_name in _REFERENCE_ARMS:
            # A non-learner: one horizontal line at its pooled level, plus each seed as a
            # faint line of its own so the spread is still visible. No curve, because
            # there is no learning for a curve to show -- see _REFERENCE_ARMS.
            for seed in seeds:
                entry = HumanLadderCurves.entry(arm=arm, seed=seed, family=None)
                ax_curve.axhline(entry[-1][0], color=color, alpha=0.22, linewidth=0.9)
            ax_curve.axhline(
                pooled[-1][0] * scale,
                color=color,
                linestyle=_LINESTYLES[arm_name],
                linewidth=2.6,
                label=label,
            )
        else:
            for seed in seeds:
                entry = HumanLadderCurves.entry(arm=arm, seed=seed, family=None)
                # That seed's OWN transition counts, not the pooled mean: a rescued seed
                # ends short by its own rescue count, and drawing it on the mean grid
                # would misplace it by up to ~20 transitions.
                ax_curve.plot(
                    arm[seed]["transitions"],
                    [s for s, _ in entry],
                    color=color,
                    alpha=0.25,
                    linewidth=0.9,
                )
            ax_curve.plot(
                xs,
                [solved * scale for solved, _ in pooled],
                color=color,
                linestyle=_LINESTYLES[arm_name],
                linewidth=2.6,
                label=label,
            )
        ax_curve.set_xlabel("online transitions")
        ax_curve.set_ylabel(
            f"test tasks solved per seed (x/{_NUM_TEST_TASKS});  legend gives the pooled count",
            fontsize=9,
        )
        ax_curve.set_ylim(-_NUM_TEST_TASKS * 0.04, _NUM_TEST_TASKS * 1.06)
        ax_curve.grid(alpha=0.25, linewidth=0.6)
        ax_curve.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
        if len(xs) == 1:
            # `skill-oracle` never practises, so its x axis spans nothing and matplotlib
            # invents a 0-1 range. Saying so beats letting a reader read the invented
            # axis as real practice.
            ax_curve.annotate(
                "this arm never practises: one evaluation, at 0 online transitions",
                (0.5, 0.06),
                xycoords="axes fraction",
                ha="center",
                fontsize=8.5,
            )

        rescues = [arm[seed]["interventions"] for seed in seeds]
        finals = HumanLadderCurves.final_per_seed(arm=arm, family=None)
        ax_rescue.scatter(rescues, finals, color=color, s=46, alpha=0.85, zorder=3)
        for seed, x, y in zip(seeds, rescues, finals, strict=True):
            ax_rescue.annotate(
                f"s{seed}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5
            )
        ax_rescue.set_xlabel("human interventions this seed was charged")
        ax_rescue.set_ylabel(f"final test tasks solved (x/{_NUM_TEST_TASKS})", fontsize=9)
        ax_rescue.set_ylim(-_NUM_TEST_TASKS * 0.04, _NUM_TEST_TASKS * 1.06)
        ax_rescue.grid(alpha=0.25, linewidth=0.6)
        if max(rescues) == 0:
            # A degenerate axis reads as a bug otherwise; saying it is the arm's defining
            # property costs one annotation.
            ax_rescue.set_xlim(-1, 1)
            ax_rescue.annotate(
                "no human was ever called in this arm",
                (0.5, 0.06),
                xycoords="axes fraction",
                ha="center",
                fontsize=8.5,
            )
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    @staticmethod
    def render_family(
        *, arms: dict, family: str | None, output: Path, title: str, legend_loc: str = "upper left"
    ) -> None:
        """All eight arms on ONE goal family: the curves, and the per-seed final spread
        beside them.

        **One figure per family, not one pooled figure**, matching
        `reset_free_ledge_curves.py` (PR #138), which draws the same three-way split for
        the reset-free ledge A/B. The reason is the same one it had there, and here it is
        the headline: the entire effect of a human rescue is RECYCLING (22/140 → 132/140)
        while TRASH is a null result (70/140 vs 68/140). A pooled curve averages a large
        effect against a flat one and shows a muted version of neither.

        **EMPTY gets no figure.** It is 20/20 in every one of the eight arms — 2 tasks per
        seed, at ceiling before any manipulation — so there is nothing for a curve to show.
        It stays in the printed report, where its denominator is visible.

        **Every level here is the FAMILY's own**, never the overall figure's carried
        across: `skill-oracle` is 140/140 on both throw families but 300/300 overall, and
        `two-way-ledge` is 127/140 on TRASH against 140/140 on RECYCLING. `pooled_curve`
        is asked for `family` and the per-seed denominator is read back off the data, so a
        family panel cannot silently inherit the wrong denominator.

        The strip on the right is there because a bar chart of eight means would hide one
        seed driving an arm, which is exactly the failure mode this domain produces. Each
        seed is one dot; the bold tick is the pooled count rescaled onto the same axis."""
        fig, (ax_curve, ax_spread) = plt.subplots(
            1, 2, figsize=(13.6, 5.4), sharey=True, width_ratios=(2.1, 1.0)
        )
        # This family's per-seed denominator (30 overall, 14 per throw family), read off
        # the data rather than hardcoded, so it cannot disagree with _COMPOSITION.
        seed_total = HumanLadderCurves.entry(
            arm=arms[_ARMS[0]], seed=sorted(arms[_ARMS[0]])[0], family=family
        )[-1][1]
        for arm_name in _ARMS:
            arm = arms[arm_name]
            color = _COLORS[arm_name]
            xs = HumanLadderCurves.transitions(arm=arm)
            pooled = HumanLadderCurves.pooled_curve(arm=arm, family=family)
            scale = seed_total / pooled[-1][1]
            rescues = sum(arm[seed]["interventions"] for seed in sorted(arm))
            label = (
                f"{_LABELS[arm_name]} — "
                f"{HumanLadderCurves.format_count(solved=pooled[-1][0], total=pooled[-1][1])}"
                f"  ({rescues} rescues)"
            )
            if arm_name in _REFERENCE_ARMS:
                ax_curve.axhline(
                    pooled[-1][0] * scale,
                    color=color,
                    linestyle=_LINESTYLES[arm_name],
                    linewidth=2.4,
                    label=label,
                )
                continue
            for seed in sorted(arm):
                entry = HumanLadderCurves.entry(arm=arm, seed=seed, family=family)
                # Each seed at its own transition counts -- see `transitions`.
                ax_curve.plot(
                    arm[seed]["transitions"],
                    [s for s, _ in entry],
                    color=color,
                    alpha=0.16,
                    linewidth=0.8,
                )
            ax_curve.plot(
                xs,
                [solved * scale for solved, _ in pooled],
                color=color,
                linestyle=_LINESTYLES[arm_name],
                linewidth=2.4,
                label=label,
            )
        # An asking arm's x-axis ends short of num_cycles * max_steps_per_interaction by
        # exactly its rescue count, because a granted rescue consumes its loop iteration.
        # Named in the xlabel rather than left for a reader to notice, since otherwise the
        # arms' curves ending at different x looks like a bug.
        ax_curve.set_xlabel(
            "online transitions\n"
            "an asking arm ends one transition short per rescue — a granted rescue "
            "consumes its loop iteration",
            fontsize=9,
        )
        ax_curve.set_ylabel(
            f"test tasks solved per seed (x/{seed_total});  legend gives the pooled count",
            fontsize=9,
        )
        ax_curve.grid(alpha=0.25, linewidth=0.6)
        ax_curve.legend(fontsize=8, loc=legend_loc, framealpha=0.95)

        for index, arm_name in enumerate(_ARMS):
            finals = HumanLadderCurves.final_per_seed(arm=arms[arm_name], family=family)
            # A tiny deterministic horizontal spread so coincident seeds stay countable;
            # index-derived rather than random, so the figure is reproducible.
            offsets = [(i % 5 - 2) * 0.055 for i in range(len(finals))]
            ax_spread.scatter(
                [index + offset for offset in offsets],
                finals,
                color=_COLORS[arm_name],
                s=42,
                alpha=0.85,
                zorder=3,
            )
            pooled = HumanLadderCurves.pooled_curve(arm=arms[arm_name], family=family)[-1]
            ax_spread.plot(
                [index - 0.26, index + 0.26],
                [pooled[0] / len(finals)] * 2,
                color=_COLORS[arm_name],
                linewidth=2.6,
            )
        ax_spread.set_xticks(range(len(_ARMS)))
        ax_spread.set_xticklabels(_ARMS, rotation=20, ha="right", fontsize=8)
        ax_spread.set_xlabel("one dot per seed; bar is the pooled mean")
        ax_spread.set_ylim(-seed_total * 0.04, seed_total * 1.06)
        ax_spread.grid(alpha=0.25, linewidth=0.6, axis="y")
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    @staticmethod
    def render_interventions(*, arms: dict, output: Path, title: str) -> None:
        """What each arm actually cost in human help, **per seed**.

        This is the cost side of the experiment, and it existed only as two pooled totals
        in prose before this figure. Pooled totals are exactly what hides the thing worth
        seeing: `359` and `348` are sums over ten seeds, and a sum cannot distinguish ten
        seeds charged ~35 each from one seed charged 300 and nine charged 6. The left
        panel therefore plots one dot per seed.

        **Left**: per-seed intervention counts, one dot per seed, with the pooled total
        written under each arm as `x` over its ten seeds. The four arms with no reachable
        human sit on the zero line.

        **Right**: what each intervention bought, as final score against interventions,
        pooled per arm -- the cost-effectiveness view. An arm high and left did well
        cheaply; an arm high and right did well expensively. The `on-stuck` and
        `at-random` arms are the pair to read against each other here: they are configured
        to spend at comparable rates, so a vertical gap between them at similar x is the
        timing effect priced per rescue."""
        fig, (ax_counts, ax_value) = plt.subplots(1, 2, figsize=(13.6, 5.2))
        for index, arm_name in enumerate(_ARMS):
            arm = arms[arm_name]
            seeds = sorted(arm)
            counts = [arm[seed]["interventions"] for seed in seeds]
            # Deterministic jitter so coincident seeds stay countable and the figure is
            # reproducible -- index-derived, never drawn from an RNG.
            offsets = [(i % 5 - 2) * 0.06 for i in range(len(counts))]
            ax_counts.scatter(
                [index + offset for offset in offsets],
                counts,
                color=_COLORS[arm_name],
                s=44,
                alpha=0.85,
                zorder=3,
            )
            ax_counts.plot(
                [index - 0.26, index + 0.26],
                [sum(counts) / len(counts)] * 2,
                color=_COLORS[arm_name],
                linewidth=2.6,
            )
            # Above the column, not below the axis: below collides with the rotated tick
            # labels, and a pooled total half-hidden behind an arm name is worse than none.
            # "N in 10 seeds", not "N/10": an intervention count is a total, not a
            # proportion, and `x/y` here would read as a success rate out of ten.
            ax_counts.annotate(
                f"{sum(counts)} in {len(counts)} seeds",
                (index, 0.97),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color=_COLORS[arm_name],
            )
        ax_counts.set_xticks(range(len(_ARMS)))
        ax_counts.set_xticklabels(_ARMS, rotation=20, ha="right", fontsize=8)
        ax_counts.set_ylabel("human interventions charged to this seed", fontsize=9)
        ax_counts.set_xlabel(
            "one dot per seed; bar is the per-seed mean, bold figure is the pooled total",
            fontsize=8.5,
        )
        ax_counts.grid(alpha=0.25, linewidth=0.6, axis="y")
        # Headroom for the pooled totals written along the top, so they never sit on top
        # of the highest seed's dot.
        busiest = max(
            arms[arm_name][seed]["interventions"] for arm_name in _ARMS for seed in arms[arm_name]
        )
        ax_counts.set_ylim(-2, max(busiest * 1.22, 1.0))

        # Sizes descend with arm order so that two arms landing on the same point stay
        # individually visible as nested rings rather than one hiding the other. The four
        # zero-rescue arms all sit at x=0 and can collide there, and a figure in which one
        # silently covers another would erase it.
        placed: dict[tuple[int, int], list[str]] = {}
        for index, arm_name in enumerate(_ARMS):
            arm = arms[arm_name]
            seeds = sorted(arm)
            pooled_solved, pooled_total = HumanLadderCurves.pooled_curve(arm=arm, family=None)[-1]
            rescues = sum(arm[seed]["interventions"] for seed in seeds)
            placed.setdefault((rescues, pooled_solved), []).append(arm_name)
            ax_value.scatter(
                [rescues],
                [pooled_solved],
                color=_COLORS[arm_name],
                s=200 - index * 22,
                alpha=0.9,
                zorder=3 + index,
                edgecolors="white",
                linewidths=0.8,
                label=(
                    f"{_LABELS[arm_name]} — "
                    f"{HumanLadderCurves.format_count(solved=pooled_solved, total=pooled_total)}"
                ),
            )
        for (rescues, solved), names in placed.items():
            if len(names) > 1:
                ax_value.annotate(
                    " = ".join(names),
                    (rescues, solved),
                    textcoords="offset points",
                    xytext=(11, -3),
                    fontsize=7.5,
                    style="italic",
                )
        ax_value.set_xlabel("human interventions charged, pooled over 10 seeds", fontsize=9)
        ax_value.set_ylabel(
            f"final test tasks solved, pooled (x/{_NUM_TEST_TASKS * 10})", fontsize=9
        )
        ax_value.grid(alpha=0.25, linewidth=0.6)
        ax_value.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    # ------------------------------------------------------------------ entry point

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            metavar="NAME=DIR",
            help="e.g. stuck-initial=results/human-ladder/stuck-initial/ees . DIR holds "
            f"<seed>/stats.json. All eight of {', '.join(_ARMS)} are required.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Where the twelve figures are written: one per arm, three cross-arm "
            "figures split by goal family (overall / TRASH / RECYCLING), and the per-seed "
            "intervention-count figure. EMPTY gets no figure: it is 20/20 in every arm.",
        )
        args = parser.parse_args()

        directories = {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            directories[name] = Path(path)
        arms = HumanLadderCurves.load_arms(directories=directories)
        HumanLadderCurves.print_report(arms=arms)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        domain = "Tossing Room (split throws, weight drawn at pickup), reset-free practice"
        for arm_name in _ARMS:
            HumanLadderCurves.render_arm(
                arms=arms,
                arm_name=arm_name,
                output=args.output_dir / f"human-ladder-{arm_name}.png",
                title=f"{domain}\n{_LABELS[arm_name]}: all test tasks, x/{_NUM_TEST_TASKS * 10}",
            )
        # Figure order follows the argument, not the data model: the two throw families
        # carry the mechanism (the whole effect is RECYCLING, TRASH is a null result) and
        # the pooled curve is the summary of them. Matches reset_free_ledge_curves.py.
        # EMPTY is deliberately absent -- 20/20 in every arm, nothing to show.
        for family, name, legend_loc in (
            ("TRASH", f"TRASH tasks, x/{_COMPOSITION['TRASH'] * 10}", "lower right"),
            ("RECYCLING", f"RECYCLING tasks, x/{_COMPOSITION['RECYCLING'] * 10}", "upper left"),
            (None, f"all test tasks, x/{_NUM_TEST_TASKS * 10}", "upper left"),
        ):
            HumanLadderCurves.render_family(
                arms=arms,
                family=family,
                output=args.output_dir
                / f"human-ladder-{'overall' if family is None else family.lower()}.png",
                title=f"{domain}\neight arms of the human-in-the-loop ladder — {name}",
                legend_loc=legend_loc,
            )
        HumanLadderCurves.render_interventions(
            arms=arms,
            output=args.output_dir / "human-ladder-interventions.png",
            title=f"{domain}\nwhat each arm cost in human help, per seed",
        )


if __name__ == "__main__":
    HumanLadderCurves.main()
