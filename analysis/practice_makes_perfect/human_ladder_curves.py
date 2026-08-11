"""Post-run analysis for a scoped human-in-the-loop measurement on Tossing Room: three
fixed arms (a no-human control and two ceilings) plus a rescue-rate dose-response sweep.

**Background.** `--practice-reset-policy never` is the real-robot condition -- a robot
practising in a lab is not teleported to a fresh start every few minutes. Measured on this
domain, it is also badly damaged: Tossing Room's one-way ledge severs rooms 0-2 from the
item pile in room 3, so a practice period that steps left once can never pick anything up
again, and under `never` that damage carries into every later period.

**This is a re-measurement of PR #195's own rate sweep** (still open, not merged, as of
this writing -- https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/195),
fixing two things Josh flagged in review there: (1) #195's `--num-cycles 10` budget had not
converged -- N=14's pooled OVERALL climbed `77, 100, ..., 259` across its 11 checkpoints,
still rising at the last one -- so this sweep runs **`--num-cycles 100`**, 10x longer; (2)
#195's `--ask-for-help at-random` trigger draws one RNG sample per policy call
(`Bernoulli(1/N)`), so two runs at the same N can still land a different number of actual
requests, confounding "what does the rate do" with "what did the RNG happen to draw". This
sweep instead uses `--ask-for-help at-fixed-interval`
(https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/200), which fires
on exactly every Nth policy call with zero RNG draws, so the request count is a deterministic
function of N and the run length alone. Per Josh's explicit instruction, N < 5 is dropped
(the old sweep's N=1,2,3 points) and the grid is extended up to N=30 (the old sweep's max
was 20).

**The four components.**

| component | `--method` | `--ask-for-help` | `--human-reset-target` | world | seeds |
| --- | --- | --- | --- | --- | --- |
| `no-human` | `ees` | `never` | -- | one-way | 10 |
| `two-way-ledge` | `ees` | `never` | -- | two-way | 10 |
| `skill-oracle` | `skill-oracle` | -- | -- | one-way | 10 (reused from #195 -- see below) |
| rate sweep | `ees` | `at-fixed-interval` | `task-initial` | one-way | 10 per N |

The rate sweep varies `--mean-steps-between-help-requests` (N; a request fires on exactly
every Nth policy call, deterministically) over N in {5, 7, 10, 14, 20, 25, 30} -- denser at
low N, where the response is expected to move fastest, sparser toward N=30 where it is
expected to have mostly flattened.

**`skill-oracle` is reused unchanged from #195, not re-run.**
`methods/oracle/cli.py`'s `SkillOracleCli.add_arguments` registers no flags at all, and its
`run` hardcodes `num_cycles=0` regardless of what the global CLI parsed -- confirmed
empirically: `--method skill-oracle --num-cycles 100` errors `unrecognized arguments:
--num-cycles 100`. An oracle never practises/learns over cycles, so its result is
cycle-count-invariant, and the code paths it depends on (`environments/tossingroom/`,
`methods/oracle/`, `core/`, `practice_loop.py`) are unchanged between #195's fork point and
this sweep's. Re-running it would reproduce #195's own 10 seeds byte-for-byte at real
compute cost, so this analysis reads them from
`docs/experiment-logs/2026-08-10-human-ladder-rate-sweep-runs/skill-oracle/` (committed on
#195's branch) unchanged.

**Which comparisons are clean.** `no-human` and every rate-sweep point share `--method ees`,
the one-way world and all ten seeds, so `PairedTests.sign_flip` applies to each N against the
control. `two-way-ledge` changes the world and `skill-oracle` changes the Method, so neither
is sign-flipped against anything -- each is reported as a ceiling level only, the same
precedent #195's own module used.

**Non-learners are drawn flat, not as curves.** `skill-oracle` never practises (no
`--num-cycles` flag exists for it) and has a single evaluation checkpoint, so it is a
horizontal reference line. `no-human` and `two-way-ledge` both learn and get real curves;
`two-way-ledge` is grey, matching `skill-oracle`, because it IS a ceiling arm in CLAUDE.md's
sense (not the manipulation under test -- it removes irreversibility from the world, not
"does an assistance mechanism exist"), distinguished from `skill-oracle` by linestyle rather
than a fourth hue (CLAUDE.md: "do not introduce a fourth hue; encode a second axis with
linestyle instead" -- no stated exception for a ceiling arm that still learns). It keeps its
own curve rather than being flattened, since flattening it would misreport a real learner as
a constant (see `_REFERENCE_ARMS`'s own comment) -- grey colour and a real curve are
independent choices, and this module makes both explicitly rather than letting the second
follow from the first.

**Colour.** `no-human` is orange (`#D55E00`): the standing "nothing helps" colour, reused
across every figure in this report. The rate sweep is blue-family (`Blues`, a sequential
colourmap from light N=5 to dark N=30): every rate-sweep arm has an assistance mechanism
available (`--ask-for-help at-fixed-interval`, always firing at a strictly positive rate
here), which is what the blue/orange rule tracks -- not the specific rate, which the
colourmap's lightness carries instead. `skill-oracle` is grey and dotted (reference line);
`two-way-ledge` is ALSO grey (a deliberate fix over #195's module, which gave it a fourth,
unreserved hue -- magenta), distinguished from `skill-oracle` by its own dash pattern
rather than a colour CLAUDE.md's rule does not grant it.

**The three training-curve figures (overall/TRASH/RECYCLING) carry all ten arms on ONE
panel each** -- the three fixed arms plus all seven rate-sweep points, so a reader sees the
whole ladder's shape over practice, not just its final-checkpoint dose-response. The seven
rate-sweep curves are thin, partly transparent and have no per-seed traces of their own (that
would be 70 more lines); they are drawn first so the three fixed arms' bold, per-seed-backed
curves stay visually on top. See `render_family` for why a colourbar replaces a named legend
entry for the seven of them.

**The separate dose-response figure is not a training curve, and stays alongside the merged
panels rather than being replaced by them.** It answers a different question -- what does
FINAL performance do as the rescue rate varies -- so its x axis is N, not online transitions,
and it draws one point per seed at each of the seven sampled N values, plus the pooled mean,
with the `no-human` and `skill-oracle` levels as reference lines for context. The merged
training-curve panels show shape over practice; this figure is still the one to read for the
exact per-N numbers the training curves intentionally omit from their (colourbar-only) legend.

**The manipulation checks are not optional here.** `num_practice_resets` must be 0
everywhere, or an arm labelled reset-free was quietly reset for free.
`num_human_interventions_recorded` must be exactly 0 for the three arms with no reachable
human (`no-human`, `two-way-ledge`, `skill-oracle`) and strictly positive for every
rate-sweep point -- at N <= 30 over 15000 policy calls (`--num-cycles 100 --max-steps-
per-interaction 150`) the deterministic trigger fires at least 500 times, so a zero there
would mean the trigger never wired rather than a legitimate null.

**Statistics.** Every `no-human`-paired comparison is `PairedTests.sign_flip`, exact by
enumerating its null in full -- no normal approximation, no scipy. Imported from
`paired_tests` rather than reimplemented; goal classification comes from
`goal_families.GoalFamilies` for the same reason.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs a
simulation or drives a `Method`). Each `--arm` points at the directory holding that arm's
`<seed>/stats.json`; each `--rate-point` does the same for one N in the sweep.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# The three fixed arms, in report order: the control, then the two ceilings.
_FIXED_ARMS = ("no-human", "two-way-ledge", "skill-oracle")

# Non-learners, drawn as a horizontal reference line at their pooled level rather than a
# curve. `skill-oracle` never practises at all. `two-way-ledge` is deliberately NOT here --
# it is EES and it genuinely learns, so flattening it would misreport it (checked in #151's
# module, unaffected by anything this file changes).
_REFERENCE_ARMS = ("skill-oracle",)

# Arms with no reachable human at all, which must therefore record zero interventions.
_ARMS_WITHOUT_A_HUMAN = ("no-human", "two-way-ledge", "skill-oracle")

# The composition the domain allocates for --num-test-tasks 30. Asserted per sweep, because
# a goal misfiled between families moves tasks between denominators invisibly.
_COMPOSITION = {"TRASH": 14, "RECYCLING": 14, "EMPTY": 2}
_NUM_TEST_TASKS = sum(_COMPOSITION.values())

# Every arm is --practice-reset-policy never, so a free harness reset anywhere means the
# arm is not what its name says.
_EXPECTED_PRACTICE_RESETS = 0

# humans/oracle.py's UnconditionalHumanOracle charges a flat 1.0 per rescue, so cost and
# count are proportional at v0. Checked rather than assumed.
_V0_INTERVENTION_COST = 1.0

# Colour carries role, per CLAUDE.md's training-curve-style convention: orange is the arm
# nothing helps, and it is reused here from every other figure in the project rather than
# picked fresh. `two-way-ledge` is grey too -- CLAUDE.md reserves grey for "reference/
# ceiling arms that aren't the manipulation under test" and separately says "do not
# introduce a fourth hue; encode a second axis with linestyle instead", with no stated
# carve-out for a ceiling that happens to still learn. #195's module gave it a fourth hue
# (magenta) on the reasoning that it is "a ceiling on the world, not a non-learning
# oracle" -- a real distinction, but the literal rule doesn't grant it a new colour for
# that, only a linestyle: `two-way-ledge` keeps its own dash pattern
# (distinct from `skill-oracle`'s dotted), which is exactly the "second axis via
# linestyle" the rule asks for. It is still NOT in `_REFERENCE_ARMS` and still drawn as a
# real curve, not flattened -- only its hue changed, not its rendering treatment (see
# `_REFERENCE_ARMS`'s own comment for why flattening it would misreport it).
_COLORS = {
    "no-human": "#D55E00",
    "two-way-ledge": "#7F7F7F",
    "skill-oracle": "#7F7F7F",
}
_LINESTYLES = {
    "no-human": "-",
    "two-way-ledge": (0, (3, 1)),
    "skill-oracle": ":",
}
_LABELS = {
    "no-human": "EES, no human (control)",
    "two-way-ledge": "EES, two-way ledge (ceiling: no irreversibility)",
    "skill-oracle": "skill oracle (ceiling: skills)",
}

# Blue: the rate sweep has an assistance mechanism (--ask-for-help at-fixed-interval)
# available and firing at every sampled N -- the blue/orange rule tracks whether the
# mechanism exists, not how often it fires.
_RATE_SWEEP_COLOR = "#0072B2"


class HumanLadderCurves:
    """A static-method container, never instantiated."""

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def format_count(*, solved: int, total: int) -> str:
        """`x/y`, never a bare percentage: the denominators here are small and uneven."""
        return f"{solved}/{total}"

    @staticmethod
    def load_arms(*, directories: dict[str, Path]) -> dict[str, dict]:
        """The three fixed arms' per-seed data, with the checks first.

        All three are required. Without `no-human` there is no control to difference
        against; without a ceiling the remaining-gap arithmetic has no ceiling. A report
        that silently printed whatever it could still make would read as a result."""
        missing = [arm for arm in _FIXED_ARMS if arm not in directories]
        if missing:
            raise ValueError(
                f"missing arm(s): {', '.join(missing)}. This comparison is three fixed arms "
                "sharing one seed set; with one absent the comparisons it exists to make "
                "are not defined."
            )
        return {
            arm: HumanLadderCurves.load_run(
                directory=directories[arm],
                label=arm,
                expect_no_human=arm in _ARMS_WITHOUT_A_HUMAN,
            )
            for arm in _FIXED_ARMS
        }

    @staticmethod
    def load_rate_sweep(*, directories: dict[int, Path]) -> dict[int, dict]:
        """The rate sweep's per-seed data, keyed by N (`--mean-steps-between-help-requests`).

        At least two points are required for a "sweep" to mean anything -- one point is a
        single arm, not a dose-response."""
        if len(directories) < 2:
            raise ValueError(
                f"rate sweep needs at least two N values to show a dose-response, got "
                f"{sorted(directories)}."
            )
        return {
            n: HumanLadderCurves.load_run(
                directory=directories[n], label=f"N={n}", expect_no_human=False
            )
            for n in sorted(directories)
        }

    @staticmethod
    def load_run(*, directory: Path, label: str, expect_no_human: bool) -> dict:
        """One arm/N-point: `{seed: {"transitions", "families", "overall",
        "interventions", "human_cost"}}`."""
        seeds = sorted(int(path.parent.name) for path in directory.glob("*/stats.json"))
        if not seeds:
            raise ValueError(f"no <seed>/stats.json under {directory}")
        loaded: dict[int, dict] = {}
        for seed in seeds:
            stats = json.loads((directory / str(seed) / "stats.json").read_text())
            HumanLadderCurves.check_manipulation(
                stats=stats,
                label=label,
                where=f"{directory}/{seed}",
                expect_no_human=expect_no_human,
            )
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
    def check_manipulation(*, stats: dict, label: str, where: str, expect_no_human: bool) -> None:
        """The things that make a run's label true, checked before any number off it is
        used.

        `expect_no_human` is a hard requirement in BOTH directions here, matching #195's
        own module: a fixed arm recording an intervention is a wiring error, but so is a
        rate-sweep point recording ZERO -- at N <= 30 over 15000 policy calls
        (`--num-cycles 100 --max-steps-per-interaction 150`) the deterministic
        `at-fixed-interval` trigger fires at least 500 times by construction (it consumes
        no RNG, so this is exact, not merely expected), so a true zero means the trigger
        never fired rather than a legitimate null (contrast `on-stuck`, which #151's
        module correctly let report zero as a finding -- that trigger is
        condition-dependent, `at-fixed-interval` is not)."""
        resets = stats.get("num_practice_resets")
        if resets != _EXPECTED_PRACTICE_RESETS:
            raise ValueError(
                f"{where}: num_practice_resets is {resets}, expected "
                f"{_EXPECTED_PRACTICE_RESETS}. Every run here is --practice-reset-policy "
                "never, so this run was quietly reset for free and its practice is not "
                "reset-free."
            )
        interventions = stats.get("num_human_interventions_recorded", 0)
        cost = stats.get("summed_human_cost_recorded", 0.0)
        if expect_no_human and interventions:
            raise ValueError(
                f"{where}: the {label} run recorded {interventions} human interventions. "
                "It has no reachable human and must never call one."
            )
        if not expect_no_human and not interventions:
            raise ValueError(
                f"{where}: the {label} run recorded zero human interventions. At this "
                "rate over 15000 policy calls that means --ask-for-help never wired, not "
                "a legitimate null."
            )
        if abs(cost - interventions * _V0_INTERVENTION_COST) > 1e-9:
            raise ValueError(
                f"{where}: summed_human_cost_recorded is {cost} against {interventions} "
                f"interventions, which is not the flat {_V0_INTERVENTION_COST} the v0 "
                "oracle charges. A different HumanOracle was wired, so every cost number "
                "here means something else."
            )

    @staticmethod
    def sweep_counts(*, outcomes: list[dict]) -> dict[str, tuple[int, int]]:
        """One sweep's `(solved, total)` per family.

        Classification is `GoalFamilies.classify`, reused rather than recopied: it tests
        the `BinEmpty` predicate before the item names, because `Goal.describe()` renders
        EMPTY naming BOTH bins, so a naive "does it mention recycling?" test swallows it and
        silently reports 16 RECYCLING / 0 EMPTY."""
        solved: Counter[str] = Counter()
        total: Counter[str] = Counter()
        for outcome in outcomes:
            family = GoalFamilies.classify(goal=outcome["goal"])
            total[family] += 1
            solved[family] += int(outcome["solved"])
        return {family: (solved[family], total[family]) for family in total}

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def entry(*, run: dict, seed: int, family: str | None) -> list[tuple[int, int]]:
        """One seed's `(solved, total)` per checkpoint, overall or for one family."""
        return run[seed]["overall"] if family is None else run[seed]["families"][family]

    @staticmethod
    def pooled_curve(*, run: dict, family: str | None) -> list[tuple[int, int]]:
        """The run's curve pooled over seeds: solved and total both SUMMED, per checkpoint.

        Summed rather than averaged, so `x/300` at ten seeds means what it says. A mean of
        per-seed rates would silently reweight a seed that ran a different number of
        tasks."""
        seeds = sorted(run)
        num_checkpoints = len(run[seeds[0]]["transitions"])
        pooled = []
        for index in range(num_checkpoints):
            solved = 0
            total = 0
            for seed in seeds:
                entry = HumanLadderCurves.entry(run=run, seed=seed, family=family)
                solved += entry[index][0]
                total += entry[index][1]
            pooled.append((solved, total))
        return pooled

    @staticmethod
    def transitions(*, run: dict) -> list[int]:
        """The pooled x axis: each checkpoint's MEAN transition count over the seeds.

        A rescue consumes its loop iteration (`PracticeLoop`'s `except
        HumanHelpRequested` branch `continue`s), so a rescued seed reaches every later
        checkpoint one transition earlier per rescue and seeds do not share a grid on any
        run that asks for help. Only a differing NUMBER of checkpoints is rejected -- that
        means a seed ran a different number of cycles, which makes the curves
        incommensurable outright."""
        lengths = {len(run[seed]["transitions"]) for seed in sorted(run)}
        if len(lengths) != 1:
            raise ValueError(
                f"seeds disagree on the number of evaluation checkpoints ({sorted(lengths)})."
            )
        seeds = sorted(run)
        return [
            round(sum(run[seed]["transitions"][index] for seed in seeds) / len(seeds))
            for index in range(next(iter(lengths)))
        ]

    @staticmethod
    def convergence_summary(*, run: dict, family: str | None = None) -> dict:
        """Whether this run's pooled curve had flattened by its last checkpoint --
        pre-registered here as a fixed rule (Methods, not Results) so the PR's answer to
        "did it converge" is not an eyeballed call made after seeing the plot.

        **The rule**: pool solved/total over the last 10 checkpoints and the 10 before
        that, take each window's solved-over-total FRACTION (not a raw count -- windows
        of counts are not comparable if a checkpoint's own total differs, which it does
        not here but the fraction is the right invariant regardless), and report their
        difference. No p-value is attached: this is a two-number descriptive comparison
        of adjacent windows on one already-pooled curve, not a hypothesis test over
        independent samples, and manufacturing a test statistic for it would claim a
        kind of inference this comparison cannot support. Report the raw numbers and let
        the reader judge "close to flat" against the run's own scale.

        **Threshold, fixed here before any real number was seen**: `|delta| < 0.01`
        (one percentage point of the pooled fraction, i.e. < 3/300 at this domain's
        --num-test-tasks 30 x 10 seeds) is called FLAT; anything at or above it is
        called STILL CLIMBING (or falling). Chosen as a round, small number relative to
        the run's own scale, not tuned to whatever this PR's data turned out to show.

        Raises rather than silently truncating a short run: below 20 checkpoints the two
        windows would overlap or run off the front of the curve, which is not what
        "last 10 vs. the 10 before" means."""
        pooled = HumanLadderCurves.pooled_curve(run=run, family=family)
        if len(pooled) < 20:
            raise ValueError(
                f"convergence_summary needs >= 20 checkpoints (last 10 vs. previous 10), "
                f"got {len(pooled)}."
            )
        last10 = pooled[-10:]
        prev10 = pooled[-20:-10]
        last_fraction = sum(s for s, _ in last10) / sum(t for _, t in last10)
        prev_fraction = sum(s for s, _ in prev10) / sum(t for _, t in prev10)
        return {
            "prev10_fraction": prev_fraction,
            "last10_fraction": last_fraction,
            "delta": last_fraction - prev_fraction,
            "final": pooled[-1],
        }

    @staticmethod
    def final_per_seed(*, run: dict, family: str | None) -> list[int]:
        """Each seed's final-checkpoint solved count, in seed order."""
        return [
            HumanLadderCurves.entry(run=run, seed=seed, family=family)[-1][0]
            for seed in sorted(run)
        ]

    @staticmethod
    def paired_final_differences(
        *, treatment: dict, control: dict, family: str | None
    ) -> list[float]:
        """`treatment` minus `control` at the final checkpoint, **within a seed**.

        The runs share seeds, so this is paired data and an unpaired test would throw that
        structure away. Zero differences are KEPT rather than dropped."""
        seeds = sorted(set(treatment) & set(control))
        return [
            float(
                HumanLadderCurves.entry(run=treatment, seed=seed, family=family)[-1][0]
                - HumanLadderCurves.entry(run=control, seed=seed, family=family)[-1][0]
            )
            for seed in seeds
        ]

    @staticmethod
    def solves_per_rescue(*, treatment: dict, control: dict) -> float | None:
        """Extra tasks solved per human rescue spent: the gap divided by what it cost.

        `None`, never a number, when the treatment spent nothing -- that is a division by
        zero, and reporting it as `inf` or `0` would both read as findings. Every
        rate-sweep point spends something by construction (checked on load), so this is
        only ever `None` if called on a fixed arm."""
        rescues = sum(treatment[seed]["interventions"] for seed in sorted(treatment))
        if not rescues:
            return None
        gap = sum(
            HumanLadderCurves.paired_final_differences(
                treatment=treatment, control=control, family=None
            )
        )
        return gap / rescues

    # ------------------------------------------------------------------ the report

    @staticmethod
    def print_report(*, arms: dict, rate_sweep: dict) -> None:
        """Every number the write-up quotes, as `x/y`, re-derived here."""
        print("fixed arms: final-checkpoint scores, pooled over seeds\n")
        for family in (None, "TRASH", "RECYCLING", "EMPTY"):
            name = "OVERALL" if family is None else family
            print(f"  {name}")
            for arm_name in _FIXED_ARMS:
                final = HumanLadderCurves.pooled_curve(run=arms[arm_name], family=family)[-1]
                per_seed = HumanLadderCurves.final_per_seed(run=arms[arm_name], family=family)
                print(
                    f"    {arm_name:>14}  "
                    f"{HumanLadderCurves.format_count(solved=final[0], total=final[1]):>8}"
                    f"   per-seed {min(per_seed)}-{max(per_seed)}"
                )
            print()

        print("rate sweep: interventions spent and final OVERALL score, per N\n")
        for n in sorted(rate_sweep):
            run = rate_sweep[n]
            seeds = sorted(run)
            interventions = [run[seed]["interventions"] for seed in seeds]
            final = HumanLadderCurves.pooled_curve(run=run, family=None)[-1]
            per_seed = HumanLadderCurves.final_per_seed(run=run, family=None)
            differences = HumanLadderCurves.paired_final_differences(
                treatment=run, control=arms["no-human"], family=None
            )
            test = PairedTests.sign_flip(differences=differences)
            better = sum(1 for d in differences if d > 0)
            worse = sum(1 for d in differences if d < 0)
            mde = PairedTests.minimum_detectable_effect(differences=differences)
            ratio = HumanLadderCurves.solves_per_rescue(treatment=run, control=arms["no-human"])
            # Every rate-sweep point spends at least one rescue by construction (checked
            # on load), so `ratio` is never `None` here -- the `n/a` guard exists anyway
            # so a future caller of this same print loop cannot crash on a fixed arm.
            shown_ratio = "n/a (never rescued)" if ratio is None else f"{ratio:.3f}"
            print(
                f"  N={n:<3}  {sum(interventions):>5} interventions pooled"
                f"   (per-seed {min(interventions)}-{max(interventions)})"
                f"   OVERALL {HumanLadderCurves.format_count(solved=final[0], total=final[1]):>8}"
                f"   per-seed {min(per_seed)}-{max(per_seed)}"
            )
            print(
                f"        vs no-human: gap {int(sum(differences)):>+4}"
                f"   better {better}/{len(differences)}  worse {worse}/{len(differences)}"
                f"   tied {test.num_zero_differences}/{len(differences)}"
                f"   p = {test.p_value:.4g}   MDE {mde:.2f}"
                f"   {shown_ratio} extra solves per rescue"
            )
        print()

        print(
            "convergence check (pre-registered): pooled OVERALL fraction, last 10 "
            "checkpoints vs. the 10 before that\n"
        )
        for arm_name in ("no-human", "two-way-ledge"):
            summary = HumanLadderCurves.convergence_summary(run=arms[arm_name])
            final_solved, final_total = summary["final"]
            print(
                f"    {arm_name:>14}  prev10 {summary['prev10_fraction']:.4f}"
                f"   last10 {summary['last10_fraction']:.4f}"
                f"   delta {summary['delta']:+.4f}"
                f"   final "
                f"{HumanLadderCurves.format_count(solved=final_solved, total=final_total)}"
            )
        for n in sorted(rate_sweep):
            summary = HumanLadderCurves.convergence_summary(run=rate_sweep[n])
            final_solved, final_total = summary["final"]
            print(
                f"    N={n:<10}  prev10 {summary['prev10_fraction']:.4f}"
                f"   last10 {summary['last10_fraction']:.4f}"
                f"   delta {summary['delta']:+.4f}"
                f"   final "
                f"{HumanLadderCurves.format_count(solved=final_solved, total=final_total)}"
            )
        print()

    # ------------------------------------------------------------------ the figures

    @staticmethod
    def render_family(
        *,
        arms: dict,
        rate_sweep: dict,
        family: str | None,
        output: Path,
        title: str,
        legend_loc: str = "upper left",
    ) -> None:
        """The three fixed arms AND all seven rate-sweep points on ONE goal family: the
        training curves, all on the same panel.

        One figure per family (overall / TRASH / RECYCLING), matching the standing
        convention -- a pooled curve would average a large per-family effect against a
        flat one and show a muted version of neither. EMPTY gets no figure: 20/20 in every
        arm, nothing for a curve to show.

        **Ten lines on one panel needs a different legend strategy than three.** The
        three fixed arms keep named legend entries with their exact pooled count, per the
        standing convention. The seven rate-sweep arms do NOT get named entries -- seven
        more `--mean-steps-between-help-requests=N -- x/300` strings would make the legend
        itself the unreadable part of the figure. They get a sequential colourmap instead
        (light N=5 to dark N=30) with a colourbar, which is the right encoding for an
        ORDERED sweep of one parameter -- Josh's own suggestion, and the natural read for
        "one arm per value of N" the way a discrete named legend is not. All seven are
        blue-family (`Blues`), matching the project's role rule: every rate-sweep arm has
        an assistance mechanism available (`--ask-for-help at-fixed-interval`, always
        firing at a strictly positive rate here -- see `check_manipulation`), which is
        what blue encodes, not the specific rate. Orange stays reserved for `no-human`,
        the one arm with no mechanism at all.

        **The rate-sweep curves have no per-seed faint traces here, unlike the fixed
        arms.** Seven arms x ten seeds is 70 more lines, which would bury the panel; their
        per-seed spread already has its own figure (`render_rate_sweep`). They are drawn
        first, thin and partly transparent, so the three fixed arms' bold lines stay on
        top and visually dominant -- the same "faint underneath, bold on top" layering the
        project's per-seed traces use, applied here to a whole population of secondary
        arms instead of one arm's seeds."""
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        seed_total = HumanLadderCurves.entry(
            run=arms[_FIXED_ARMS[0]], seed=sorted(arms[_FIXED_ARMS[0]])[0], family=family
        )[-1][1]

        ns = sorted(rate_sweep)
        cmap = matplotlib.colormaps["Blues"]
        norm = matplotlib.colors.Normalize(vmin=min(ns), vmax=max(ns))
        for n in ns:
            run = rate_sweep[n]
            # 0.30 floor keeps even the lightest sampled N visible against a white panel.
            color = cmap(0.30 + 0.65 * norm(n))
            xs = HumanLadderCurves.transitions(run=run)
            pooled = HumanLadderCurves.pooled_curve(run=run, family=family)
            scale = seed_total / pooled[-1][1]
            ax.plot(
                xs,
                [solved * scale for solved, _ in pooled],
                color=color,
                linewidth=1.5,
                alpha=0.85,
                zorder=2,
            )
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.015, fraction=0.045)
        colorbar.set_label(
            "--ask-for-help at-fixed-interval, N (--mean-steps-between-help-requests) "
            "-- exact period, no RNG",
            fontsize=8,
        )
        colorbar.set_ticks(ns)

        for arm_name in _FIXED_ARMS:
            run = arms[arm_name]
            color = _COLORS[arm_name]
            linestyle = _LINESTYLES[arm_name]
            xs = HumanLadderCurves.transitions(run=run)
            pooled = HumanLadderCurves.pooled_curve(run=run, family=family)
            scale = seed_total / pooled[-1][1]
            # CLAUDE.md's own legend-entry example is "env resets -- mean, n=10": the
            # seed count, not a score, is what lets a reader check n sums to the seed
            # total without re-deriving it from the plot. The final pooled score is
            # still useful, so it stays too -- just after the n=, never replacing it.
            label = (
                f"{_LABELS[arm_name]} — mean, n={len(run)} "
                f"({HumanLadderCurves.format_count(solved=pooled[-1][0], total=pooled[-1][1])})"
            )
            if arm_name in _REFERENCE_ARMS:
                for seed in sorted(run):
                    entry = HumanLadderCurves.entry(run=run, seed=seed, family=family)
                    ax.axhline(
                        entry[-1][0], color=color, linestyle=linestyle, alpha=0.22, linewidth=0.9
                    )
                ax.axhline(
                    pooled[-1][0] * scale,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.6,
                    label=label,
                    zorder=4,
                )
                continue
            for seed in sorted(run):
                entry = HumanLadderCurves.entry(run=run, seed=seed, family=family)
                ax.plot(
                    run[seed]["transitions"],
                    [s for s, _ in entry],
                    color=color,
                    linestyle=linestyle,
                    alpha=0.16,
                    linewidth=0.8,
                    zorder=3,
                )
            ax.plot(
                xs,
                [solved * scale for solved, _ in pooled],
                color=color,
                linestyle=linestyle,
                linewidth=2.3,
                label=label,
                zorder=4,
            )
        ax.set_xlabel("online transitions")
        ax.set_ylabel("test tasks solved per seed", fontsize=9)
        ax.set_ylim(-seed_total * 0.04, seed_total * 1.06)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8.5, loc=legend_loc, framealpha=0.95)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    @staticmethod
    def render_rate_sweep(*, arms: dict, rate_sweep: dict, output: Path, title: str) -> None:
        """The dose-response figure: mean + variance of final OVERALL score against the
        rescue-rate knob N.

        Not a training curve -- there is no online-transitions axis here, since each point
        is a wholly separate arm at a different N, not one arm's progress over time. One
        faint dot per seed at each sampled N (jittered so ties stay countable), a bold mean
        line connecting the pooled-per-seed average at each N, and the `no-human` /
        `skill-oracle` levels as reference lines so the sweep reads against the same
        ceilings the fixed-arm figures use.

        **The shaded band is the 25th-75th percentile (IQR) across the 10 seeds at each N,
        not a symmetric +-1 std band.** #195's own (unconverged, `at-random`) sweep found
        genuine bimodality at some N -- final checkpoint splitting into a high cluster and
        a low cluster rather than a noisy unimodal spread -- and a symmetric std band
        centred on the mean would visually assert a single-peaked distribution that isn't
        there, while also having no reason to respect the [0, 30] physical range. The IQR
        is a plain robust spread statistic that does neither, and whether this run's own
        deterministic-trigger, 100-cycle data reproduces that bimodality is exactly one of
        the questions this PR checks -- see `render_rate_sweep_trajectories`, which draws
        the individual seed trajectories the IQR band alone cannot reveal."""
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        ns = sorted(rate_sweep)
        means = []
        q1s = []
        q3s = []
        for n in ns:
            run = rate_sweep[n]
            finals = HumanLadderCurves.final_per_seed(run=run, family=None)
            offsets = [(i % 5 - 2) * 0.12 for i in range(len(finals))]
            ax.scatter(
                [n + offset for offset in offsets],
                finals,
                color=_RATE_SWEEP_COLOR,
                s=34,
                alpha=0.35,
                zorder=2,
            )
            means.append(sum(finals) / len(finals))
            quartiles = statistics.quantiles(finals, n=4, method="inclusive")
            q1s.append(quartiles[0])
            q3s.append(quartiles[2])
        ax.fill_between(
            ns,
            q1s,
            q3s,
            color=_RATE_SWEEP_COLOR,
            alpha=0.16,
            zorder=1,
            label="IQR (25th-75th pctile) across 10 seeds per N",
        )
        ax.plot(
            ns,
            means,
            color=_RATE_SWEEP_COLOR,
            linewidth=2.3,
            marker="o",
            markersize=5,
            zorder=3,
            label="--ask-for-help at-fixed-interval, task-initial — per-seed mean, n=10 per N",
        )

        for reference_arm in ("no-human", "skill-oracle"):
            run = arms[reference_arm]
            finals = HumanLadderCurves.final_per_seed(run=run, family=None)
            pooled_solved, pooled_total = HumanLadderCurves.pooled_curve(run=run, family=None)[-1]
            ax.axhline(
                sum(finals) / len(finals),
                color=_COLORS[reference_arm],
                linestyle=_LINESTYLES[reference_arm],
                linewidth=2.0,
                label=(
                    f"{_LABELS[reference_arm]} — "
                    f"{HumanLadderCurves.format_count(solved=pooled_solved, total=pooled_total)}"
                ),
            )

        ax.set_xticks(ns)
        ax.set_xlabel(
            "--mean-steps-between-help-requests (N); a request fires on exactly every "
            "Nth policy call (deterministic, no RNG)"
        )
        ax.set_ylabel("final test tasks solved per seed", fontsize=9)
        ax.set_ylim(-_NUM_TEST_TASKS * 0.04, _NUM_TEST_TASKS * 1.06)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    @staticmethod
    def render_rate_sweep_trajectories(
        *, arms: dict, rate_sweep: dict, output: Path, title: str
    ) -> None:
        """One small multiple per N (seven here), showing what the IQR band in
        `render_rate_sweep` cannot: the actual per-seed clustering.

        **This is the honest version of "spread".** A shaded band around a mean asserts a
        single distribution; a reader cannot tell a wide-but-unimodal N from a genuinely
        bimodal one by looking at a band alone. Here all ten of that N's individual seed
        trajectories are drawn (this repo's standard faint/thin per-seed convention), so a
        split into a high cluster and a low cluster -- which is real at some N -- is
        directly visible as two groups of lines rather than inferred from a summary
        statistic.

        Each panel also carries the `no-human` control's own mean training curve (bold,
        orange, the standing "nothing helps" colour) and the `skill-oracle` ceiling (flat,
        dotted, grey) for the same reason the merged fixed-arm panels do: the rate-sweep
        arm's practice is only interpretable next to what zero help and privileged skills
        achieve on the same axis.

        2 rows, one column per two N values (4 columns at this PR's 7-point grid, one panel
        hidden), ordered by N (row-major, so the sampled points read left-to-right
        top-to-bottom in ascending N) -- a single row was tried first and judged too
        compressed to show individual seed lines. Column count is derived from how many N
        points were passed
        rather than hardcoded to 4, so a smaller sweep (as in this module's own tests)
        still lays out cleanly instead of leaving `zip`-mismatched axes; any axis beyond
        the number of N points supplied is hidden rather than left showing empty grid
        lines with no data on them, which would read as a rendering bug."""
        ns = sorted(rate_sweep)
        num_cols = max(1, -(-len(ns) // 2))  # ceil(len(ns) / 2), at least one column
        fig, axes = plt.subplots(
            2,
            num_cols,
            figsize=(4.6 * num_cols, 8.8),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )

        no_human = arms["no-human"]
        no_human_xs = HumanLadderCurves.transitions(run=no_human)
        no_human_pooled = HumanLadderCurves.pooled_curve(run=no_human, family=None)
        no_human_scale = _NUM_TEST_TASKS / no_human_pooled[-1][1]

        oracle = arms["skill-oracle"]
        oracle_solved, oracle_total = HumanLadderCurves.pooled_curve(run=oracle, family=None)[-1]
        oracle_level = oracle_solved * (_NUM_TEST_TASKS / oracle_total)

        axes_flat = list(axes.flat)
        for ax in axes_flat[len(ns) :]:
            ax.set_visible(False)
        for panel_index, (ax, n) in enumerate(zip(axes_flat, ns, strict=False)):
            run = rate_sweep[n]
            # Only the very first seed line of the very first panel is labelled -- one
            # legend entry for "individual seeds", not one per line. Labelling every seed
            # in just the first panel (a bug caught by inspecting the rendered figure) gave
            # ten identical "individual seeds, this N" legend rows instead of one.
            for seed_index, seed in enumerate(sorted(run)):
                entry = HumanLadderCurves.entry(run=run, seed=seed, family=None)
                ax.plot(
                    run[seed]["transitions"],
                    [s for s, _ in entry],
                    color=_RATE_SWEEP_COLOR,
                    alpha=0.55,
                    linewidth=1.0,
                    zorder=2,
                    label=(
                        "individual seeds, this N" if panel_index == 0 and seed_index == 0 else None
                    ),
                )
            ax.plot(
                no_human_xs,
                [solved * no_human_scale for solved, _ in no_human_pooled],
                color=_COLORS["no-human"],
                linestyle=_LINESTYLES["no-human"],
                linewidth=2.0,
                zorder=4,
                label=(_LABELS["no-human"] + " (mean)") if panel_index == 0 else None,
            )
            ax.axhline(
                oracle_level,
                color=_COLORS["skill-oracle"],
                linestyle=_LINESTYLES["skill-oracle"],
                linewidth=1.6,
                zorder=3,
                label=_LABELS["skill-oracle"] if panel_index == 0 else None,
            )
            final_solved, final_total = HumanLadderCurves.pooled_curve(run=run, family=None)[-1]
            ax.set_title(
                f"N={n} — {HumanLadderCurves.format_count(solved=final_solved, total=final_total)}",
                fontsize=10,
            )
            ax.set_ylim(-_NUM_TEST_TASKS * 0.04, _NUM_TEST_TASKS * 1.06)
            ax.grid(alpha=0.2, linewidth=0.5)

        for ax in axes[:, 0]:
            ax.set_ylabel("test tasks solved per seed", fontsize=9)
        for ax in axes[-1, :]:
            ax.set_xlabel("online transitions", fontsize=9)

        # An axes-level legend, not a figure-level one: a figure-level "outside upper
        # center" legend collided with fig.suptitle regardless of call order (both are
        # constrained_layout "outside" artists competing for the same top margin).
        # `loc="best"` rather than a hardcoded corner: #195's module fixed this at
        # "upper right" because its smallest N (N=1) spent nearly every policy call on a
        # rescue and left that corner empty by construction -- an assumption that does not
        # carry to this PR's grid, whose smallest N is 5 and rescues far less densely, so
        # letting matplotlib place it in whichever corner the first panel's own data
        # leaves empty is the choice that does not silently break on the next grid change
        # either.
        axes_flat[0].legend(loc="best", fontsize=7.5, framealpha=0.95)
        fig.suptitle(title, fontsize=10.5)
        fig.savefig(output, dpi=150)
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
            help="e.g. no-human=docs/experiment-logs/2026-08-11-human-ladder-fixed-"
            f"interval-10x/no-human/ees . DIR holds <seed>/stats.json. All three of "
            f"{', '.join(_FIXED_ARMS)} are required.",
        )
        parser.add_argument(
            "--rate-point",
            action="append",
            required=True,
            metavar="N=DIR",
            help="e.g. 5=docs/experiment-logs/2026-08-11-human-ladder-fixed-interval-10x/"
            "rate-sweep/N5/ees . At least two points are required.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Where the five figures are written: overall/TRASH/RECYCLING training "
            "curves (fixed arms plus every rate-sweep point, same panels), the "
            "rate-sweep dose-response figure (mean + IQR band), and the per-N "
            "seed-trajectory small-multiples figure.",
        )
        args = parser.parse_args()

        arm_directories = {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            arm_directories[name] = Path(path)
        rate_directories = {}
        for spec in args.rate_point:
            name, _, path = spec.partition("=")
            rate_directories[int(name)] = Path(path)

        arms = HumanLadderCurves.load_arms(directories=arm_directories)
        rate_sweep = HumanLadderCurves.load_rate_sweep(directories=rate_directories)
        HumanLadderCurves.print_report(arms=arms, rate_sweep=rate_sweep)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        domain = "Tossing Room (split throws, weight drawn at pickup), reset-free practice"
        for family, name, legend_loc in (
            ("TRASH", f"TRASH tasks, x/{_COMPOSITION['TRASH'] * 10}", "lower right"),
            ("RECYCLING", f"RECYCLING tasks, x/{_COMPOSITION['RECYCLING'] * 10}", "upper left"),
            (None, f"all test tasks, x/{_NUM_TEST_TASKS * 10}", "upper left"),
        ):
            HumanLadderCurves.render_family(
                arms=arms,
                rate_sweep=rate_sweep,
                family=family,
                output=args.output_dir
                / f"human-ladder-{'overall' if family is None else family.lower()}.png",
                title=f"{domain}\nfixed arms + at-fixed-interval rate sweep — {name}",
                legend_loc=legend_loc,
            )
        HumanLadderCurves.render_rate_sweep(
            arms=arms,
            rate_sweep=rate_sweep,
            output=args.output_dir / "human-ladder-rate-sweep.png",
            title=(
                f"{domain}\n"
                f"rescue-rate dose-response, --ask-for-help at-fixed-interval "
                f"(overall test tasks, of {_NUM_TEST_TASKS})"
            ),
        )
        HumanLadderCurves.render_rate_sweep_trajectories(
            arms=arms,
            rate_sweep=rate_sweep,
            output=args.output_dir / "human-ladder-rate-sweep-trajectories.png",
            title=(
                f"{domain}\n"
                f"rescue-rate sweep, per-N individual seed trajectories "
                f"(overall test tasks, of {_NUM_TEST_TASKS})"
            ),
        )


if __name__ == "__main__":
    HumanLadderCurves.main()
