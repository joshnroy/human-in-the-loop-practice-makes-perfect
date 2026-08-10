"""Post-run analysis for a scoped human-in-the-loop measurement on Tossing Room: three
fixed arms (a no-human control and two ceilings) plus a rescue-rate dose-response sweep.

**Background.** `--practice-reset-policy never` is the real-robot condition -- a robot
practising in a lab is not teleported to a fresh start every few minutes. Measured on this
domain, it is also badly damaged: Tossing Room's one-way ledge severs rooms 0-2 from the
item pile in room 3, so a practice period that steps left once can never pick anything up
again, and under `never` that damage carries into every later period.

**This supersedes PR #151, which is closed, not merged.** #151 measured an eight-arm ladder
including `on-stuck` (novelty-triggered rescue) and a single `at-random` point at
`--mean-steps-between-help-requests 150`. Neither is here: `on-stuck` is out of scope for
this comparison by deliberate choice, and the raw per-seed data behind #151's numbers was
never committed (`results/` is gitignored) and could not be found intact anywhere on the
machine that built it -- what was found on disk carried a deleted CLI flag
(`--human-intervention-trigger`) from before the help-seeking interface was reshaped onto
`--ask-for-help`/`--human-reset-target`, which is exactly the condition #151's own
"Superseded numbers" precedent says must not be quoted or plotted alongside a current run.
So this is a **fresh measurement**, not a re-plot: every number below comes from a run
against the CURRENT interface, verified by exactly reproducing the one number that could
be checked against a known-good source (`no-human` at 112/300 pooled, matching #151's
control to the task).

**The four components.**

| component | `--method` | `--ask-for-help` | `--human-reset-target` | world | seeds |
| --- | --- | --- | --- | --- | --- |
| `no-human` | `ees` | `never` | -- | one-way | 10 |
| `two-way-ledge` | `ees` | `never` | -- | two-way | 10 |
| `skill-oracle` | `skill-oracle` | -- | -- | one-way | 10 |
| rate sweep | `ees` | `at-random` | `task-initial` | one-way | 10 per N |

The rate sweep varies `--mean-steps-between-help-requests` (N; each policy call asks with
probability 1/N) over a deliberately non-uniform grid from N=1 (asks almost every call) to
N=20 (asks roughly once every twenty calls) -- denser at low N, where the response is
expected to move fastest, sparser toward N=20 where it is expected to have mostly flattened.
This is new territory, not a re-run of #151's `at-random` point: N=150 there is a much
lower-intervention-rate regime than anything in this sweep.

**Which comparisons are clean.** `no-human` and every rate-sweep point share `--method ees`,
the one-way world and all ten seeds, so `PairedTests.sign_flip` applies to each N against the
control. `two-way-ledge` changes the world and `skill-oracle` changes the Method, so neither
is sign-flipped against anything -- each is reported as a ceiling level only, the same
precedent #151's module used.

**Non-learners are drawn flat, not as curves.** `skill-oracle` never practises (no
`--num-cycles` flag exists for it) and has a single evaluation checkpoint, so it is a
horizontal reference line. `no-human` and `two-way-ledge` both learn and get real curves;
`two-way-ledge` gets its own colour rather than the blue/orange assistance-axis pair, because
it is a ceiling on the WORLD (irreversibility removed), not a "does an assistance mechanism
exist" arm -- the same reasoning CLAUDE.md's colour rule already carves out for
`skill-oracle`/`random-skills`.

**Colour.** `no-human` is orange (`#D55E00`): the standing "nothing helps" colour, reused
across every figure in this report. The rate sweep is blue-family (`Blues`, a sequential
colourmap from light N=1 to dark N=20): every rate-sweep arm has an assistance mechanism
available (`--ask-for-help at-random`, always firing at a strictly positive rate here), which
is what the blue/orange rule tracks -- not the specific rate, which the colourmap's lightness
carries instead. `skill-oracle` is grey and dotted (reference line); `two-way-ledge` gets a
third, unreserved colour since it sits on neither axis.

**The three training-curve figures (overall/TRASH/RECYCLING) carry all eleven arms on ONE
panel each** -- the three fixed arms plus all eight rate-sweep points, so a reader sees the
whole ladder's shape over practice, not just its final-checkpoint dose-response. The eight
rate-sweep curves are thin, partly transparent and have no per-seed traces of their own (that
would be 80 more lines); they are drawn first so the three fixed arms' bold, per-seed-backed
curves stay visually on top. See `render_family` for why a colourbar replaces a named legend
entry for the eight of them.

**The separate dose-response figure is not a training curve, and stays alongside the merged
panels rather than being replaced by them.** It answers a different question -- what does
FINAL performance do as the rescue rate varies -- so its x axis is N, not online transitions,
and it draws one point per seed at each of the eight sampled N values, plus the pooled mean,
with the `no-human` and `skill-oracle` levels as reference lines for context. The merged
training-curve panels show shape over practice; this figure is still the one to read for the
exact per-N numbers the training curves intentionally omit from their (colourbar-only) legend.

**The manipulation checks are not optional here.** `num_practice_resets` must be 0
everywhere, or an arm labelled reset-free was quietly reset for free.
`num_human_interventions_recorded` must be exactly 0 for the three arms with no reachable
human (`no-human`, `two-way-ledge`, `skill-oracle`) and strictly positive for every
rate-sweep point -- a zero there at N <= 20 over 1500 policy calls would mean the trigger
never wired rather than a legitimate null.

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
# picked fresh. `two-way-ledge` sits on neither the assistance axis nor the non-learner
# reference set, so it gets its own colour, matching #151's module's precedent.
_COLORS = {
    "no-human": "#D55E00",
    "two-way-ledge": "#CC79A7",
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

# Blue: the rate sweep has an assistance mechanism (--ask-for-help at-random) available and
# firing at every sampled N -- the blue/orange rule tracks whether the mechanism exists, not
# how often it fires.
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

        `expect_no_human` is a hard requirement in BOTH directions here, unlike #151's
        module: a fixed arm recording an intervention is a wiring error, but so is a
        rate-sweep point recording ZERO -- at N <= 20 over 1500 policy calls the expected
        intervention count is in the dozens, so a true zero means the trigger never fired
        rather than a legitimate null (contrast `on-stuck`, which #151's module correctly
        let report zero as a finding -- that trigger is condition-dependent, `at-random`
        is not)."""
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
                "rate over 1500 policy calls that means --ask-for-help never wired, not a "
                "legitimate null."
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
        """The three fixed arms AND all eight rate-sweep points on ONE goal family: the
        training curves, all on the same panel.

        One figure per family (overall / TRASH / RECYCLING), matching the standing
        convention -- a pooled curve would average a large per-family effect against a
        flat one and show a muted version of neither. EMPTY gets no figure: 20/20 in every
        arm, nothing for a curve to show.

        **Eleven lines on one panel needs a different legend strategy than three.** The
        three fixed arms keep named legend entries with their exact pooled count, per the
        standing convention. The eight rate-sweep arms do NOT get named entries -- eight
        more `--mean-steps-between-help-requests=N -- x/300` strings would make the legend
        itself the unreadable part of the figure. They get a sequential colourmap instead
        (light N=1 to dark N=20) with a colourbar, which is the right encoding for an
        ORDERED sweep of one parameter -- Josh's own suggestion, and the natural read for
        "one arm per value of N" the way a discrete named legend is not. All eight are
        blue-family (`Blues`), matching the project's role rule: every rate-sweep arm has
        an assistance mechanism available (`--ask-for-help at-random`, always firing at a
        strictly positive rate here -- see `check_manipulation`), which is what blue
        encodes, not the specific rate. Orange stays reserved for `no-human`, the one arm
        with no mechanism at all.

        **The rate-sweep curves have no per-seed faint traces here, unlike the fixed
        arms.** Eight arms x ten seeds is 80 more lines, which would bury the panel; their
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
            # 0.30 floor keeps even N=1 (the lightest) visible against a white panel.
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
            "--ask-for-help at-random, N (--mean-steps-between-help-requests)", fontsize=8
        )
        colorbar.set_ticks(ns)

        for arm_name in _FIXED_ARMS:
            run = arms[arm_name]
            color = _COLORS[arm_name]
            linestyle = _LINESTYLES[arm_name]
            xs = HumanLadderCurves.transitions(run=run)
            pooled = HumanLadderCurves.pooled_curve(run=run, family=family)
            scale = seed_total / pooled[-1][1]
            label = (
                f"{_LABELS[arm_name]} — "
                f"{HumanLadderCurves.format_count(solved=pooled[-1][0], total=pooled[-1][1])}"
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
        """The dose-response figure: final OVERALL score against the rescue-rate knob N.

        Not a training curve -- there is no online-transitions axis here, since each point
        is a wholly separate arm at a different N, not one arm's progress over time. One
        faint dot per seed at each sampled N (jittered so ties stay countable), a bold mean
        line connecting the pooled-per-seed average at each N, and the `no-human` /
        `skill-oracle` levels as reference lines so the sweep reads against the same
        ceilings the fixed-arm figures use."""
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        ns = sorted(rate_sweep)
        means = []
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
        ax.plot(
            ns,
            means,
            color=_RATE_SWEEP_COLOR,
            linewidth=2.3,
            marker="o",
            markersize=5,
            zorder=3,
            label="--ask-for-help at-random, task-initial — per-seed mean, n=10 per N",
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
            "--mean-steps-between-help-requests (N); each policy call asks with probability 1/N"
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

    # ------------------------------------------------------------------ entry point

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            metavar="NAME=DIR",
            help="e.g. no-human=results/human-ladder-v2/no-human/ees . DIR holds "
            f"<seed>/stats.json. All three of {', '.join(_FIXED_ARMS)} are required.",
        )
        parser.add_argument(
            "--rate-point",
            action="append",
            required=True,
            metavar="N=DIR",
            help="e.g. 5=results/human-ladder-v2/rate-sweep/N5/ees . At least two points "
            "are required.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Where the four figures are written: overall/TRASH/RECYCLING training "
            "curves (fixed arms plus all eight rate-sweep points, same panels), plus the "
            "rate-sweep dose-response figure.",
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
                title=f"{domain}\nfixed arms + at-random rate sweep — {name}",
                legend_loc=legend_loc,
            )
        HumanLadderCurves.render_rate_sweep(
            arms=arms,
            rate_sweep=rate_sweep,
            output=args.output_dir / "human-ladder-rate-sweep.png",
            title=(
                f"{domain}\n"
                f"rescue-rate dose-response, --ask-for-help at-random "
                f"(overall test tasks, of {_NUM_TEST_TASKS})"
            ),
        )


if __name__ == "__main__":
    HumanLadderCurves.main()
