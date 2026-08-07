"""Post-run analysis of Ball-Ring's `PlaceBallOnTable`: the skill that records
0 successes in 3280 practice executions while EES selects it as a practice target in
10/10 seeds.

**The question this is for.** A skill at 0/3280 has four candidate explanations that
need different fixes, and `evaluations` cannot tell them apart: its add-effects are
unsatisfiable as written; its controller genuinely cannot achieve them; its success
check disagrees with the dynamics (Tossing3D's `NearBin` defect, inverted); or it is only
ever attempted from states where it cannot succeed. The answer here is the second, and
it is **deliberate** -- `place_ball_fall_prob = 1.0` means a bare ball placed on any
table always falls, so `BallOnTable` is never achieved by this skill. That is the
Ball-Ring analogue of Light Switch's `JumpToLight`, and it is a domain-design fact
rather than a defect.

What this script exists for is the part the design fact does *not* cover: what EES does
about it. Three things, each of which needs a different record in `stats.json`:

**Which sampler pool the executions land in.** `PlaceBallOnTable` declares
`param_dim = 2`, so a sampler *is* built and consulted -- this is not the `NO_SAMPLER`
case. `stats.json` stores only three of `SamplerConsultation`'s four pools as fields
(`random`, `informed`, `unparameterized`); `UNINFORMATIVE` is deliberately not stored,
because `SkillPracticeTally` derives it as the remainder. So reading the pool split off
the raw JSON *requires* that subtraction, and getting it wrong is easy: the remainder is
`attempts - random - informed - unparameterized`, and dropping the last term silently
folds `NO_SAMPLER` in. `pool_totals` is the one place that arithmetic lives.

**Target selection versus execution count.** EES executes a chosen candidate's whole
plan prefix, so a skill accrues executions without ever having been the thing EES wanted
to practise -- that is exactly how Tossing3D's `MoveToThrowPose` recorded 175/175
executions while being dropped from every candidate list. `target_totals` reads
`practice_target_outcomes_per_cycle`, which counts *decisions* rather than executions, so
the two can be compared directly.

**Whether EES ever stops.** `skip_perfect` drops a candidate whose measured success rate
is exactly 1.0. There is no symmetric rule for one whose rate is exactly 0.0, and
`score_ground_skill` substitutes the *optimistically extrapolated* competence for the
candidate being scored -- so the worse an impossible skill does, the larger the plan-cost
reduction its improvement appears to promise. `selection_share_per_cycle` plots that
over training: a share that rises rather than decays is the signature.

Never runs a simulation or drives a `Problem`/`Method` (CLAUDE.md's `analysis/`
convention): it reads back `<results-root>/ees/<seed>/stats.json` as written under
`--output-dir`.

**Per-seed spread, not just a mean.** Both panels draw one faint line per seed under the
pooled curve, because with ten seeds a mean can be one seed's behaviour.

Counts, never bare percentages: every printed cell and every axis label is `x/y`.
"""

import argparse
import json
from pathlib import Path

import matplotlib
from pydantic import BaseModel

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


class SkillPools(BaseModel):
    """One skill's practice executions, split by which `SamplerConsultation` pool each
    landed in. `num_uninformative_attempts` is the derived remainder -- see the module
    docstring for why it is not a stored field."""

    model_config = {"frozen": True}

    num_attempts: int = 0
    num_successes: int = 0
    num_random_attempts: int = 0
    num_informed_attempts: int = 0
    num_unparameterized_attempts: int = 0

    @property
    def num_uninformative_attempts(self) -> int:
        return (
            self.num_attempts
            - self.num_random_attempts
            - self.num_informed_attempts
            - self.num_unparameterized_attempts
        )


class SkillTargets(BaseModel):
    """One skill's practice-target *decisions*, as opposed to its executions."""

    model_config = {"frozen": True}

    num_scored: int = 0
    num_selected: int = 0
    num_declined_perfect: int = 0
    num_unreachable: int = 0


class CycleSelection(BaseModel):
    """One practice window: how often this skill was the selected target, against how
    often *any* skill was."""

    model_config = {"frozen": True}

    cycle: int
    num_selected: int
    num_selected_total: int

    @property
    def share(self) -> float:
        return self.num_selected / self.num_selected_total if self.num_selected_total else 0.0


class BallRingImpossibleSkill:
    """A static-method container, never instantiated."""

    @staticmethod
    def load_runs(*, results_root: Path) -> dict[int, dict]:
        """seed -> parsed `stats.json`, from `<results-root>/ees/<seed>/stats.json`."""
        runs: dict[int, dict] = {}
        for path in sorted(results_root.glob("ees/*/stats.json")):
            runs[int(path.parent.name)] = json.loads(path.read_text())
        if not runs:
            raise ValueError(f"no ees/<seed>/stats.json found under {results_root}")
        return runs

    @staticmethod
    def pool_totals(*, runs: dict[int, dict], skill_name: str) -> SkillPools:
        """Executions of `skill_name`, summed over every cycle of every given run."""
        fields = {
            "num_attempts": 0,
            "num_successes": 0,
            "num_random_attempts": 0,
            "num_informed_attempts": 0,
            "num_unparameterized_attempts": 0,
        }
        for stats in runs.values():
            for cycle in stats["practice_outcomes_per_cycle"]:
                entry = cycle.get(skill_name)
                if entry is None:
                    continue
                for key in fields:
                    fields[key] += entry[key]
        return SkillPools(**fields)

    @staticmethod
    def target_totals(*, runs: dict[int, dict], skill_name: str) -> SkillTargets:
        """Practice-target decisions about `skill_name`, summed over cycles and runs.

        Requires `practice_target_outcomes_per_cycle`, the record PR #126 added; a run
        produced before it raises rather than reporting silent zeros, because "never
        selected" and "not recorded" are exactly the two states that must not be
        conflated.
        """
        fields = {
            "num_scored": 0,
            "num_selected": 0,
            "num_declined_perfect": 0,
            "num_unreachable": 0,
        }
        for seed, stats in runs.items():
            if "practice_target_outcomes_per_cycle" not in stats:
                raise ValueError(
                    f"seed {seed}: stats.json has no practice_target_outcomes_per_cycle; "
                    "this run predates the practice-target record and cannot be read for "
                    "selection decisions."
                )
            for cycle in stats["practice_target_outcomes_per_cycle"]:
                entry = cycle.get(skill_name)
                if entry is None:
                    continue
                for key in fields:
                    fields[key] += entry[key]
        return SkillTargets(**fields)

    @staticmethod
    def selection_share_per_cycle(
        *, runs: dict[int, dict], skill_name: str
    ) -> list[CycleSelection]:
        """Per practice window, pooled over runs: this skill's selections against all
        skills' selections."""
        num_cycles = max(len(s["practice_target_outcomes_per_cycle"]) for s in runs.values())
        out: list[CycleSelection] = []
        for cycle_index in range(num_cycles):
            mine = 0
            total = 0
            for stats in runs.values():
                per_cycle = stats["practice_target_outcomes_per_cycle"]
                if cycle_index >= len(per_cycle):
                    continue
                for name, entry in per_cycle[cycle_index].items():
                    total += entry["num_selected"]
                    if name == skill_name:
                        mine += entry["num_selected"]
            out.append(
                CycleSelection(cycle=cycle_index, num_selected=mine, num_selected_total=total)
            )
        return out

    @staticmethod
    def per_seed_selection_share(
        *, runs: dict[int, dict], skill_name: str
    ) -> dict[int, list[CycleSelection]]:
        """`selection_share_per_cycle` for each seed separately, for the spread lines."""
        return {
            seed: BallRingImpossibleSkill.selection_share_per_cycle(
                runs={seed: stats}, skill_name=skill_name
            )
            for seed, stats in runs.items()
        }

    @staticmethod
    def print_report(*, runs: dict[int, dict], skill_name: str) -> None:
        pools = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name=skill_name)
        targets = BallRingImpossibleSkill.target_totals(runs=runs, skill_name=skill_name)
        n = pools.num_attempts
        print(f"=== {skill_name}, pooled over {len(runs)} seeds ===")
        print(f"  executions succeeded         : {pools.num_successes}/{n}")
        print(f"  pool EPSILON_RANDOM          : {pools.num_random_attempts}/{n}")
        print(f"  pool INFORMED                : {pools.num_informed_attempts}/{n}")
        print(f"  pool NO_SAMPLER              : {pools.num_unparameterized_attempts}/{n}")
        print(f"  pool UNINFORMATIVE (derived) : {pools.num_uninformative_attempts}/{n}")
        print(f"  selected as practice target  : {targets.num_selected}")
        print(f"  scored as a candidate        : {targets.num_scored}")
        print(f"  declined as already perfect  : {targets.num_declined_perfect}")
        print(f"  outranked but unreachable    : {targets.num_unreachable}")
        print("  per-seed executions (succeeded/attempted):")
        for seed in sorted(runs):
            one = BallRingImpossibleSkill.pool_totals(
                runs={seed: runs[seed]}, skill_name=skill_name
            )
            tgt = BallRingImpossibleSkill.target_totals(
                runs={seed: runs[seed]}, skill_name=skill_name
            )
            print(
                f"    seed {seed}: {one.num_successes}/{one.num_attempts} executions, "
                f"{tgt.num_selected} selections"
            )

    @staticmethod
    def plot(*, runs: dict[int, dict], skill_name: str, output_path: Path) -> None:
        """Two panels: the selection share over training, and the pool split."""
        pooled = BallRingImpossibleSkill.selection_share_per_cycle(runs=runs, skill_name=skill_name)
        per_seed = BallRingImpossibleSkill.per_seed_selection_share(
            runs=runs, skill_name=skill_name
        )
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

        for _seed, series in sorted(per_seed.items()):
            xs = [c.cycle for c in series if c.num_selected_total > 0]
            ys = [c.share for c in series if c.num_selected_total > 0]
            left.plot(xs, ys, color="tab:red", alpha=0.22, linewidth=1.0)
        xs = [c.cycle for c in pooled if c.num_selected_total > 0]
        ys = [c.share for c in pooled if c.num_selected_total > 0]
        left.plot(xs, ys, color="tab:red", linewidth=2.4, label=f"{skill_name} (pooled)")
        left.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        total_sel = sum(c.num_selected for c in pooled)
        total_all = sum(c.num_selected_total for c in pooled)
        left.set_xlabel("practice window (cycle)")
        left.set_ylabel("selections of this skill / all practice-target selections")
        left.set_ylim(0.0, 1.0)
        left.set_title(
            f"EES keeps choosing an impossible skill\n"
            f"{total_sel}/{total_all} of all practice targets, over {len(runs)} seeds"
        )
        left.legend(loc="lower right", fontsize=8)

        names = [skill_name, "PlaceCupWithoutBallOnTable"]
        labels = ["EPSILON_RANDOM", "INFORMED", "UNINFORMATIVE", "NO_SAMPLER"]
        width = 0.38
        for offset, name in zip((-width / 2, width / 2), names, strict=True):
            p = BallRingImpossibleSkill.pool_totals(runs=runs, skill_name=name)
            values = [
                p.num_random_attempts,
                p.num_informed_attempts,
                p.num_uninformative_attempts,
                p.num_unparameterized_attempts,
            ]
            bars = right.bar(
                [i + offset for i in range(len(labels))], values, width=width, label=name
            )
            for rect, value in zip(bars, values, strict=True):
                right.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 40,
                    f"{value}/{p.num_attempts}",
                    ha="center",
                    fontsize=7,
                )
        right.set_xticks(range(len(labels)))
        right.set_xticklabels(labels, fontsize=8)
        right.set_ylabel("practice executions (count / total for that skill)")
        right.set_title(
            "Which sampler pool the executions land in\n"
            "impossible skill vs the decisive learnable one"
        )
        right.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--skill-name", type=str, default="PlaceBallOnTable")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = BallRingImpossibleSkill.load_runs(results_root=args.results_root)
    BallRingImpossibleSkill.print_report(runs=runs, skill_name=args.skill_name)
    if args.output is not None:
        BallRingImpossibleSkill.plot(runs=runs, skill_name=args.skill_name, output_path=args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
