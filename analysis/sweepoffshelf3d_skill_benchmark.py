# ruff: noqa: E501  (the page template is HTML; wrapping it would change the page)
"""The SweepOffShelf3D skill benchmark's page: `report.html` and its figures, read back
from a benchmark output directory (`report.json`, `results-*.jsonl`, `runs/`).

Post-run only: nothing here drives a skill. Counts are x/y throughout.
"""

import html
from collections import Counter, defaultdict
from pathlib import Path

from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.report import REPORT_NAME
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner import ShardWorker
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import (
    FAILURE_LABELS,
    BenchmarkReport,
    CaseResult,
)

# Okabe-Ito, one colour per outcome label (not per method: this figure has none).
LABEL_COLOURS = {
    "success": "#009E73",
    "no_plan": "#D55E00",
    "plan_blocked": "#E69F00",
    "execution_failed": "#CC79A7",
    "effects_missing": "#56B4E9",
    "budget": "#0072B2",
    "harness_error": "#000000",
}
FIGURE = "skills.png"
SWEEP_FIGURE = "sweep_states.png"
PAGE = "report.html"


class SkillBenchmarkPage:
    """A static-method container, never instantiated."""

    @staticmethod
    def render(*, out: Path, title: str = "SweepOffShelf3D skill benchmark") -> Path:
        report = BenchmarkReport.model_validate_json((out / REPORT_NAME).read_text())
        results = [r for r in ShardWorker.results(out=out) if r.label != "invalid"]
        SkillBenchmarkPage.skills_figure(report=report, results=results, path=out / FIGURE)
        SkillBenchmarkPage.sweep_figure(report=report, path=out / SWEEP_FIGURE)
        page = out / PAGE
        page.write_text(SkillBenchmarkPage.page(report=report, results=results, title=title))
        return page

    @staticmethod
    def skills_figure(*, report: BenchmarkReport, results: list[CaseResult], path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        skills = [v.skill for v in report.skills]
        counts = {
            v.skill: Counter(r.label for r in results if r.skill == v.skill) for v in report.skills
        }
        fig, ax = plt.subplots(figsize=(10, 5.4))
        bottoms = [0] * len(skills)
        for label in ("success", *FAILURE_LABELS, "harness_error"):
            heights = [counts[s].get(label, 0) for s in skills]
            if not any(heights):
                continue
            ax.bar(
                range(len(skills)),
                heights,
                bottom=bottoms,
                color=LABEL_COLOURS[label],
                label=f"{label} (n={sum(heights)})",
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
        for i, verdict in enumerate(report.skills):
            mark = "pass" if verdict.passed else "FAIL"
            counts = f"gated {verdict.gated_successes}/{verdict.gated_total}"
            if verdict.reported_total:
                counts += f"\nreported-only {verdict.reported_successes}/{verdict.reported_total}"
            ax.text(
                i,
                bottoms[i] + 0.5,
                f"{counts}\n{mark}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(range(len(skills)))
        ax.set_xticklabels(skills, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("benchmark runs")
        ax.set_ylim(0, max(bottoms + [1]) * 1.25)
        ax.set_title(
            f"SweepOffShelf3D skill benchmark, MuJoCo execution, commit {report.git_commit[:9]}"
        )
        # Below the axes: inside, it covers the tallest bar's x/y label.
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    @staticmethod
    def sweep_figure(*, report: BenchmarkReport, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        per_state = {}
        for verdict in report.skills:
            per_state.update({f"{verdict.skill} {k}": v for k, v in verdict.per_state.items()})
        fig, ax = plt.subplots(figsize=(8, max(2.0, 0.35 * len(per_state) + 1)))
        names = sorted(per_state)
        for i, name in enumerate(names):
            ok, tried = per_state[name]
            ax.barh(i, ok, color="#009E73" if ok else "#D55E00")
            ax.text(ok + 0.1, i, f"{ok}/{tried}", va="center", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("parameter settings that succeeded")
        ax.set_title("SweepOffShelf3D, SweepShelf: succeeding settings per benchmark state")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    @staticmethod
    def page(*, report: BenchmarkReport, results: list[CaseResult], title: str) -> str:
        esc = html.escape
        verdict_rows = []
        for v in report.skills:
            labels = ", ".join(f"{esc(k)} {n}" for k, n in sorted(v.labels.items()))
            pybullet = "n/a" if v.pybullet_planned is None else f"{v.pybullet_planned}/{v.total}"
            verdict_rows.append(
                f"<tr class='{'pass' if v.passed else 'fail'}'><td>{esc(v.skill)}</td>"
                f"<td>{esc(v.target)}</td><td>{v.gated_successes}/{v.gated_total}</td>"
                f"<td>{v.reported_successes}/{v.reported_total}</td>"
                f"<td>{'pass' if v.passed else 'FAIL'}{': ' + esc(v.reason) if v.reason else ''}"
                f"</td><td>{pybullet}</td><td>{labels}</td></tr>"
            )
        sweep_rows = []
        for v in report.skills:
            for state, (ok, tried) in sorted(v.per_state.items()):
                sweep_rows.append(
                    f"<tr class='{'pass' if ok else 'fail'}'><td>{esc(v.skill)}</td>"
                    f"<td>{esc(state)}</td><td>{ok}/{tried}</td></tr>"
                )
        by_family: dict[str, list[CaseResult]] = defaultdict(list)
        for r in results:
            by_family[r.family].append(r)
        family_rows = [
            f"<tr><td>{esc(f)}</td><td>{sum(r.label == 'success' for r in rs)}/{len(rs)}</td></tr>"
            for f, rs in sorted(by_family.items())
        ]
        failure_rows = []
        for r in sorted(results, key=lambda r: (r.skill, r.case_id)):
            if r.label == "success":
                continue
            video = f"<a href='{esc(r.video)}'>video</a>" if r.video else ""
            trace = f"<a href='runs/{esc(r.case_id)}/trace.json'>trace</a>"
            failure_rows.append(
                f"<tr><td>{esc(r.skill)}</td><td>{esc(r.case_id)}</td><td>{esc(r.label)}</td>"
                f"<td>{esc(r.pybullet)}</td><td>{r.controller_steps}</td>"
                f"<td>{esc(r.error or '')}</td><td>{video} {trace}</td></tr>"
            )
        status = "PASSED" if report.passed else "FAILED"
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --pass:#e6f4ea; --fail:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16161a; --fg:#eee; --muted:#aaa; --line:#333; --pass:#16321f; --fail:#3a1a1a; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 24px; font-size:13px; }}
td,th {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }}
tr.pass td {{ background:var(--pass); }} tr.fail td {{ background:var(--fail); }}
img {{ max-width:100%; }} .muted {{ color:var(--muted); }} .wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{esc(title)}: {status}</h1>
<p class="muted">commit {esc(report.git_commit)}{" (dirty tree)" if report.git_dirty else ""};
kindergarden {esc(report.kindergarden_commit[:12])}, kinder-models {esc(report.kinder_models_commit[:12])};
state set {esc(Path(report.state_set).name)} (sha256 {esc(report.state_set_sha256[:12])});
{report.num_cases} runs, {report.num_invalid} invalid; {report.elapsed_seconds / 60:.0f} min.</p>
<p>Every run executes the skill in MuJoCo with KINDER's planners and controllers. A failure's label says which
simulator it belongs to: <b>no_plan</b> (no candidate planned in KINDER's PyBullet scene, or the scripted routine
found no route), <b>plan_blocked</b> (planned in PyBullet, refused by the MuJoCo re-check: the planning/execution gap),
<b>execution_failed</b> (a plan ran in MuJoCo and failed), <b>effects_missing</b> (no error, a promised effect absent).
"PyBullet planned" counts runs whose every reached stage had a plan in PyBullet.</p>
<img src="{FIGURE}" alt="Runs per skill, stacked by outcome label">
<div class="wrap"><table><tr><th>skill</th><th>target</th><th>gated success</th><th>reported-only success</th><th>verdict</th><th>PyBullet planned</th><th>labels</th></tr>
{"".join(verdict_rows)}</table></div>
<h2>SweepShelf: succeeding settings per state</h2>
<img src="{SWEEP_FIGURE}" alt="Succeeding sweep settings per state">
<div class="wrap"><table><tr><th>skill</th><th>state</th><th>settings succeeded</th></tr>{"".join(sweep_rows)}</table></div>
<h2>By state family</h2>
<div class="wrap"><table><tr><th>family</th><th>success</th></tr>{"".join(family_rows)}</table></div>
<h2>Every failure</h2>
<div class="wrap"><table><tr><th>skill</th><th>case</th><th>label</th><th>PyBullet</th><th>control steps</th><th>error</th><th></th></tr>
{"".join(failure_rows)}</table></div>
</body></html>
"""


if __name__ == "__main__":
    import sys

    print(SkillBenchmarkPage.render(out=Path(sys.argv[1])))
