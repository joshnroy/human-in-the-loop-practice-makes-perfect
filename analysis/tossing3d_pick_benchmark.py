# ruff: noqa: E501  (the page template is HTML; wrapping it would change the page)
"""The Tossing3D pick benchmark's page: `report.html` and `picks.png`, read back from a
benchmark output directory. Post-run only. Counts are x/y throughout."""

import html
from collections import Counter, defaultdict
from pathlib import Path

from hitl_pmp.environments.tossing3d.pick_benchmark.report import REPORT_NAME
from hitl_pmp.environments.tossing3d.pick_benchmark.runner import PickShardWorker
from hitl_pmp.environments.tossing3d.pick_benchmark.types import PickReport, PickResult

LABEL_COLOURS = {
    "success": "#009E73",
    "no_plan": "#D55E00",
    "execution_failed": "#CC79A7",
    "budget": "#0072B2",
    "harness_error": "#000000",
}
FIGURE = "picks.png"
PAGE = "report.html"


class PickBenchmarkPage:
    """A static-method container, never instantiated."""

    @staticmethod
    def render(*, out: Path) -> Path:
        report = PickReport.model_validate_json((out / REPORT_NAME).read_text())
        results = PickShardWorker.results(out=out)
        PickBenchmarkPage.figure(results=results, path=out / FIGURE, commit=report.git_commit)
        page = out / PAGE
        page.write_text(PickBenchmarkPage.page(report=report, results=results))
        return page

    @staticmethod
    def families(*, results: list[PickResult]) -> list[str]:
        return sorted({r.family for r in results})

    @staticmethod
    def figure(*, results: list[PickResult], path: Path, commit: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            (f, s)
            for f in PickBenchmarkPage.families(results=results)
            for s in ("PickCube", "PickCubeFromBin")
        ]
        rows = [(f, s) for f, s in rows if any(r.family == f and r.skill == s for r in results)]
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.8))
        for i, (family, skill) in enumerate(rows):
            runs = [r for r in results if r.family == family and r.skill == skill]
            left = 0
            for label, colour in LABEL_COLOURS.items():
                n = sum(r.label == label for r in runs)
                if n:
                    ax.barh(i, n, left=left, color=colour)
                    left += n
            ok = sum(r.label == "success" for r in runs)
            outside = sum(not r.in_target for r in runs)
            note = f"{ok}/{len(runs)}" + (f" ({outside} outside operator)" if outside else "")
            ax.text(left + 0.5, i, note, va="center", fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{s} · {f}" for f, s in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("pick runs")
        ax.set_title(f"Tossing3D pick benchmark, MuJoCo execution, commit {commit[:9]}")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLOURS.values()]
        totals = Counter(r.label for r in results)
        # Lower right: every bar but the longest leaves that corner empty.
        ax.legend(
            handles,
            [f"{label} (n={totals.get(label, 0)})" for label in LABEL_COLOURS],
            fontsize=7,
            loc="lower right",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def page(*, report: PickReport, results: list[PickResult]) -> str:
        esc = html.escape
        rows = []
        for v in report.skills:
            labels = ", ".join(f"{esc(k)} {n}" for k, n in sorted(v.labels.items()))
            rows.append(
                f"<tr class='{'pass' if v.passed else 'fail'}'><td>{esc(v.skill)}</td><td>{v.successes}/{v.total}</td>"
                f"<td>{'pass' if v.passed else 'FAIL'}</td><td>{v.pybullet_planned}/{v.total}</td>"
                f"<td>{v.dry_run_plannable}/{v.total}</td><td>{v.dry_run_agrees}/{v.total}</td>"
                f"<td>{v.outside_operator}</td><td>{labels}</td></tr>"
            )
        grid: dict[tuple[str, str], list[PickResult]] = defaultdict(list)
        for r in results:
            grid[(r.family, r.skill)].append(r)
        family_rows = []
        for (family, skill), runs in sorted(grid.items()):
            counted = [r for r in runs if r.in_target]
            ok = sum(r.label == "success" for r in counted)
            dry = sum(bool(r.dry_run_plannable) for r in runs)
            family_rows.append(
                f"<tr><td>{esc(family)}</td><td>{esc(skill)}</td><td>{ok}/{len(counted)}</td>"
                f"<td>{dry}/{len(runs)}</td><td>{len(runs) - len(counted)}</td></tr>"
            )
        failures = []
        for r in sorted(results, key=lambda r: (r.family, r.case_id)):
            if r.label == "success":
                continue
            video = f"<a href='{esc(r.video)}'>video</a>" if r.video else ""
            failures.append(
                f"<tr><td>{esc(r.case_id)}</td><td>{esc(r.family)}</td><td>{esc(r.label)}</td>"
                f"<td>{'yes' if r.dry_run_plannable else 'no'}</td><td>{'yes' if r.in_target else 'no'}</td>"
                f"<td>{r.controller_steps}</td><td>{esc(r.error or '')}</td>"
                f"<td>{video} <a href='runs/{esc(r.case_id)}/run.json'>run</a></td></tr>"
            )
        status = "PASSED" if report.passed else "FAILED"
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tossing3D pick benchmark</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --pass:#e6f4ea; --fail:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16161a; --fg:#eee; --muted:#aaa; --line:#333; --pass:#16321f; --fail:#3a1a1a; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 24px; font-size:13px; }}
td,th {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }}
tr.pass td {{ background:var(--pass); }} tr.fail td {{ background:var(--fail); }}
img {{ max-width:100%; }} .muted {{ color:var(--muted); }} .wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>Tossing3D pick benchmark: {status}</h1>
<p class="muted">pick implementation: {esc(report.pick_implementation)}; commit {esc(report.git_commit)}{" (dirty tree)" if report.git_dirty else ""};
kindergarden {esc(report.kindergarden_commit[:12])}, kinder-models {esc(report.kinder_models_commit[:12])};
state set sha256 {esc(report.state_set_sha256[:12])}; {report.num_cases} runs; {report.elapsed_seconds / 60:.0f} min.</p>
<p>Each run restores a fixed KINDER state, asks the PickPlannable dry run (the controller's own planning, in KINDER's
PyBullet scene) for its verdict, then dispatches the pick in MuJoCo. <b>no_plan</b>: the dispatched controller found no plan;
<b>execution_failed</b>: it planned (in PyBullet) and the cube is not held after MuJoCo ran it, the planning/execution gap.
"Counted" runs are those the operator admits apart from PickPlannable (hand empty, bin upright, cube on the robot's side);
the rest are run and shown, never counted.</p>
<img src="{FIGURE}" alt="Pick runs per family, stacked by outcome">
<div class="wrap"><table><tr><th>skill</th><th>success (counted)</th><th>verdict</th><th>PyBullet planned</th><th>dry run plannable</th><th>dry run agrees with dispatch</th><th>outside operator</th><th>labels</th></tr>
{"".join(rows)}</table></div>
<h2>By family</h2>
<div class="wrap"><table><tr><th>family</th><th>skill</th><th>success (counted)</th><th>dry run plannable</th><th>outside operator</th></tr>{"".join(family_rows)}</table></div>
<h2>Every failure</h2>
<div class="wrap"><table><tr><th>case</th><th>family</th><th>label</th><th>dry run plannable</th><th>counted</th><th>control steps</th><th>error</th><th></th></tr>
{"".join(failures)}</table></div>
</body></html>
"""


if __name__ == "__main__":
    import sys

    print(PickBenchmarkPage.render(out=Path(sys.argv[1])))
