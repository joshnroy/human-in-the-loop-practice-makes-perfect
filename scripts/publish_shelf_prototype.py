"""Publish clearly labeled physical diagnostic evidence, never a readiness claim."""

import hashlib
import json
import shutil
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "scratchpad/shelf-forward-reset"
dest = Path(
    "/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/shelf-forward-reset-prototype"
)
dest.mkdir(parents=True, exist_ok=True)
clips = [
    (
        "native-forward",
        "native-seed1-v1",
        "Native skills, scripted order: 5/8 physical goals",
        "16/16 symbolic skill effects and 8/8 OnFixture atoms; actual task fails because cubes are stacked/tilted. This is not the bilevel benchmark.",
    ),
    (
        "isolated-retrieval",
        "reverse-singlecube-complete-v12",
        "One-cube retrieval completes: 1/1 latest diagnostic",
        "Actual recorded first-placement state; new retrieval skill lifts cube 1 and returns the arm home in 1640 ticks. Saved-state diagnostic, not a continuous eight-cube reset.",
    ),
    (
        "ground-return",
        "return-precision-v17",
        "One-cube ground return completes, yaw still incorrect",
        "2496 ticks; cube is on the ground in its native initial region, gripper open and arm home. Final yaw differs by 1.45 degrees from the declared zero-yaw start. Not a complete distribution reset.",
    ),
]
manifest = {
    "status": "not ready for learning",
    "native_pins": {"kinder": "8f600231", "controllers": "427ad6c"},
    "clips": [],
}
sections = []
for name, folder, title, description in clips:
    src = source / folder
    shutil.copy2(src / "physical-demo.mp4", dest / f"{name}.mp4")
    shutil.copy2(src / "result.json", dest / f"{name}.json")
    digest = hashlib.sha256((dest / f"{name}.mp4").read_bytes()).hexdigest()
    manifest["clips"].append({
        "name": name,
        "source": str(src),
        "video_sha256": digest,
        "description": description,
    })
    sections.append(
        f'<section><h2>{title}</h2><p>{description}</p><video controls preload="metadata" src="{name}.mp4"></video><p><a href="{name}.mp4">Download video</a> · <a href="{name}.json">Raw action/result record</a></p></section>'
    )
for file in (
    "support-contact-audit.json",
    "native-start-audit.json",
    "reverse-collision-audit.json",
):
    shutil.copy2(source / file, dest / file)
(dest / "summary.json").write_text(json.dumps(manifest, indent=2))
html = (
    """<!doctype html><html><head><meta charset="utf-8"><title>Shelf forward/reset diagnostics</title>
<style>body{font:18px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#202633;line-height:1.55}h1,h2{line-height:1.2}video{max-width:100%;width:960px;background:#111}section{border-top:1px solid #ddd;margin:32px 0;padding-top:12px}.status{background:#fff2d8;border-left:5px solid #cc8500;padding:15px}table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #ddd}a{color:#006ba4}</style></head><body>
<p><a href="/">Experiment log</a> · <a href="/shelf-native-assessment/">Native benchmark assessment</a></p>
<h1>Shelf3D-o8: physical forward/reset diagnostics</h1>
<p class="status"><strong>Not ready for Ours versus EES.</strong> No successful continuous eight-cube forward-plus-reset video exists. The clips below show real physical diagnostics and their limitations.</p>
<p>30 September 2026. Native physics, forward controllers, predicates, and goal regions remain unchanged. The new reverse skills are a separate prototype. Every attempt and earlier failure is preserved.</p>
<table><tr><th>Gate</th><th>Evidence</th><th>Remaining issue</th></tr>
<tr><td>Native forward task</td><td>Scripted seed 1: 5/8 actual goal cubes, despite 8/8 symbolic OnFixture</td><td>Stacking and a mismatch between broad symbolic effects and physical goal regions</td></tr>
<tr><td>Isolated retrieval</td><td>Latest variant completes 1/1 recorded-state diagnostic</td><td>Eight-cube packed state has no clear grasp among tested candidates</td></tr>
<tr><td>Ground return</td><td>Native position region, OnGround, HandEmpty and controller termination pass</td><td>Yaw drift means the declared initial distribution is not fully restored</td></tr>
<tr><td>Learning experiment</td><td>Not launched</td><td>Both baseline and full physical reset gates must pass; Shelf learning budgets and costs are not frozen</td></tr></table>
<p>These are scripted/replay diagnostics, not benchmark success rates. Forward controller cap: 400 ticks; reverse diagnostic cap: 3000 ticks. At 10 Hz the videos play at real simulation speed. Neither cap defines the future learning protocol.</p>
"""
    + "".join(sections)
    + """<section><h2>Diagnoses</h2><ul>
<li><a href="native-start-audit.json">Initial native footprint audit</a>: some starts overlap projected cube obstacles despite no physical robot contacts.</li>
<li><a href="support-contact-audit.json">Shelf support contact</a>: initial settled penetration is 0.212 mm; unchanged full native collision checks pass after 0.25 mm upward separation.</li>
<li><a href="reverse-collision-audit.json">Packed shelf grasp collisions</a>: neighboring cubes obstruct the gripper. No collision objects were removed.</li>
</ul><p><a href="summary.json">Artifact provenance and video hashes</a></p></section></body></html>"""
)
(dest / "index.html").write_text(html)
print(dest)
