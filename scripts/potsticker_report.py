"""Summarize raw parameter-existence checks, never reinterpret failures as success."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(*, root: Path) -> None:
    relabel_path = root / "open-loop-goal-relabel-checks.json"
    relabeled = {
        (row["source"], row["requested"])
        for row in (json.loads(relabel_path.read_text()) if relabel_path.exists() else [])
        if row["goal"] and row["settled"]
    }
    rows = []
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        if "mode" not in result:
            continue
        p = result["parameters"]
        _, x, y, _ = result["after"]["bodies"]["pitcher"]["quaternion"]
        upright_degrees = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        clean = (
            result["error"] is None
            and result["settled"]
            and result["delivered"] + result["retained"] == result["count"]
            and upright_degrees < 5
        )
        # A count-independent open-loop trajectory supplies a candidate for its
        # observed count, but requires a replay with that count explicitly set.
        candidate = (
            result["delivered"]
            if clean and result["mode"] != "feedback"
            else (result["requested"] if clean and result["success"] else None)
        )
        rows.append(
            dict(
                payload=result["count"],
                requested=result["requested"],
                mode=result["mode"],
                tilt=p["tilt_degrees"],
                amplitude_mm=p["amplitude_m"] * 1000,
                angular_amplitude=p["angular_amplitude_degrees"],
                duration=p["duration_seconds"],
                return_speed_deg_s=result.get("return_speed_deg_s"),
                continuous_return=result.get("continuous_return", False),
                controller_revision=result.get("controller_revision"),
                anticipation_seconds=result.get("anticipation_seconds", 0.0),
                delivered=result["delivered"],
                retained=result["retained"],
                settled=result["settled"],
                upright_degrees=upright_degrees,
                success=result["success"],
                candidate_count=candidate,
                full_replay=result.get("full_replay", False),
                full_cycle_success=result.get("full_cycle_success", False),
                error=result["error"],
                seconds_wall=result["seconds_wall"],
                result_path=str(path),
            )
        )
    if not rows:
        return
    with (root / "trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for count in (3, 10):
        for target in range(1, count + 1):
            relevant = [r for r in rows if r["payload"] == count and r["candidate_count"] == target]
            proofs = [r for r in relevant if r["full_replay"] and r["success"]]
            best = proofs or relevant
            status = "not found yet"
            if relevant:
                status = "candidate found"
            if any((r["result_path"], target) in relabeled for r in relevant):
                status = "goal-checked open-loop witness"
            if any(r["success"] for r in relevant):
                status = "prepared-state pour passed"
            if proofs:
                status = "full replay passed"
            summary.append({
                "payload": count,
                "target": target,
                "status": status,
                "best": min(
                    best,
                    key=lambda r: (
                        not r["full_cycle_success"],
                        r["mode"] != "feedback",
                        r["anticipation_seconds"] > 0,
                        r["return_speed_deg_s"] != 10,
                        r["controller_revision"] != "bounded_ik_consistent_observation_v4",
                        r["duration"],
                    ),
                )
                if best
                else None,
            })
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Frozen potsticker parameter-existence checks",
        "",
        "Seed 0 only. These are parameter-existence checks, not success rates across starts.",
        "A candidate passes exact pan count, retained remainder, settling, "
        "upright vessel (<5 degrees), and no controller error.",
        "Full replays start at the initial scene and additionally attempt pitcher placement.",
        "The bounded search failing to find a recipe does not establish impossibility.",
        "",
        "| Starting pieces | Requested | Evidence | Controller | Tilt | Amplitude | "
        "Feeding limit | Pan / retained | Full cycle |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        b = row["best"]
        if b:
            link = str(Path(b["result_path"]).resolve())
            mode = b["mode"]
            if b["anticipation_seconds"]:
                mode += f" ({b['anticipation_seconds']} s lookahead)"
            if b["return_speed_deg_s"] != 10:
                mode += f" ({b['return_speed_deg_s']} deg/s return)"
            amplitude = (
                f"{b['angular_amplitude']} deg"
                if b["mode"] == "angular"
                else ("unused" if b["mode"] == "tilt_hold" else f"{b['amplitude_mm']} mm")
            )
            lines.append(
                f"| {row['payload']} | {row['target']} | [{row['status']}]({link}) | "
                f"{mode} | {b['tilt']} deg | {amplitude} | "
                f"{b['duration']} s | {b['delivered']} / {b['retained']} | "
                f"{'passed' if b['full_cycle_success'] else 'not validated'} |"
            )
        else:
            lines.append(
                f"| {row['payload']} | {row['target']} | not found yet | — | — | — | — | — | — |"
            )
    lines += [
        "",
        "Angular-wiggle amplitude is recorded separately in trials.csv. "
        "Tilt-and-hold ignores amplitude.",
        "See PROTOCOL.md for model assumptions, controller repairs, "
        "and checkpoint/replay methodology.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    for row in summary:
        print(f"{row['payload']} -> {row['target']}: {row['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summarize(root=parser.parse_args().root)
