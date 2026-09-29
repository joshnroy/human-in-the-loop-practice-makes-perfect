"""(EXP-17/18 copy: the reset label takes --reset-note, since EES arms pay 5, 0.05 or 0.)
Render per-cycle practice clips and per-sweep eval clips from a retained run's logs.

Consumes plan.json (built by build_timeline.py from the decision log, state log,
episode_traces and progress), restores every logged physics tick into a fresh scene
built from the state log's own header, and burns in phase/cycle/action labels.
Every tick becomes one frame; the speed-up comes only from the playback rate.

    python render_clips.py RUN_DIR plan.json OUT_DIR [--only practice-1,eval-0]
"""

import argparse
import subprocess
import os
from array import array
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader

SIM_HZ = 20
SPEED = 4
STRIDE = 2  # draw every other simulator tick
FPS = SIM_HZ * SPEED // STRIDE
W, H = 1280, 720
HEAD = 84
SUB = 44
SCENE_W, SCENE_H = 800, 592
PANEL_X = 812

COL = {"practice": (0, 114, 178), "eval": (0, 150, 110), "baseline": (200, 130, 0)}
BG, PANEL, TEXT, MUTED = (18, 20, 26), (28, 31, 40), (232, 234, 240), (154, 160, 176)
S_COL, F_COL, R_COL = (81, 226, 153), (240, 100, 90), (0, 158, 115)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


F_TITLE, F_SUB, F_ROW, F_ROWB, F_SMALL, F_CARD, F_CARD2 = (
    font(40, True), font(24, True), font(17), font(17, True), font(14), font(64, True), font(30))


def mono(size, bold=False):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


F_ROW, F_ROWB = mono(16), mono(16, True)


def sf(success):
    return "" if success is None else ("S" if success else "F")


class Clip:
    def __init__(self, *, path, kind, title, subtitle_card, footer):
        self.path, self.kind, self.title = path, kind, title
        self.subtitle_card, self.footer = subtitle_card, footer
        self.writer = imageio.get_writer(str(path), fps=FPS, codec="h264_nvenc", pixelformat="yuv420p",
                                         quality=None, macro_block_size=16,
                                         output_params=["-preset", "p4", "-cq", "23"])
        self.frames = 0

    def put(self, img, full, n=1):
        arr = np.asarray(img)
        for _ in range(n):
            self.writer.append_data(arr)
        self.frames += n

    def close(self):
        self.writer.close()
        tmp = self.path.with_suffix(".ready.mp4")
        subprocess.run([os.environ["IMAGEIO_FFMPEG_EXE"], "-loglevel", "error", "-y", "-i", str(self.path), "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(self.path)
        self.path.with_suffix(".done").write_text(str(self.frames))


def compose(*, clip, scene, action_line, outcome, rows, current_row, footer_extra):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, HEAD], fill=COL[clip.kind])
    d.text((20, HEAD // 2), clip.title, font=F_TITLE, fill=(255, 255, 255), anchor="lm")
    d.text((W - 20, HEAD // 2), f"{SPEED}x speed", font=F_SUB, fill=(255, 255, 255), anchor="rm")
    d.rectangle([0, HEAD, W, HEAD + SUB], fill=PANEL)
    d.text((20, HEAD + SUB // 2), action_line, font=F_SUB, fill=TEXT, anchor="lm")
    if outcome is not None:
        d.text((W - 20, HEAD + SUB // 2), outcome[0], font=F_SUB, fill=outcome[1], anchor="rm")
    img.paste(Image.fromarray(scene).resize((SCENE_W, SCENE_H)), (0, HEAD + SUB))
    d.rectangle([PANEL_X - 4, HEAD + SUB, W, H], fill=PANEL)
    y = HEAD + SUB + 10
    row_h = min(27, (H - y - 110) // max(1, len(rows)))
    for i, (text, colour) in enumerate(rows):
        if i == current_row:
            d.rectangle([PANEL_X, y - 2, W - 6, y + row_h - 4], fill=(55, 60, 78))
        d.text((PANEL_X + 8, y), text, font=F_ROWB if i == current_row else F_ROW, fill=colour)
        y += row_h
    notes = [footer_extra.strip(" ·")] + clip.footer.split("\n")
    for j, note in enumerate(reversed(notes)):
        d.text((PANEL_X + 8, H - 8 - 20 * j), note, font=F_SMALL, fill=MUTED, anchor="ld")
    return img


def card(*, clip, scene):
    img = Image.new("RGB", (W, H), BG)
    img.paste(Image.fromarray(scene).resize((960, 720)), (160, 0))
    img = Image.blend(img, Image.new("RGB", (W, H), BG), 0.72)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H // 2 - 130, W, H // 2 + 130], fill=COL[clip.kind])
    d.text((W // 2, H // 2 - 40), clip.title, font=F_CARD, fill=(255, 255, 255), anchor="mm")
    d.text((W // 2, H // 2 + 50), clip.subtitle_card, font=F_CARD2, fill=(255, 255, 255), anchor="mm")
    d.text((W // 2, H - 30), f"playback {SPEED}x real time; every {STRIDE}nd simulator tick shown",
           font=F_SMALL, fill=TEXT, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--footer", required=True)
    ap.add_argument("--reset-note", default="cost 5")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    plan = json.loads(a.plan.read_text())
    state_file = a.run.joinpath("tossing3d_state_log.jsonl").open("rb")
    offsets = array("Q")
    offset = 0
    for line in state_file:
        offsets.append(offset)
        offset += len(line)

    def read_line(index):
        state_file.seek(offsets[index])
        return state_file.readline()

    hv = json.loads(read_line(0))
    for k in ("kind", "timestamp", "elapsed_seconds"):
        hv.pop(k, None)
    header = StateLogHeader(**hv)
    problem = Tossing3DCli.build_problem(args=argparse.Namespace(**header.model_dump()))
    env, backend = problem.env, problem.env.backend()
    problem.hard_reset()
    problem.reset_to_task(task=problem.sample_train_task())
    assert backend.render_fps() == SIM_HZ, backend.render_fps()

    def tick_frame(line_no):
        r = json.loads(read_line(line_no))
        assert r["kind"] == "tick"
        backend.restore(snapshot=backend.plain_to_snapshot(plain={k: list(v) for k, v in r["state"].items()}))
        return backend.render(follow_robot=True)

    a.out.mkdir(parents=True, exist_ok=True)
    full_name = "full-loop.mp4" if not only else "full-loop-partial.mp4"
    full = type("NoWriter", (), {"append_data": lambda self, a: None, "close": lambda self: None})()
    meta = {"speed": SPEED, "fps": FPS, "clips": []}
    hold = lambda s: max(1, round(s * FPS))
    last = backend.render(follow_robot=True)
    base_footer = a.footer.replace("\\n", "\n")

    order = []
    NCY = len(plan["practice"])
    for c in range(NCY + 1):
        order.append(("eval", c))
        if c < NCY:
            order.append(("practice", c))
    for kind, idx in order:
        key = f"{kind}-{idx}"
        if only and key not in only:
            continue
        filename = f"practice-cycle{idx + 1:02d}.mp4" if kind == "practice" else f"eval-sweep{idx:02d}.mp4"
        if (a.out / filename).with_suffix(".done").exists():
            continue
        if kind == "practice":
            p = plan["practice"][idx]
            acts = p["actions"]
            ntoss = sum(x["skill"] == "MoveToTossLocationAndToss" for x in acts)
            nts = sum(x["skill"] == "MoveToTossLocationAndToss" and x["success"] for x in acts)
            clip = Clip(path=a.out / f"practice-cycle{idx + 1:02d}.mp4", kind="practice",
                        title=f"PRACTICE — Cycle {idx + 1} of 50",
                        subtitle_card=f"{len(acts)} actions, ended by {p['end']}",
                        footer=base_footer)
            rows_done = []
            first = next((x for x in acts if x["block"] is not None and x["ticks"] > 0), None)
            if first is not None:
                last = tick_frame(first["block"] + 1)
            clip.put(card(clip=clip, scene=last), full, hold(2.0))
            tosses_so_far = [0, 0]

            def rows(cur):
                out = [(f"{i + 1:>2}. {x['label']:<8} {sf(x['success']) if i < cur else ''}",
                        (S_COL if x["success"] else F_COL) if i < cur else TEXT)
                       for i, x in enumerate(acts[: cur + 1])]
                return out

            for i, x in enumerate(acts):
                line = f"Action {i + 1} of {len(acts)}:  {x['label']}"
                extra = lambda: f"  ·  tosses so far {tosses_so_far[0]}/{tosses_so_far[1]}"
                if x["block"] is None:
                    # a human reset writes no ticks; show the state the next action starts from
                    nxt = next((y for y in acts[i + 1:] if y["block"] is not None and y["ticks"] > 0), None)
                    if nxt is not None:
                        last = tick_frame(nxt["block"] + 1)
                    line += f" (human, {a.reset_note})"
                    clip.put(compose(clip=clip, scene=last, action_line=line, outcome=("S", S_COL),
                                     rows=rows(i + 1)[: i + 1], current_row=i, footer_extra=extra()),
                             full, hold(1.5))
                    continue
                for t in range(0, x["ticks"], STRIDE):
                    last = tick_frame(x["block"] + 1 + t)
                    clip.put(compose(clip=clip, scene=last, action_line=line, outcome=None,
                                     rows=rows(i), current_row=i, footer_extra=extra()), full)
                if x["skill"] == "MoveToTossLocationAndToss":
                    tosses_so_far[1] += 1
                    tosses_so_far[0] += int(x["success"])
                if x["ticks"] == 0:
                    line += " (not executed)"
                clip.put(compose(clip=clip, scene=last, action_line=line,
                                 outcome=(f"→ {sf(x['success'])}", S_COL if x["success"] else F_COL),
                                 rows=rows(i + 1)[: i + 1], current_row=i, footer_extra=extra()),
                         full, hold(0.75))
            end = Image.new("RGB", (W, H), BG)
            dd = ImageDraw.Draw(end)
            dd.rectangle([0, H // 2 - 110, W, H // 2 + 110], fill=COL["practice"])
            dd.text((W // 2, H // 2 - 35), f"END OF PRACTICE CYCLE {idx + 1}", font=F_CARD, fill=(255, 255, 255), anchor="mm")
            dd.text((W // 2, H // 2 + 45), f"tosses {nts}/{ntoss} succeeded  ·  ended by {p['end']}", font=F_CARD2, fill=(255, 255, 255), anchor="mm")
            if p["end"].startswith("CRASH"):
                dd.text((W // 2, H // 2 + 85), "next toss: bin at (-0.008, -1.276), standoff 2.54 m - all 4 directions blocked", font=F_CARD2, fill=(255, 255, 255), anchor="mm")
            clip.put(end, full, hold(4.0 if p["end"].startswith("CRASH") else 1.5))
            summary = {"tosses": f"{nts}/{ntoss}", "picks": f"{sum(x['skill']=='PickCube' and x['success'] for x in acts)}/{sum(x['skill']=='PickCube' for x in acts)}",
                       "resets": sum(x["block"] is None for x in acts), "actions": len(acts), "end": p["end"]}
        else:
            e = plan["eval"][idx]
            kindc = "baseline" if idx == 0 else "eval"
            title = "BASELINE EVAL — before practice" if idx == 0 else f"EVAL — after cycle {idx}"
            clip = Clip(path=a.out / f"eval-sweep{idx:02d}.mp4", kind=kindc, title=title,
                        subtitle_card=f"sweep {idx}: 10 fixed test tasks, {e['solved']}/10 solved",
                        footer=base_footer + "\neval Pick S/F = grasp contact at end of skill")
            tasks = e["tasks"]
            first = next(y for t in tasks for y in t["actions"] if y["ticks"] > 0)
            last = tick_frame(first["block"] + 1)
            clip.put(card(clip=clip, scene=last), full, hold(2.0))
            solved_so_far = 0

            def trow(t, upto, done):
                parts = " ".join(
                    ("noop" if y["skill"] == "no-op" else f"{y['label']}({sf(y['success'])})")
                    for y in t["actions"][:upto])
                res = ("SOLVED" if t["solved"] else "unsolved") if done else "…"
                colour = (S_COL if t["solved"] else F_COL) if done else TEXT
                return (f"Task {t['task'] + 1:>2}: {parts}  {res}", colour)

            for ti, t in enumerate(tasks):
                base_title = title
                clip.title = f"{base_title} (task {t['task'] + 1} of 10)"
                done_rows = [trow(u, len(u["actions"]), True) for u in tasks[:ti]]
                for ai, y in enumerate(t["actions"]):
                    label = "No-op (no plan found)" if y["skill"] == "no-op" else y["label"] + ("" if y["ticks"] else " (not executed)")
                    line = f"Sweep {idx} · task {t['task'] + 1}, step {ai + 1}:  {label}"
                    extra = f"  ·  solved so far {solved_so_far}/{ti}"
                    cur = done_rows + [trow(t, ai, False)]
                    for k in range(0, y["ticks"], STRIDE):
                        last = tick_frame(y["block"] + 1 + k)
                        clip.put(compose(clip=clip, scene=last, action_line=line, outcome=None,
                                         rows=cur, current_row=ti, footer_extra=extra), full)
                    oc = None if y["success"] is None else (f"→ {sf(y['success'])}", S_COL if y["success"] else F_COL)
                    clip.put(compose(clip=clip, scene=last, action_line=line, outcome=oc,
                                     rows=done_rows + [trow(t, ai + 1, False)], current_row=ti,
                                     footer_extra=extra), full, hold(0.75 if y["ticks"] else 0.4))
                solved_so_far += int(t["solved"])
                res = ("TASK SOLVED", S_COL) if t["solved"] else ("TASK NOT SOLVED", F_COL)
                clip.put(compose(clip=clip, scene=last, action_line=f"Task {t['task'] + 1} of 10 finished",
                                 outcome=res, rows=done_rows + [trow(t, len(t["actions"]), True)],
                                 current_row=ti, footer_extra=f"  ·  solved so far {solved_so_far}/{ti + 1}"),
                         full, hold(1.0))
                clip.title = base_title
            assert solved_so_far == e["solved"]
            end = Image.new("RGB", (W, H), BG)
            dd = ImageDraw.Draw(end)
            dd.rectangle([0, H // 2 - 110, W, H // 2 + 110], fill=COL[kindc])
            dd.text((W // 2, H // 2 - 35), title, font=font(50, True), fill=(255, 255, 255), anchor="mm")
            dd.text((W // 2, H // 2 + 45), f"{e['solved']}/10 test tasks solved", font=F_CARD2, fill=(255, 255, 255), anchor="mm")
            clip.put(end, full, hold(1.5))
            summary = {"solved": f"{e['solved']}/10"}
        clip.close()
        meta["clips"].append({"key": key, "file": clip.path.name, "frames": clip.frames,
                              "seconds": round(clip.frames / FPS, 1), **summary})
        print(key, clip.frames, flush=True)
    full.close()
    (a.out / ("clips.json" if not only else "clips-partial.json")).write_text(json.dumps(meta, indent=1))
    env.close()
    state_file.close()


if __name__ == "__main__":
    main()
