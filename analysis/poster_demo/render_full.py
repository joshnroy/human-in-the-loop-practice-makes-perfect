"""Full chronological practice replay, every recorded tick at native 20 Hz."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import time

import render_base as r
from PIL import Image, ImageDraw

RUN = Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c/gpoff/results/local_trend-lam3e-6/pomdp/0")
OUT = Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c-poster-full-demo")
OUT.mkdir(parents=True, exist_ok=True)
r.FPS = 20
r.SPEED = 1
r.STRIDE = 1
plan = json.loads(Path(__file__).with_name("plan.json").read_text())
stream = (RUN / "tossing3d_state_log.jsonl").open("rb")
offsets = array("Q")
offset = 0
for line in stream:
    offsets.append(offset)
    offset += len(line)
def record(index):
    stream.seek(offsets[index])
    return json.loads(stream.readline())
header = record(0)
for key in ("kind", "timestamp", "elapsed_seconds"):
    header.pop(key, None)
header = r.StateLogHeader(**header)
problem = r.Tossing3DCli.build_problem(args=argparse.Namespace(**header.model_dump()))
problem.hard_reset()
problem.reset_to_task(task=problem.sample_train_task())
backend = problem.env.backend()
assert backend.render_fps() == 20

def frame(index):
    value = record(index)
    assert value["kind"] == "tick"
    backend.restore(snapshot=backend.plain_to_snapshot(plain={k: list(v) for k,v in value["state"].items()}))
    return backend.render(follow_robot=True)

def label(action):
    return "Human reset" if action["block"] is None else action["label"]

counts = [0,0]
manifest = dict(fps=20, speed=1, stride=1, cycles=[], actions=[], physics_frames=0,
                reset_display_seconds=1.5, cycle_banner_seconds=2, final_tosses=[153,418])
started = time.monotonic()
clip = r.Clip(path=OUT/"all-practice-1x.mp4", kind="practice", title="", subtitle_card="", footer="")
last = backend.render(follow_robot=True)

def compose(scene, actions, current, done):
    image = Image.new("RGB", (1280,720), r.BG)
    image.paste(Image.fromarray(scene).resize((800,600)), (0,60))
    draw = ImageDraw.Draw(image)
    draw.rectangle([812,0,1279,719], fill=r.PANEL)
    draw.text((832,22), "Skill log", font=r.F_SUB, fill=r.TEXT)
    for index, action in enumerate(actions[:current+1]):
        complete = index < current or done
        y = 67 + index*27
        if index == current:
            draw.rectangle([820,y-3,1272,y+23], fill=(55,60,78))
        text = f"{index+1:>2}. {label(action):<12}" + (r.sf(action["success"]) if complete else "")
        color = (r.S_COL if action["success"] else r.F_COL) if complete else r.TEXT
        draw.text((832,y), text, font=r.F_ROWB if index == current else r.F_ROW, fill=color)
    draw.text((832,677), f"Tosses so far: {counts[0]}/{counts[1]}", font=r.F_ROW, fill=r.TEXT)
    return image

for cycle_index, cycle in enumerate(plan["practice"]):
    actions = cycle["actions"]
    assert cycle["cycle"] == cycle_index
    first = next(a for a in actions if a["block"] is not None and a["ticks"])
    last = frame(first["block"]+1)
    begin = clip.frames
    banner = Image.new("RGB", (1280,720), r.BG)
    draw = ImageDraw.Draw(banner)
    draw.text((640,360), f"Practice cycle {cycle_index+1}", font=r.F_CARD, fill=r.TEXT, anchor="mm")
    clip.put(banner, None, 40)
    for index, action in enumerate(actions):
        begin_action = clip.frames
        if action["block"] is None:
            following = next((a for a in actions[index+1:] if a["block"] is not None and a["ticks"]),None)
            if following:
                last=frame(following["block"]+1)
            clip.put(compose(last,actions,index,True),None,30)
        else:
            for tick in range(action["ticks"]):
                last=frame(action["block"]+1+tick)
                done = tick == action["ticks"]-1
                if done and action["skill"] == "MoveToTossLocationAndToss":
                    counts[1]+=1
                    counts[0]+=int(action["success"])
                rendered=compose(last,actions,index,done)
                clip.put(rendered,None)
                manifest["physics_frames"]+=1
                if cycle_index in (0,24,49) and index==len(actions)-1 and done:
                    rendered.save(OUT/f"sample-cycle{cycle_index+1:02d}.png")
        manifest["actions"].append(dict(cycle=cycle_index+1,action=index+1,start_frame=begin_action,end_frame=clip.frames,decision=action["decision"],ticks=action["ticks"]))
    manifest["cycles"].append(dict(cycle=cycle_index+1,start_frame=begin,end_frame=clip.frames,tosses=counts.copy()))
    progress=dict(cycles_done=cycle_index+1,cycles_total=50,frames=clip.frames,elapsed_seconds=time.monotonic()-started)
    (OUT/"progress.json").write_text(json.dumps(progress))
    print(json.dumps(progress),flush=True)
assert counts == [153,418],counts
assert manifest["physics_frames"] == 92404
assert len(manifest["actions"]) == 985
clip.close()
manifest.update(frames=clip.frames,duration_seconds=clip.frames/20,render_seconds=time.monotonic()-started,run=str(RUN),commit="76732331")
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k not in ("cycles","actions")}),flush=True)
