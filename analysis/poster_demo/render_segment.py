"""Full chronological practice replay, every recorded tick at native 20 Hz."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import time

import render_base as r
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument("--run",type=Path,required=True)
parser.add_argument("--plan",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--method",required=True)
args=parser.parse_args()
RUN,OUT=args.run,args.out
OUT.mkdir(parents=True, exist_ok=True)
r.FPS = 20
r.SPEED = 1
r.STRIDE = 1
plan = json.loads(args.plan.read_text())
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

# Recover EES outcomes with its exact pinned all-add-effects skill criterion.
# Existing sampler outcomes independently verify every available logged toss outcome.
observed={}
for line in (RUN/"sampler_draws.jsonl").open():
    entry=json.loads(line)
    if entry["skill"]=="MoveToTossLocationAndToss":observed.setdefault(entry["cycle"],[]).append(entry)
parity=0
for c,cycle in enumerate(plan["practice"][:27]):
    toss_index=0
    for action in cycle["actions"]:
        if action["skill"]!="MoveToTossLocationAndToss":continue
        if action["success"] is None:
            value=record(action["block"]+action["ticks"])
            backend.restore(snapshot=backend.plain_to_snapshot(plain={k:list(v) for k,v in value["state"].items()}))
            from hitl_pmp.core.method.types import GroundSkill
            from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
            from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
            env=problem.env
            state=env._observed_state(seed=0,steps_taken=0)
            ground=GroundSkill(skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,objects=(env.robot,env.bin,env.cube,env.barrier,Tossing3DSides.opposite))
            action["success"]=all(atom.predicate.holds(state,atom.objects) for atom in ground.add_effects)
        if args.method=="ees" and toss_index<len(observed.get(c,[])):
            assert bool(action["success"])==observed[c][toss_index]["success"],(c,toss_index)
            parity+=1
        toss_index+=1
prefix=[a for c in plan["practice"][:24] for a in c["actions"] if a["skill"]=="MoveToTossLocationAndToss"]
counts=[sum(a["success"] for a in prefix),len(prefix)]

manifest = dict(fps=20, speed=1, stride=1, cycles=[], actions=[], physics_frames=0,
                reset_display_seconds=1.5, cycle_banner_seconds=2, initial_tosses=counts.copy(), outcome_parity_checks=parity, method=args.method)
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

for cycle_index, cycle in enumerate(plan["practice"][24:27],start=24):
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
                if cycle_index in (24,25,26) and index==len(actions)-1 and done:
                    rendered.save(OUT/f"sample-cycle{cycle_index+1:02d}.png")
        manifest["actions"].append(dict(cycle=cycle_index+1,action=index+1,start_frame=begin_action,end_frame=clip.frames,decision=action["decision"],ticks=action["ticks"]))
    manifest["cycles"].append(dict(cycle=cycle_index+1,start_frame=begin,end_frame=clip.frames,tosses=counts.copy()))
    progress=dict(cycles_done=cycle_index-23,cycles_total=3,frames=clip.frames,elapsed_seconds=time.monotonic()-started)
    (OUT/"progress.json").write_text(json.dumps(progress))
    print(json.dumps(progress),flush=True)
assert len(manifest["cycles"])==3
assert len(manifest["actions"])==sum(len(c["actions"]) for c in plan["practice"][24:27])
manifest["final_tosses"]=counts.copy()
manifest["human_resets"]=sum(a["block"] is None for c in plan["practice"][24:27] for a in c["actions"])
clip.close()
manifest.update(frames=clip.frames,duration_seconds=clip.frames/20,render_seconds=time.monotonic()-started,run=str(RUN),commit="76732331")
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k not in ("cycles","actions")}),flush=True)
