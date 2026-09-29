"""Validate and publish the complete poster replay without exposing partial files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

HERE=Path(__file__).resolve().parent
OUT=Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c-poster-full-demo")
SITE=Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c/poster-drafts")
GEN=Path("/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/.codex-worktrees/exp22c-checkpoint/analysis/exp22c/poster")
while not (OUT/"manifest.json").exists():
    state=subprocess.check_output(["systemctl","--user","show","exp22c-full-poster-demo.service","-p","ActiveState","--value"],text=True).strip()
    if state != "active":
        raise RuntimeError("Renderer stopped without a complete manifest: "+state)
    time.sleep(20)
manifest=json.loads((OUT/"manifest.json").read_text())
assert manifest["frames"]==98634
assert manifest["physics_frames"]==92404
assert len(manifest["cycles"])==50
assert len(manifest["actions"])==985
assert manifest["cycles"][-1]["tosses"]==[153,418]
assert manifest["duration_seconds"]==4931.7
video=OUT/"all-practice-1x.mp4"
probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames,r_frame_rate,duration","-of","json",str(video)],text=True))["streams"][0]
assert (probe["width"],probe["height"],probe["r_frame_rate"],int(probe["nb_frames"]))==(1280,720,"20/1",98634),probe
assert abs(float(probe["duration"])-4931.7)<.001
for cycle in (1,25,50):
    assert (OUT/f"sample-cycle{cycle:02d}.png").exists()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
manifest["video_sha256"]=sha(video)
manifest["renderer_sha256"]=sha(HERE/"render_full.py")
manifest["source_plan_sha256"]=sha(HERE/"plan.json")
manifest["probe"]=probe
archive=SITE/"historical-cycle1-4x"
archive.mkdir(exist_ok=True)
for name in ("demo.mp4","timeline.html","timeline-provenance.json"):
    if (SITE/name).exists() and not (archive/name).exists():
        shutil.copy2(SITE/name,archive/name)
for name in ("all-practice-1x.mp4","demo.mp4"):
    tmp=SITE/(name+".tmp")
    if tmp.exists():tmp.unlink()
    os.link(video,tmp)
    os.replace(tmp,SITE/name)
(SITE/"demo-manifest.json").write_text(json.dumps(manifest,indent=2))
for cycle in (1,25,50):
    shutil.copy2(OUT/f"sample-cycle{cycle:02d}.png",SITE/f"demo-sample-cycle{cycle:02d}.png")
shutil.copy2(HERE/"timeline_html.py",GEN/"timeline_html.py")
subprocess.run(["python",str(GEN/"timeline_html.py")],check=True)
p=GEN/"poster_index.py"
s=p.read_text().replace("Cycle-1 illustrative demo","All50/50 practice cycles at1×")
s=s.replace("The existing demo below covers only cycle1.","The complete practice-only demo below covers50/50cycles at1×.")
s=s.replace("<h2>Existing demo</h2>","<h2>Full practice demo ·50/50cycles</h2>")
s=s.replace("42.75s,4× replay, WS Bλ3e-6 GP-off seed0 cycle1:1/20 reset actions,10/10 picks,0/9 tosses. Early practice illustration; not a successful-toss highlight.","82min11.7s,1× replay of all50/50practice cycles, WS ModelBλ3e-6 GP-off seed0, commit76732331. All92404/92404recorded physics frames at20fps;153/418tosses succeeded. Includes50two-second cycle-start banners and141human-reset indications of1.5s each; those indications are presentation time, not recorded physics. No evaluation footage. Skill log and cumulative toss counter retained; method/configuration metadata appears only here. <a href=\"all-practice-1x.mp4\">Separate full video</a> · <a href=\"demo-manifest.json\">Frame and action provenance</a> · <a href=\"historical-cycle1-4x/demo.mp4\">Archived original cycle1 demo</a>.")
p.write_text(s)
subprocess.run(["python",str(p)],check=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
url="http://100.65.117.46:8766/exp22c/poster-drafts/demo.mp4"
with opener.open(urllib.request.Request(url,headers={"Range":"bytes=0-1023"}),timeout=20) as response:
    assert response.status==206
    assert len(response.read())==1024
links=(SITE/"timeline.html").read_text().count('href="demo.mp4#t=')
assert links==985,links
report=dict(ready=True,url="http://agni:8766/exp22c/poster-drafts/demo.mp4",cycles="50/50",actions="985/985",physics_frames="92404/92404",frames=98634,duration_seconds=4931.7,http_range_status=206,video_sha256=manifest["video_sha256"])
(OUT/"published.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
