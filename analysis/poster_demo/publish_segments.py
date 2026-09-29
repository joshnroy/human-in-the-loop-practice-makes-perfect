import json,os,shutil,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c-poster-representative")
SITE=Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c/poster-drafts/representative")
SITE.mkdir(exist_ok=True)
for method in ("ours","ees"):
 out=ROOT/method
 while not (out/"manifest.json").exists():time.sleep(5)
 m=json.loads((out/"manifest.json").read_text())
 assert [c["cycle"] for c in m["cycles"]]==[25,26,27]
 assert m["human_resets"]==({"ours":11,"ees":18}[method])
 expected=m["physics_frames"]+120+30*m["human_resets"]
 assert m["frames"]==expected
 probe=json.loads(subprocess.check_output(["/home/josh/miniconda3/envs/ffnv/bin/ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames,r_frame_rate,duration","-of","json",str(out/"all-practice-1x.mp4")],text=True))["streams"][0]
 assert int(probe["nb_frames"])==expected and probe["r_frame_rate"]=="20/1"
 name=f"{method}-cycles25-27-1x.mp4"
 tmp=SITE/(name+".tmp")
 if tmp.exists():tmp.unlink()
 os.link(out/"all-practice-1x.mp4",tmp);os.replace(tmp,SITE/name)
 m["video_filename"]=name;m["probe"]=probe
 (SITE/f"{method}-manifest.json").write_text(json.dumps(m,indent=2))
 for c in (25,26,27):shutil.copy2(out/f"sample-cycle{c:02d}.png",SITE/f"{method}-sample-cycle{c:02d}.png")
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(urllib.request.Request("http://100.65.117.46:8766/exp22c/poster-drafts/representative/"+name,headers={"Range":"bytes=0-1023"}),timeout=20) as response:
  assert response.status==206
  assert len(response.read())==1024
 print(json.dumps(dict(method=method,ready=True,duration_seconds=m["duration_seconds"],resets=m["human_resets"],frames=m["frames"],parity=m["outcome_parity_checks"])),flush=True)
(ROOT/"published.json").write_text(json.dumps(dict(ready=True,methods=["ours","ees"],cycles=[25,26,27])))
