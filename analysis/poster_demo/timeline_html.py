"""Offline full-run timeline; links available replay actions at frame-exact offsets."""
import html,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
SITE=Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c/poster-drafts")
SITE.mkdir(parents=True,exist_ok=True)
plan=json.loads((HERE/"plan.json").read_text())
run=Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c/gpoff/results/local_trend-lam3e-6/pomdp/0")
stats=json.loads((run/"stats.json").read_text())
colors={"PickCube":"#0072B2","MoveToTossLocationAndToss":"#8E44AD","ask_for_reset_cube_bin_only":"#009E73","OpenGripper":"#6B7A8F"}
parts=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP-22c practice timeline</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:5px}a.box,span.box{display:block;border-radius:4px;min-width:72px;height:64px;text-align:center;padding:6px;color:white;text-decoration:none;font-size:12px}.scroll{overflow:auto;max-height:80vh;border:1px solid #ddd}thead th{position:sticky;top:0;background:white;z-index:2}tbody th{position:sticky;left:0;background:white;z-index:1}img,video{max-width:100%}</style></head><body><p><a href="./">Poster drafts</a></p><h1>EXP-22c: all 50 practice episodes</h1><p>Ours Model B λ3e-6, goal pursuit off, workstation seed0, commit76732331. Complete run: 50/50 recorded practice episodes. All episodes are included; no outcome-based selection. S/F is the recorded practice outcome, not the far-side evaluation score. Human reset destinations are planner choices, read from action.objects[-1].</p><p>Demo: all50/50 practice cycles in chronological order at1× recorded simulation speed. Every recorded physics tick is shown at20fps. Each cycle begins with a2s title; human resets have no recorded physics ticks and use a1.5s indication of the next recorded state. No evaluation footage is included. All985/985 action boxes seek to the corresponding video moment.</p>']
parts.append('<p><b>Coverage: 50/50 practice episodes, 985 recorded actions.</b> Human resets: 141/985 actions (137/141 robot-side, 4/141 far-side). Toss outcomes: 153/418 scored. <a href="historical-first10/timeline.html">Historical first-ten excerpt</a>.</p>')
if (SITE/'demo.mp4').exists():parts.append('<video id="demo" controls preload="none" src="demo.mp4"></video>')
parts.append('<div class="scroll"><table><thead><tr><th>Practice episode</th>'+''.join(f'<th>{i+1}</th>' for i in range(20))+'<th>After-episode eval</th></tr></thead><tbody>')
video_manifest=json.loads((SITE/'demo-manifest.json').read_text())
timing={(a['cycle'],a['action']):a['start_frame']/video_manifest['fps'] for a in video_manifest['actions']}
provenance=[]
for cycle in plan['practice']:
 c=cycle['cycle'];parts.append(f'<tr><th>{c+1}</th>');seconds=2.0
 for i,a in enumerate(cycle['actions']):
  color=colors.get(a['skill'],'#666');label=html.escape(a['label'])+'<br>'+('S' if a['success'] else 'F')
  start=timing[(c+1,i+1)]
  if a['block'] is None:seconds+=1.5
  else:seconds+=math.ceil(a['ticks']/2)/40+.75
  title=html.escape(f"Cycle{c+1}, action{i+1}; decision{a['decision']}; ticks{a['ticks']}; destination{a.get('side')}")
  if (SITE/'demo.mp4').exists():cell=f'<a class="box" style="background:{color}" href="demo.mp4#t={start:.3f}" title="{title}">{label}</a>'
  else:cell=f'<span class="box" style="background:{color}" title="{title}">{label}</span>'
  parts.append('<td>'+cell+'</td>');provenance.append(dict(cycle=c+1,action=i+1,seconds=start,rendered=True,**a))
 parts.extend('<td></td>' for _ in range(20-len(cycle['actions'])))
 parts.append(f"<td>{stats['evaluations'][c+1][1]}/10</td></tr>")
parts.append('</tbody></table></div><p>Interactive boxes encode skills, with an explicit key: blue=Pick, purple=Toss, green=human reset, grey=OpenGripper. Learning-curve colours retain the method key.</p><p><a href="b-practice-timeline.png">Poster PNG</a> · <a href="b-practice-timeline.svg">Poster SVG</a> · <a href="timeline-provenance.json">Action/timing provenance</a></p><img loading="lazy" src="b-practice-timeline.png" alt="All 50 practice episodes"></body></html>')
(SITE/'timeline.html').write_text(''.join(parts));(SITE/'timeline-provenance.json').write_text(json.dumps(provenance,indent=1))
print('TIMELINE',SITE/'timeline.html')
