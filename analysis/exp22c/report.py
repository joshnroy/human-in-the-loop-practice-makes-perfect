"""EXP-22c figures, paired tests and self-contained static page."""
import html
import itertools
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.colors import to_rgb
from arms import ARMS
from comparisons import COMPARISONS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SITE = Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c")
POST = SITE / "poster-drafts"
POST.mkdir(parents=True, exist_ok=True)
rows = json.loads((DATA / "rows.json").read_text())
e23 = json.loads((DATA / "exp23.json").read_text())
R = defaultdict(dict)
for r in rows:
    R[(r["src"],r["group"],r["arm"])][r["seed"]] = r
COL = {"A":"#56B4E9", "B":"#0072B2", "ees":"#009E73"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

def paired_values(x,y,seeds):
    d=np.array(x,dtype=float)-np.array(y,dtype=float)
    result={"seeds":seeds,"n":len(d),"x":list(x),"y":list(y),"mean_difference":float(np.mean(d)) if len(d) else None,"exact_p":None,"paired_t_p":None}
    if len(d)<2:
        return result
    observed=abs(np.mean(d))
    permutations=[abs(np.mean(d*np.array(sign))) for sign in itertools.product((-1,1),repeat=len(d))]
    result["exact_p"]=sum(v>=observed-1e-12 for v in permutations)/len(permutations)
    if np.all(d==0):
        result["paired_t_p"]=1.0
    elif np.std(d,ddof=1)>1e-12:
        result["paired_t_p"]=float(stats.ttest_rel(x,y).pvalue)
    else:
        result["paired_t_note"]="Undefined: zero sample variance in nonzero differences"
    return result

def paired(src,a,b,metric):
    aa,bb=R[(src,)+a],R[(src,)+b]
    seeds=sorted(set(aa)&set(bb))
    return paired_values([aa[s][metric] for s in seeds],[bb[s][metric] for s in seeds],seeds)

tests=[]
for src in ("della","ws"):
    for label,a,b in COMPARISONS:
        for metric in ("last5","resets","peak"):
            tests.append(dict(src=src,label=label,metric=metric,**paired(src,a,b,metric)))
exp23_tests=[]
for a,b in ((1,5),(20,5),(20,1)):
    aa={e["seed"]:e for e in e23 if e["cost"]==a}
    bb={e["seed"]:e for e in e23 if e["cost"]==b}
    seeds=sorted(set(aa)&set(bb))
    for metric in ("last5","resets_dec","optional"):
        val=lambda r:r["split"].get("optional",0) if metric=="optional" else r[metric]
        exp23_tests.append(dict(label=f"cost {a} minus {b}",metric=metric,**paired_values([val(aa[s]) for s in seeds],[val(bb[s]) for s in seeds],seeds)))
(DATA / "tests.json").write_text(json.dumps({"exp22c":tests,"exp23c":exp23_tests},indent=1,allow_nan=False))

def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=170,bbox_inches="tight");plt.close(fig)

def curves(ax,src,g,a,col,reset,label):
    rr=R[(src,g,a)]
    if not rr:return
    ev=np.array([r["evals"] for r in rr.values()])
    ls="-" if reset else (0,(4,2))
    for e in ev:ax.plot(range(len(e)),e,color=col,ls=ls,alpha=.16,lw=.8)
    ax.plot(range(ev.shape[1]),ev.mean(axis=0),color=col,ls=ls,lw=2.3,label=f"{label} — n={len(ev)}/3 seeds")

def learning(src):
    fig,axs=plt.subplots(1,2,figsize=(12,4.3),sharey=True)
    for ax,(g,a,nr,col,label) in zip(axs,[("gpoff","local_trend-lam3e-6","noreset",COL["B"],"Ours B, λ3e-6"),("ees-gpoff","ees-lamcost5","ees-noreset",COL["ees"],"EES, cost5")]):
        curves(ax,src,g,a,col,True,label+", reset")
        curves(ax,src,nr,a,col,False,label+", no reset")
        ax.set(xlabel="Practice cycle",ylim=(-.3,10.3),title=label+": eval tasks solved (of 10)")
        ax.legend(fontsize=8,loc="lower right")
    axs[0].set_ylabel("Tasks solved per seed")
    fig.suptitle(f"EXP-22c · Tossing3D · {'Della' if src=='della' else 'workstation'} · faint traces=individual seeds")
    save(fig,SITE/f"learning-{src}.png")
    shutil.copy2(SITE/f"learning-{src}.png",POST/f"a-learning-{src}.png")

def tradeoff(src,gp):
    fig,(ax,table_ax)=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={"width_ratios":[1.8,1]})
    trade_rows=[]
    for group,key,label in (("gpon" if gp else "gpoff","A","Ours A λ"),("gpon" if gp else "gpoff","B","Ours B λ"),("ees-gpon" if gp else "ees-gpoff","ees","EES cost")):
        selected=[a for a in ARMS if a[0]==group and a[3]==key]
        pts=[]
        for ki,(g,a,l,k,reset) in enumerate(selected):
            mix=.40*(1-ki/max(1,len(selected)-1))
            shade=tuple((1-mix)*v+mix for v in to_rgb(COL[key]))
            rr=list(R[(src,g,a)].values())
            if not rr:continue
            x=[r["resets"] for r in rr];y=[r["last5"] for r in rr]
            ax.scatter(x,y,color=shade,s=20,alpha=.3)
            mx,my=np.mean(x),np.mean(y);pts.append((mx,my))
            ax.scatter([mx],[my],color=shade,s=80,edgecolors="black",lw=.4)
            trade_rows.append(["Ours "+key if key != "ees" else "EES",f"{rr[0]['knob']:g}",f"{len(rr)}/3",f"{mx:.1f}",f"{my:.2f}/10"])
        if pts:ax.plot(*zip(*pts),color=COL[key],lw=1.5,label=label)
    for g,a,k,label in (("noreset","local_trend-lam3e-6","B","Ours B no reset"),("ees-noreset","ees-lamcost5","ees","EES no reset")):
        rr=list(R[(src,g,a)].values())
        if rr:
            ax.scatter([r["resets"] for r in rr],[r["last5"] for r in rr],color=COL[k],marker="s",s=20,alpha=.3)
            ax.plot([np.mean([r["resets"] for r in rr])],[np.mean([r["last5"] for r in rr])],color=COL[k],marker="s",ls="--",label=f"{label}, n={len(rr)}/3")
    ax.set(xlabel="Human resets per run",ylabel="Mean eval solved, last 5 checkpoints (of 10)",ylim=(-.3,10.3),title=f"Tossing3D · {src} · GP {'on' if gp else 'off'} · dots=seeds; large=means")
    ax.legend(fontsize=8,loc="lower right")
    table_ax.axis("off")
    tab=table_ax.table(cellText=trade_rows,colLabels=["Method","λ / cost","Seeds","Resets","Eval"],loc="center")
    tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.5)
    table_ax.set_title("Arm means (reset available)",fontsize=10)
    save(fig,SITE/f"tradeoff-{src}-gp{int(gp)}.png")

def refusals(src):
    fig,ax=plt.subplots(figsize=(15,4.5))
    for i,(g,a,label,k,reset) in enumerate(ARMS):
        for s,r in sorted(R[(src,g,a)].items()):
            x=i+(s-1)*.2
            ax.scatter(x,r["refused"],color=COL[k],s=20)
            ax.annotate(f"{r['refused']}/{r['picks']}",(x,r["refused"]),rotation=90,fontsize=6,ha="center",xytext=(0,6),textcoords="offset points")
    ax.set_xticks(range(len(ARMS)));ax.set_xticklabels([a[2] for a in ARMS],rotation=40,ha="right",fontsize=7)
    ax.set(ylabel="Refused practice picks",title=f"Tossing3D · {src} · each label=refused/practice picks; seeds 0,1,2 left→right")
    ax.set_ylim(-.3,max([r["refused"] for r in rows if r["src"]==src]+[1])+2)
    save(fig,SITE/f"refusals-{src}.png")

for src in ("della","ws"):
    learning(src);tradeoff(src,False);tradeoff(src,True);refusals(src)

fig,axs=plt.subplots(1,2,figsize=(11,4.6))
for ci,c in enumerate((1,5,20)):
    rr=sorted([e for e in e23 if e["cost"]==c],key=lambda e:e["seed"])
    for e in rr:
        x=ci*4+e["seed"];opt=e["split"].get("optional",0);n=e["resets_dec"]
        axs[0].bar(x,opt,color=COL["B"],label="Optional: root has PickCube" if x==0 else None)
        axs[0].bar(x,n-opt,bottom=opt,color="#999999",label="Needed: root has no PickCube" if x==0 else None)
        axs[0].text(x,n+3,f"{opt}/{n}",ha="center",fontsize=7)
        axs[1].scatter(ci+(e["seed"]-1)*.06,e["last5"],color=COL["B"],s=25,alpha=.5)
    axs[1].scatter(ci,np.mean([e["last5"] for e in rr]),color=COL["B"],s=95,edgecolor="black")
axs[0].set_xticks(range(0,11));axs[0].set_xticklabels(["s0","s1","s2","","s0","s1","s2","","s0","s1","s2"],fontsize=8)
axs[0].set(xlabel="Reset cost1                       cost5                       cost20",ylabel="Human resets per run",title="Optional/total dispatched resets · Della · n=3/3 each")
axs[0].legend(fontsize=8)
axs[1].set(xticks=[0,1,2],xticklabels=["1","5","20"],xlabel="Human reset cost",ylabel="Mean eval solved, last 5 checkpoints (of 10)",ylim=(-.3,10.3),title="Small dots=seeds; large dots=means")
fig.suptitle("EXP-23c · Tossing3D · Ours B λ3e-4 · Della-only cost comparison")
save(fig,SITE/"exp23-resets.png");shutil.copy2(SITE/"exp23-resets.png",POST/"d-exp23-optional-vs-needed.png")

# Machine comparisons are descriptive; same seeds are not identical trajectories.
pairs=[]
for g,a,label,k,reset in ARMS:
    ws,de=R[("ws",g,a)],R[("della",g,a)]
    for seed in sorted(set(ws)&set(de)):
        w,d=ws[seed],de[seed]
        sw=json.loads((DATA/"scan"/(w["tag"]+".json")).read_text())["skills"]
        sd=json.loads((DATA/"scan"/(d["tag"]+".json")).read_text())["skills"]
        first_hash=next((i for i,(x,y) in enumerate(zip(sw,sd)) if x["hash"]!=y["hash"]),None)
        if first_hash is None and len(sw)!=len(sd):first_hash=min(len(sw),len(sd))
        pairs.append(dict(group=g,arm=a,label=label,key=k,seed=seed,first_state_hash_difference=first_hash,ws_last5=w["last5"],della_last5=d["last5"],ws_resets=w["resets"],della_resets=d["resets"],stats_identical=(Path(w["run"])/"stats.json").read_bytes()==(Path(d["run"])/"stats.json").read_bytes()))
(DATA/"machine-comparison.json").write_text(json.dumps(pairs,indent=1))
fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for p in pairs:
    axs[0].scatter(p["ws_last5"],p["della_last5"],color=COL[p["key"]],s=20,alpha=.7)
    axs[1].scatter(p["ws_resets"],p["della_resets"],color=COL[p["key"]],s=20,alpha=.7)
for ax,lim,label in ((axs[0],10,"Last5 mean eval (of 10)"),(axs[1],max([max(p["ws_resets"],p["della_resets"]) for p in pairs]+[300]),"Human resets")):
    ax.plot([0,lim],[0,lim],color="#999999",ls=":")
    ax.set(xlabel="Workstation",ylabel="Della",title=label)
fig.suptitle(f"Tossing3D · {len(pairs)}/57 matched arm/seed pairs · descriptive, separate machines")
save(fig,SITE/"machine-comparison.png")

def esc(x):return html.escape(str(x))
def table(headers,records):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in records)+'</tbody></table></div>'
def pv(x):return "not estimable" if x is None else f"{x:.4g}"
def img(name,alt):return f'<a href="{name}"><img loading="lazy" src="{name}" alt="{esc(alt)}"></a>'
stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
nd=sum(r["src"]=="della" for r in rows);nw=sum(r["src"]=="ws" for r in rows)
page=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP-22c / EXP-23c</title><style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;color:#182330}a{color:#0072B2}img{max-width:100%}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #ccd4dd;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8;position:sticky;top:0}.scroll{overflow:auto;max-height:650px}.note{background:#edf3f8;padding:16px}section{margin:36px 0}video{max-width:100%}</style></head><body>',f'<p><a href="../">Experiment log</a> · <a href="poster-drafts/">Poster drafts</a></p><h1>EXP-22c and EXP-23c</h1><p>Updated {stamp}. Commit <code>76732331</code>; PickCube is available only when its own controller planner finds a plan.</p>',f'<p class="note"><b>Status:</b> Della EXP-22c {nd}/57 complete; EXP-23c {len(e23)-3}/6 additional runs; workstation {nw}/57 complete. Incomplete workstation arms are provisional. Each machine is analysed separately; same seed does not imply the same trajectory across machines.</p>','<h2>Question and method</h2><p>Can the practice planner learn far-side tossing while choosing when to pay for human resets and which bin side to request? EXP-22c keeps the 19-arm, seeds 0–2 design. EXP-23c varies human reset cost 1/5/20 at Model B λ3e-4 with goal pursuit off. Cost 5 comes from Della EXP-22c. No forced resets or experiment-design changes.</p>','<p>Primary summary: mean tasks solved at the last 5 evaluation checkpoints, each of 10 tasks. Tables also give the summed numerator over 50 evaluation trials per run. These repeated checkpoints are not treated as independent seeds. Human resets are dispatched resets over 50 practice cycles. Practice/eval action labels are separated with the EXP-22b episode-trace matching scanner; POMDP pick/toss counts must agree with stats. Workstation EES counts must agree with independent practice_events, and the documented omission of the last action in capped sessions must exactly reconcile to stats. Della does not retain that sidecar: its independent-event check is unavailable, and only compatibility with capped-cycle omissions is checked (at most one missing final physical action per capped cycle, none otherwise). Refusal denominators count all executed/logged picks, including these EES final actions.</p>','<p><b>Inference:</b> comparisons are paired by seed within machine. Exact two-sided sign-flip tests use the absolute mean paired difference (all 2ⁿ sign assignments), assuming independent seeds and symmetric paired differences under the null hypothesis. At n=3 the smallest attainable two-sided exact p-value is 0.25. Paired t-tests are supplementary, assume normally distributed paired differences, and are fragile at n=3. Tests are exploratory/unadjusted; no cross-arm pooling or cross-machine replay claim. Zero variance in a nonzero difference is reported as not estimable rather than p=0.</p>']
core=[]
for src in ("della","ws"):
    for g,a,label in (("gpoff","local_trend-lam3e-6","Ours B λ3e-6, reset"),("noreset","local_trend-lam3e-6","Ours B λ3e-6, no reset"),("ees-gpoff","ees-lamcost5","EES cost5, reset"),("ees-noreset","ees-lamcost5","EES, no reset")):
        rr=list(R[(src,g,a)].values())
        if rr:
            core.append([src,label,f"{len(rr)}/3",f"{np.mean([r['last5'] for r in rr]):.2f}/10",f"{np.mean([r['resets'] for r in rr]):.1f}",f"{sum(r['refused'] for r in rr)}/{sum(r['picks'] for r in rr)}"])
page += ['<h2>Core descriptive results</h2>',table(['Machine','Arm','Seeds complete','Last-5 mean eval','Mean human resets','Refused/practice picks'],core),'<p>Same preselected baseline arms as EXP-22b; no selection of the best-performing λ. These means are descriptive. Reset-versus-no-reset paired tests:</p>',table(['Machine','Comparison','Metric','Pairs','Mean difference','Exact p','Paired t p'],[[t['src'],t['label'],t['metric'],f"{t['n']}/3",f"{t['mean_difference']:.3g}" if t['mean_difference'] is not None else 'pending',pv(t['exact_p']),pv(t['paired_t_p'])] for t in tests if 'reset vs no reset' in t['label'] and t['metric'] in ('last5','resets')])]

for src,name in (("della","Della"),("ws","Workstation")):
    rr=[r for r in rows if r["src"]==src]
    fp=sum(r["refused"] for r in rr);pp=sum(r["picks"] for r in rr);fe=sum(r["eval_refused"] for r in rr);pe=sum(r["eval_picks"] for r in rr)
    page += [f'<section><h2>{name}: results</h2>',f'<p>Refused practice picks: <b>{fp}/{pp}</b>; evaluation picks: <b>{fe}/{pe}</b>. The initial seed 0 handoff denominators combined practice and eval; this page separates them and preserves that earlier report.</p>',img(f'learning-{src}.png','Reset versus no reset'),img(f'tradeoff-{src}-gp0.png','Evaluation versus human resets, GP off'),img(f'tradeoff-{src}-gp1.png','Evaluation versus human resets, GP on')]
    armrows=[]
    for g,a,label,k,reset in ARMS:
        runset=R[(src,g,a)]
        armrows.append([label,f"{len(runset)}/3",*[f"{runset[s]['last5']:.1f}/10; {runset[s]['resets']} resets" if s in runset else "pending" for s in range(3)],f"{np.mean([r['last5'] for r in runset.values()]):.2f}/10" if runset else "pending",f"{np.mean([r['resets'] for r in runset.values()]):.1f}" if runset else "pending"])
    page += [table(["Arm","Complete seeds","Seed0 last5; resets","Seed1 last5; resets","Seed2 last5; resets","Mean last5","Mean resets"],armrows),'<h3>Paired comparisons</h3>',table(["Comparison","Metric","Pairs","A−B mean","Exact p","Paired t p"],[[t['label'],t['metric'],f"{t['n']}/3",f"{t['mean_difference']:.3g}" if t['mean_difference'] is not None else "pending",pv(t['exact_p']),pv(t['paired_t_p'])] for t in tests if t['src']==src]),'<h3>Every completed run</h3>',img(f'refusals-{src}.png','Refused picks per run'),table(["Group / arm","Seed","Last5 sum","Last5 mean","Peak","Resets / practice cycles","Practice refused/picks","Eval refused/picks","Longest refused streak","Elapsed hours","Scan"],[[r['group']+' / '+r['arm'],r['seed'],f"{r['last5_solved']}/50",f"{r['last5']:.1f}/10",f"{r['peak']}/10",f"{r['resets']}/50",f"{r['refused']}/{r['picks']}",f"{r['eval_refused']}/{r['eval_picks']}",r['longest'],f"{r['elapsed']/3600:.2f}" if r['elapsed'] is not None else 'unavailable',("valid; EES event parity" if r['ees_accounting']['independent_events_available'] else "state/eval valid; EES cap-compatible, no sidecar") if r.get('ees_accounting') else "valid"] for r in rr]),'</section>']
page += ['<section><h2>EXP-23c: optional versus needed resets</h2><p><b>Operational definition:</b> a dispatched reset is optional exactly when PickCube appears among valid root actions in that decision log; otherwise it is labelled needed. This preserves the EXP-23b classification. Needed means pick unavailable under the robot’s current symbolic model, not a proof that a human is physically necessary or that every other robot action is impossible. PickPlannable replaces obsolete GraspClear explanations. Counts must match dispatched resets in stats. Destination is read from the last action object.</p>',img('exp23-resets.png','EXP23 optional versus needed resets'),table(['Reset cost','Seed','Optional/total','Needed/total','Last5','Reset destinations','Needed categories'],[[e['cost'],e['seed'],f"{e['split'].get('optional',0)}/{e['resets_dec']}",f"{e['resets_dec']-e['split'].get('optional',0)}/{e['resets_dec']}",f"{e['last5']:.1f}/10",str(e['dest']),str({k:v for k,v in e['split'].items() if k!='optional'})] for e in sorted(e23,key=lambda e:(e['cost'],e['seed']))]),table(['Paired cost comparison','Metric','Pairs','Mean difference','Exact p','Paired t p'],[[e['label'],e['metric'],f"{e['n']}/3",f"{e['mean_difference']:.3g}",pv(e['exact_p']),pv(e['paired_t_p'])] for e in exp23_tests]),'</section>','<section><h2>Della and workstation</h2><p>Different floating-point trajectories are additional machine-specific samples, not replays. Descriptive matched-arm/seed comparison below; no pooling of arm observations as independent seeds. State hashes follow the EXP-22b scanner (skill/parameters and ticks, timestamps excluded); differing skill indices are zero-based. Equal aggregate stats do not establish a replay.</p>',img('machine-comparison.png','Della versus workstation descriptive comparison'),table(['Arm','Seed','Workstation eval','Della eval','Workstation resets','Della resets','Stats byte identical','First state-hash differing skill'],[[p['label'],p['seed'],f"{p['ws_last5']:.1f}/10",f"{p['della_last5']:.1f}/10",p['ws_resets'],p['della_resets'],str(p['stats_identical']),p['first_state_hash_difference'] if p['first_state_hash_difference'] is not None else 'none'] for p in pairs]),'</section>','<h2>Artifacts and provenance</h2><p><a href="poster-drafts/">Poster drafts: curves, timeline, demo and EXP-23 figure</a>. <a href="rows.json">Per-run data</a> · <a href="tests.json">All paired tests</a> · <a href="validation.json">Scanner validation</a> · <a href="exp23.json">EXP-23 reset categories</a>.</p><p>Generated by analysis/exp22c in isolated branch codex/exp22c-analysis. Scan and decision extraction are adapted from the backed-up EXP-22b pipeline. Remote logs are retained until verified local copies exist.</p></body></html>']
(SITE/'index.html').write_text(''.join(page))
for name in ('rows.json','tests.json','validation.json','exp23.json','machine-comparison.json'):
    shutil.copy2(DATA/name,SITE/name)
poster=['<!doctype html><html><head><meta charset="utf-8"><title>EXP-22c poster drafts</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px}img,video{max-width:100%}</style></head><body><p><a href="../">Full EXP-22c / EXP-23c analysis</a></p><h1>Poster drafts · EXP-22c / EXP-23c</h1>',f'<p>Updated {stamp}. Della {nd}/57, workstation {nw}/57. Workstation curves provisional until all 57/57 finish. No exact n=3 paired test can attain p&lt;0.05.</p>', '<h2>Reset versus no reset</h2>',img('a-learning-della.png','Della learning curves'),img('a-learning-ws.png','Workstation learning curves'),'<h2>Practice timeline and demo</h2>']
if (POST/'timeline.html').exists():poster.append('<p><a href="timeline.html">All 50/50 EXP-22c practice episodes</a></p>')
else:poster.append('<p>EXP-22c timeline preparation in progress; prior EXP-22b artifact is not relabelled as EXP-22c.</p>')
if (POST/'demo.mp4').exists():poster.append('<video controls preload="none" src="demo.mp4"></video><p>State-log replay of the documented EXP-22c practice cycle; see timeline for provenance.</p>')
else:poster.append('<p>EXP-22c demo replay in progress.</p>')
poster += ['<h2>EXP-23c optional versus needed resets</h2>',img('d-exp23-optional-vs-needed.png','EXP23 figure'),'</body></html>']
(POST/'index.html').write_text(''.join(poster))
print('PUBLISHED',SITE,'Della',f'{nd}/57','workstation',f'{nw}/57',flush=True)
