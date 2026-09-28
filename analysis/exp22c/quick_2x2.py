"""Immediate completed-only Della core2x2 curves; no EXP23 dependency."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path("/home/josh/Documents/repos/research/hitl-pmp-results/exp22c-della")
SITE=Path("/home/josh/Documents/repos/research/hitl-pmp-site/tossing3d-baseline/exp22c")
fig,axs=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
summary=[]
for ax,(group,nr,arm,method,color,label) in zip(axs,[("gpoff","noreset","local_trend-lam3e-6","pomdp","#0072B2","Ours Model B · λ3e-6"),("ees-gpoff","ees-noreset","ees-lamcost5","ees","#009E73","EES · reset cost5")]):
    for reset,g in ((True,group),(False,nr)):
        ev=[]
        for seed in range(3):
            p=ROOT/g/"results"/arm/method/str(seed)
            s=json.loads((p/"stats.json").read_text());t=json.loads((p/"timing.json").read_text());cfg=json.loads((p/"config_snapshot.json").read_text())
            assert t["returncode"]==0 and t["succeeded"] and len(s["evaluations"])==51 and len(s["practice_session_ends"])==50
            assert cfg["git_commit"].startswith("76732331")
            assert all(e[2]==10 for e in s["evaluations"])
            y=[e[1] for e in s["evaluations"]];ev.append(y)
            summary.append(dict(group=g,arm=arm,seed=seed,last5=sum(y[-5:])/5,resets=s["num_human_interventions_recorded"]))
        style="-" if reset else (0,(4,2))
        for y in ev:ax.plot(range(51),y,color=color,ls=style,alpha=.16,lw=.8)
        ax.plot(range(51),np.mean(ev,axis=0),color=color,ls=style,lw=2.5,label=("Human reset available" if reset else "No human reset")+" · 3/3 seeds")
    ax.set(title=label,xlabel="Practice cycle",ylim=(-.3,10.3),xlim=(0,50))
    ax.legend(loc="lower right",fontsize=9);ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel("Evaluation tasks solved per seed")
fig.suptitle("EXP-22c · Tossing3D · Della completed runs · evaluation tasks (of10)\nGoal pursuit off; faint traces=individual seeds, bold=mean",fontsize=12)
fig.tight_layout();fig.savefig(SITE/"della-core-2x2-ready.png",dpi=180);fig.savefig(SITE/"della-core-2x2-ready.svg");plt.close(fig)
(SITE/"della-core-2x2-ready.json").write_text(json.dumps(summary,indent=1))
print(json.dumps(summary,indent=1));print("READY",SITE/"della-core-2x2-ready.png")
