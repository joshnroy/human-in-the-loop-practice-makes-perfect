"""Self-contained, lossless exploration of a recorded expectimax search."""

import json
from pathlib import Path

from .types import SearchTrace


class SearchHtml:
    """Export actual evaluations, including memoized successor references."""

    @staticmethod
    def write(*, path: Path, trace: SearchTrace, budget: float | None) -> None:
        payload = json.dumps({"budget": budget, "events": trace.events}).replace("<", "\\u003c")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SearchHtml.page.replace("TRACE_PAYLOAD", payload), encoding="utf-8")

    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>POMDP STOP decision — complete search</title>
<style>
body{font:16px system-ui;background:#151522;color:#eee;margin:24px}
summary,button{cursor:pointer}details{margin:10px 0 10px 18px;border-left:2px solid #777;
padding-left:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.value{color:#78deb1}button{padding:8px}h1{font-size:24px}
</style>
<h1>POMDP STOP decision — complete recorded search</h1>
<p id="metadata"></p>
<p>Expand each state, action, and outcome. All recorded branches and samples are included.
Shared cached states are linked to their full evaluations; nothing is recomputed.
Values are estimated utility, not observed improvement.</p>
<button id="download">Download complete trace JSON</button><main id="tree"></main>
<script type="application/json" id="data">TRACE_PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById("data").textContent);
const nodes=new Map(), keys=new Map();
const key=(state,belief,cost,horizon)=>JSON.stringify([state,belief,cost,horizon]);
for(const e of data.events){
 if(e.event==="node"){
  nodes.set(e.node,{info:e,events:[]});
  keys.set(key(e.environment_state,e.belief_state,e.summed_cost,e.horizon),e.node);
 }
 if(nodes.has(e.node))nodes.get(e.node).events.push(e);
}
const root=nodes.get(0);
document.getElementById("metadata").textContent=root?
 "Horizon "+root.info.horizon+" | Spent "+root.info.summed_cost+
 " | Budget "+(data.budget??"linear-cost mode")+" | Evaluated states "+nodes.size:
 "No recorded states";
function raw(parent,label,value){
 const d=document.createElement("details"),s=document.createElement("summary");
 s.textContent=label;d.append(s);const p=document.createElement("pre");
 p.textContent=JSON.stringify(value,null,2);d.append(p);parent.append(d);
}
function state(parent,id){
 const n=nodes.get(id),d=document.createElement("details"),s=document.createElement("summary");
 if(!n){raw(parent,"Missing recorded successor",id);return;}
 const choice=n.events.find(e=>e.event==="choice");
 s.textContent="State "+id+" | H="+n.info.horizon+" | cost="+n.info.summed_cost+
 " | V="+(choice?.value??"?")+" | choose "+JSON.stringify(choice?.action);
 s.className="value";d.append(s);parent.append(d);
 d.addEventListener("toggle",()=>{
  if(!d.open||d.dataset.built)return;d.dataset.built="yes";
  raw(d,"State and belief",n.info);
  const stop=n.events.find(e=>e.event==="stop_value");
  raw(d,"STOP value = "+stop?.value,n.events.filter(e=>
   e.event==="sample"||e.event==="stop_value"));
  const actions=n.events.filter(e=>e.event==="action_value");
  if(!actions.length){
   const p=document.createElement("p");
   p.textContent=n.info.horizon===0?"Horizon exhausted: no actions evaluated.":
    data.budget!==null&&n.info.summed_cost>=data.budget?
    "Budget exhausted: only STOP is allowed.":"No applicable affordable practice actions.";
   d.append(p);
  }
  for(const a of actions){
   const ad=document.createElement("details"),as=document.createElement("summary");
   as.textContent=JSON.stringify(a.action)+" | expected value="+a.value;
   ad.append(as);d.append(ad);
   for(const b of n.events.filter(e=>e.event==="branch"&&
      JSON.stringify(e.action)===JSON.stringify(a.action))){
    const bd=document.createElement("details"),bs=document.createElement("summary");
    bs.textContent="p="+b.probability+" × V="+b.successor_value+
     " = "+b.contribution+" | action cost="+b.sampled_cost;
    bd.append(bs);ad.append(bd);raw(bd,"Complete outcome record",b);
    state(bd,keys.get(key(b.successor,b.belief_state,b.summed_cost,b.horizon)));
   }
  }
  raw(d,"Selected action and reason",choice);
 });
 return d;
}
if(root)state(document.getElementById("tree"),0).open=true;
document.getElementById("download").onclick=()=>{
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],
 {type:"application/json"}));const a=document.createElement("a");
 a.href=url;a.download="search-trace.json";a.click();URL.revokeObjectURL(url);
};
</script></html>"""
