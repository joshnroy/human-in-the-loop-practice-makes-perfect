# Full EXP-22c poster replay

Selected source: workstation Model B, lambda3e-6, goal pursuit off, seed0,
experiment commit76732331. The retained plan covers50/50practice cycles and985/985actions.
`render_full.py` restores all92404/92404recorded simulation ticks and renders at20fps,
with no skipped frames or speed multiplier. Human resets have no recorded ticks;
their141/141indications each last1.5seconds. Each cycle has one2second opening banner.
There are no evaluation clips, end banners or additional outcome holds.

Total98634frames /20fps =4931.7seconds (82min11.7s). The only persistent overlays are
the current cycle skill log and cumulative153/418toss outcomes. Method, seed, cost,
bin side, commit and playback annotations are absent from the video.

`render_base.py` is the retained optimized renderer helper; `render_full.py` directly
restores backend snapshots to avoid recomputing motion-plan predicates for a replay.
It imports experiment code through the pinned EXP-22c env.sh, not this worktree.

`publish.py` waits for the complete manifest, checks stream dimensions/framecount/fps,
archives the original demo, then atomically publishes both the requested demo filename
and a distinct full-practice filename. It regenerates985/985timeline seek links with
frame-derived offsets. Publication checks HTTP206 byte-range support. Renderer and
source plan hashes and all cycle/action offsets are in demo-manifest.json.

Persistent services: exp22c-full-poster-demo and exp22c-full-poster-publish, each6GiB,
MemorySwapMax0. Run outputs and progress are under results sibling exp22c-poster-full-demo.
