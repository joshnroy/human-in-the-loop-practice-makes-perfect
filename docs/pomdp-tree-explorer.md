# POMDP tree evolution explorer

Run from the repository root:

```bash
scripts/with_env.sh python -m analysis.pomdp_tree_explorer.server \
  --results-root results/my-experiment/pomdp \
  --port 8766
```

Open the loopback address printed by the server in your browser. Stop the server
with Ctrl-C. It only reads existing logs; it does not run the simulator, require
KINDER, or modify results. The browser assets are bundled: no CDN, Node runtime,
frontend build, or separate download is needed.

The root must contain numeric seed folders:

```text
results/my-experiment/pomdp/
  0/pomdp_decisions.jsonl
  1/pomdp_decisions.jsonl
```

Each file must contain the full POMDP `decision` records, including `search`
events. Other diagnostic records are ignored. Plain JSONL is currently supported;
compressed on-disk logs require a future reader extension. HTTP transfers of
individual decisions are losslessly gzip-compressed.

## Exploring a run

- Choose a seed, then a cycle/decision from the selector or previous/next buttons.
- The sidebar shows the completed decision's skill values and competence estimates.
- Expand states, actions, and chance outcomes to inspect probabilities, costs,
  weighted contributions, beliefs, and all recorded theta samples.
- Use Start search, Event, Play search, or the slider to replay structural events.
  Completed tree returns to the final recorded result.
- Download this decision saves its complete trace as JSON.

Replay advances through node/branch/value/choice events rather than stopping for
every theta sample. Samples are still present in the expandable records. Links
from parents to children become available when recursive evaluations return;
the active recorded node is shown separately before its parent returns. Cached
states are shared and may appear under more than one branch. No missing search
branches or terminal STOP decision are invented.

## Large logs and limitations

The first visit to a seed scans its log once to build a small byte-offset index;
this can take tens of seconds. Later selections seek directly to a decision.
Only one request is processed at a time, and only the selected search is loaded
into the browser, but a single large search can still need substantial memory.
An incomplete final JSONL line is ignored; malformed completed records are errors.
Reload the seed to refresh its index if a run is still appending logs.

For large experiments on Linux, a memory-capped foreground server is advisable:

```bash
scripts/with_env.sh systemd-run --user --scope \
  -p MemoryMax=2G -p OOMPolicy=continue \
  scripts/with_env.sh python -m analysis.pomdp_tree_explorer.server \
  --results-root results/my-experiment/pomdp
```

The server binds only to loopback, serves only its bundled assets and the selected
results root, and provides no mutation endpoints. Do not expose it as a public
web service. Downloaded JSON can be large; raw state/sample sections are rendered
only when opened.

## Tests

```bash
scripts/with_env.sh pytest -q tests/analysis/test_pomdp_tree_explorer.py
scripts/with_env.sh node tests/analysis/pomdp_tree_explorer_ui.cjs
```

The JavaScript test uses Node's built-in modules and a small DOM fixture
to check loading, navigation, lazy branch expansion, and replay. No browser test
framework is required; it is a functional test, not a visual-layout test.
