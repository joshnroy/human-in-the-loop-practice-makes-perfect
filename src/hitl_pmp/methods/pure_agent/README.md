# `pure_agent/` — the pure agent baseline (step 2.5)

`--method pure-agent`. Step 2.5 of the recipe in [*Now what? A recipe for after the
problem setting*](https://tomsilver.github.io/blog/2026/now-whats-your-solution/), which
slots between the random baseline (step 2, `--method random-skills`) and the privileged
oracle (step 3, `--method skill-oracle`).

This is **not** `pure_vla.py` or `in_context_vla.py`, the two VLA baselines the design
doc plans. Those wait on the EES reproduction; this one does not, because it needs no
learned component at all.

## The agent writes the policy; it is not the policy

The blog proposes querying an agent inside `step()`. The vehicle for this baseline —
[`prpl-agent-utils`](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/prpl-agent-utils),
and specifically its worked example `examples/pendulum_pure_agent.ipynb` — refuses that,
and says why: at 20 Hz a Pendulum episode is 200 actions and an evaluation is 4000, and
no agent closes that loop at the necessary rate or price. Verbatim from the notebook:

> So the pure agent writes the policy instead of being the policy.

So the agent authors a `policy.py` in a sandbox, that file is loaded and evaluated, and
the outcome is handed back for it to revise. **No API call happens at decision time**,
which is the only reason this is compatible with a harness that runs thousands of steps
per sweep.

The same arithmetic is worse here than on Pendulum, and this baseline measures it:
`PureAgentMethod.num_decisions()` counts every point at which the harness asked for an
action over a whole run. That count is the price of the be-the-policy variant, recorded
so that arm can be priced from a run that already happened rather than by building it.

## Agent and robot are two different things here

This is the one place in the repo where both exist at once, so the project's naming rule
matters more here than anywhere else:

- The **agent** is Claude Code. It runs in a sandbox, writes `policy.py`, and is not
  running at all by the time anything moves in the environment.
- The **robot** is what that file drives. Every step, every skill, every throw is the
  robot's.

So "the agent recovered the force relation" is a claim about the authoring; "the robot
never landed a throw" is a claim about the run. They can both be true at once, and the
whole point of this baseline is the gap between them.

## How it maps onto this harness — no `core/` change

| Notebook | Here |
| --- | --- |
| `train()` loops {query → load → score → feed back} | `end_cycle()`, which `PracticeLoop` calls after each interaction period and *before* that cycle's evaluation sweep |
| score on training seeds | `practice_outcomes()`, per-lifted-skill executions and how many hit that skill's own add effects |
| `step()` calls the authored function | `get_task_policy()` returns a `Policy` that does |

Round 0 is authored **lazily, on first use** — the initial evaluation sweep. A run with
`--num-cycles N` therefore authors N+1 policies, and sweep 0 measures what the agent can
write with no feedback at all, from the symbolic layer alone. That point separates "read
the domain and solved it" from "needed the feedback", and it is the interesting one.

## Domain-agnostic, and it selects ground skills

Everything domain-specific is read off the injected `SkillProvider` — the same interface
`RandomSkillsMethod` acts through. There is no `isinstance` dispatch and no environment
import anywhere in this subpackage, so the same class runs on any `--env`.

The authored policy picks an **index into the applicable ground skills** at the current
state, not a raw action, because the comparison of interest is against EES, which selects
ground skills. Indexing the applicable set means an authored policy structurally cannot
choose something whose preconditions fail; what it has to get right is which skill, and
what continuous parameters to give it.

The observation it is called with is plain JSON-shaped builtins:

    {
      "goal":    ["Predicate(obj, obj)", ...],
      "objects": [{"name": str, "type": str, "features": {name: float}}, ...],
      "atoms":   ["Predicate(obj, obj)", ...],
      "skills":  [{"index": int, "name": str, "objects": [str, ...], "param_dim": int}, ...]
    }

and returns `{"skill_index": int, "params": [float, ...]}`. Everything is sorted or in
provider order — two processes must build the same observation from the same state, or an
authored policy that breaks a tie by position becomes nondeterministic and replay stops
reproducing anything.

## Record-then-replay

Authoring queries a real agent: it costs money and does not repeat. So it never happens
inside a measured run.

- An **authoring** run queries the agent and writes an `AuthoringTranscript` — every
  round's `policy.py`, its prompt, and what the query cost.
- Every **measured** run replays that transcript (`--pure-agent-replay`), makes no API
  call at all, and writes a byte-stable `stats.json`.

The transcript carries **every** round, not just the final policy. A replay holding only
the last one would evaluate a fully-revised policy at every checkpoint, flattening the
learning curve into something indistinguishable from a method that converged immediately.
Running off the end of the recorded rounds raises rather than reusing the last — the same
argument `TossingRoomEnvironment`'s weight schedule makes for never wrapping.

It also resolves the sweep-width problem: authoring is serial, replay parallelises.

The two are **separate `--method` names**, not a flag on one:

    # authors, spends money, writes <output-dir>/transcript.json
    python -m hitl_pmp.cli --env tossingroom --method pure-agent-author \
      --pure-agent-sandbox-dir <fresh dir> --output-dir <dir> --num-cycles 2

    # measures, free, deterministic
    python -m hitl_pmp.cli --env tossingroom --method pure-agent \
      --pure-agent-replay <dir> --output-dir <other dir> --num-cycles 2

so a measured run has no code path that could reach a backend at all, and a mistyped
flag cannot spend money. Use a **fresh** sandbox directory per authoring run: the
conversation persists in it, so reusing one silently makes round 0 something other than a
zero-feedback policy.

On a machine where the shell predates the `docker` group being added, wrap the authoring
command in `sg docker -c "..."`; a bare `docker` call returns permission-denied on the
socket, which is not Docker being broken.

## Two prompt arms

`--pure-agent-prompt-arm minimal|described`.

- **minimal** — the symbolic layer and nothing else: lifted skills with their
  preconditions and effects, predicates, object types, and the objects to ground over.
  Exactly what a domain-agnostic planning `Method` knows, and not one word more.
- **described** — the same, plus a natural-language account of the domain supplied by the
  operator. The analogue of the notebook naming `Pendulum-v1`, which flags itself:
  *"The prompt names the environment, which is a large hint."*

`described` with an empty description is refused rather than degraded: it would be the
minimal arm wearing the other arm's label, and every reader downstream — including
`config_snapshot.json` — would pool the two.

## The security boundary, stated plainly

The sandbox protects **authoring**, not **evaluation**. `AuthoredPolicy` executes
agent-written code in this process; the notebook flags the same gap in its own text.
Closing it would mean evaluating inside the container too and passing only a score back.
Until then: **do not replay a transcript you did not author without reading its
`round_*_policy.py` files first.** They are written out beside `transcript.json` for
exactly that reason.

## What `practice_outcomes` means here

This arm has no `LearnedSkillSampler`, so `SamplerConsultation`'s pool split does not
carry its usual meaning. A `param_dim == 0` skill is filed `NO_SAMPLER`, which is exactly
true. A parameterized skill is filed `INFORMED`, on the reading that its parameters
reflect something the method learned — the authored policy *is* what this method learned.

**Only the overall `num_successes/num_attempts` is comparable to EES's.** The pools are
not, and a chart putting this arm's `informed` bucket beside EES's would be comparing a
classifier's argmax against a line of hand-written arithmetic.

## Testing

`ScriptedAgentBackend` is a deterministic stand-in that hands back a fixed sequence of
`policy.py` contents and records the prompts it was given. It is a mock of the *API*, not
of the agent: everything downstream — loading agent-written code, running it against the
observation contract, recovering from a file that does not import — is the real code path.
So the whole method is exercised in CI with no network, no Docker, no API key and no cost.
