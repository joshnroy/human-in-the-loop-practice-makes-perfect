# `pure_agent`: the agent IS the policy

The **pure agent** baseline — step 2.5 of Tom Silver's recipe, between the random
baseline (step 2) and the privileged oracle (step 3).

A Claude Code agent is queried **once per environment step**. It is handed the
observation at that decision point and replies with one line of JSON naming the ground
skill to execute and the continuous parameters to execute it with. Nothing it produces is
saved and re-executed; every action in every episode is a fresh decision.

This is the design the `prpl-agent-utils` notebook declines on cost grounds. It is
affordable here because a decision is one short assistant turn with **no tools**:
measured at 2.4–4.2 s and well under a cent per call on Opus. It is still the most
expensive thing in this repo by a wide margin — see "What a run costs" below, and read it
before choosing `--num-cycles`.

## The three decisions that shape this package

### 1. The conversation persists within an episode and is cleared at its boundary

- One practice period (default 150 steps) is **one conversation**. The agent sees what its
  last action did and adapts within the period.
- One evaluation episode (~12 steps on Tossing Room) is **one conversation**. The agent
  acts coherently across a plan.
- Neither conversation outlives its unit.

The alternative — one conversation for a whole run — is rejected twice over. It is
unaffordable: cost per call grows roughly linearly with conversation length (measured on a
40-turn probe: $0.0015 at turn 0, $0.0079 at turn 34), so a 5,000-turn run would spend
most of its money re-reading itself. And it cannot be firewalled: one conversation spanning
practice and evaluation carries every held-out task into the next one.

### 2. Exactly one thing crosses from practice to evaluation

At the end of each practice period the agent is asked to write down what it learned
(`PureAgentMethod.end_cycle`). That note — natural language, written by the agent for
itself, never executed — is prepended to the opening prompt of every later episode. It is
the whole learning channel across periods.

The **firewall** is that nothing else crosses, and it is structural rather than a
convention:

| guarantee | where it lives |
| --- | --- |
| practice and evaluation are two `AgentBackend` instances with two sandboxes and two conversations | `PureAgentMethod.practice_backend` / `.evaluation_backend` |
| the note is produced by a query on the **practice** conversation, so it can only contain what practice saw | `end_cycle` |
| no evaluation prompt has a parameter that could carry an outcome, a tally or a score | `PromptBuilder.evaluation_opening` / `.evaluation_step` |
| the evaluation conversation is reset at every held-out task | `get_task_policy` |

`tests/methods/pure_agent/test_pure_agent_method.py::test_no_evaluation_step_ever_reaches_the_practice_agent`
asserts the practice backend's call count **exactly**. Routing evaluation decisions
through the practice backend fails it at `120 == 12` — measured, by making that exact
mutation — rather than on a judgement call.

### 3. A run reproduces only as a replay of itself

Every call is journalled to `agent_calls.jsonl`, including a SHA-256 digest of the
observation it decided against. Replaying those decisions in order reproduces the run
exactly **as long as the state sequence is unchanged**, and the digest is what turns a
divergence into a loud failure instead of a silently different run.

What a replay does **not** do is reproduce the *method*. A second live run makes different
decisions, because the agent is not deterministic. So a replay is a re-scoring of one run,
never a second sample, and a set of replays is not a set of seeds.

## Isolation: what actually runs, and how it differs from the package default

`prpl_agent_utils.agents.ClaudeCodeAgent` defaults to `use_docker=True`. This backend
defaults to **False**, deliberately, and the deviation is worth stating precisely because
it changes the isolation properties:

- Docker **cannot** run on this machine at all: `docker images` returns
  `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`.
  So `True` is not a choice that is available here, it is a choice that fails.
- In host mode the CLI runs with `--dangerously-skip-permissions` and a `CLAUDE_CONFIG_DIR`
  pointed at `<sandbox>/.agent_home`, so the operator's live `~/.claude` is neither read
  nor written. With `CLAUDE_CODE_OAUTH_TOKEN` set, no credential is copied into the sandbox
  at all.
- The container's job is to contain an agent that writes files and runs commands. **This
  agent is given no tools** (`tools=""`), so the only thing it can emit is text. Measured:
  `num_tool_calls == 0` on every call of every probe and pilot run to date.

Turn Docker back on wherever it is available; the argument above is about what is
proportionate here, not about containers being unnecessary in general.

## Spending, and the two caps

Both caps are on by default, and the run-level one is **required** on the CLI:

| cap | flag | what it bounds |
| --- | --- | --- |
| per call | `--pure-agent-max-cost-usd-per-query` (default 0.50) | one long-tailed call |
| per run | `--pure-agent-max-total-cost-usd` (**required**) | the whole run |

The run-level one is the one that matters. A run makes one call per environment step —
thousands of them — against a weekly allowance that has **no overflow**
(`extra_usage.is_enabled: false`, `can_purchase_credits: false`): at 100% every agent on
the machine stops until the window resets, and there is no way to buy through it. When the
ceiling is reached the run makes no further calls, finishes on no-ops, and says so on
stderr; its results from that point are not a measurement of the method.

The per-call cap needs *handling*, not merely enabling: the CLI's budget stop emits a
`result` with `subtype: "error_max_budget_usd"` and **no `result` field**, which
`prpl_agent_utils._parse_stream` treats as fatal, discarding a decision already paid for.
`ClaudeCodeAgentBackend.recover_from_stream_log` reads both the cost and the agent's last
assistant text back out of the stream log the package already wrote, so a capped call that
had already answered still yields its answer.

## What a run costs

Decisions per run:

```text
num_cycles * max_steps_per_interaction            (practice)
  + (num_cycles + 1) * num_test_tasks * horizon   (evaluation)
```

plus one opening call per episode and one digest call per cycle. At the EES-matching
defaults on Tossing Room (`--num-cycles 10 --max-steps-per-interaction 150
--num-test-tasks 30`, horizon 12) that is `10*150 + 11*30*12 = 5,460` decisions. **Every
one is a network call**, and within a run they are strictly sequential.

**Measured cost per call, Opus, Tossing Room evaluation episodes** (92 calls, first
sweep of the pilot run): mean **$0.030**, median $0.023, max $0.204, `0/92` malformed,
`0/92` with any tool call, mean 3.7 s.

That is **not** the figure to extrapolate from the authoring arm, which measured
$1.64–$3.24 per call. The difference is what an agentic *round* is versus what a decision
is: the authoring arm spent a round reading, writing and revising a file over many turns,
while a decision here is one assistant turn with no tools against a short conversation.
Anything sized off the authoring number over-estimates this arm by roughly two orders of
magnitude, and anything sized off the number above must still account for practice
periods separately — those conversations run to 150 turns and their per-call cost climbs
with the context, where an evaluation episode's ~12 turns do not.

Across seeds they are independent and genuinely parallel: `prpl_agent_utils`'
`claude_auth.py` serialises only its file-copy credential fallback, and a stable
`CLAUDE_CODE_OAUTH_TOKEN` bypasses that path. Verified here — two concurrent runs
completed in 21.2 s against summed serial times of 20.2 s and 21.2 s, with `0/2` logs
containing `"Another Claude run is using file-based credentials"`.

Costs reported by the CLI are **subscription allowance, not an invoice**: the CLI
authenticates against a Claude subscription and no `ANTHROPIC_API_KEY` is set. They are the
API-equivalent price of the tokens, which is the right quantity for comparing arms and the
wrong one to call money owed.

## Reading a run

- `agent_calls.jsonl` — one `AgentCallRecord` per call, appended as it happens so a crash
  at hour four still reports what was spent. `phase` says which side of the firewall;
  `kind` says whether it was an opening, a decision or a digest.
- The `kind: "digest"` records hold the agent's own notes, in full. They are the most
  interesting thing a run produces: they are what it believed it had learned, in its own
  words, and they can be read against whether the next sweep improved.

## One caveat when this arm is plotted beside EES

`practice_outcomes()` files every parameterized attempt under
`SamplerConsultation.INFORMED`, because this baseline has no `LearnedSkillSampler` and
none of the four values describes it exactly (`EPSILON_RANDOM` is documented as "carrying
no belief", which is false here). So the headline `num_successes/num_attempts` **is**
comparable across arms and the informed/random/uninformative split is **not**. Quote the
first, never the second, when this arm sits beside EES.
