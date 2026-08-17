---
name: tracker
description: Render the project's task tracker — live PR state across this repo, kinder-baselines and kindergarden, grouped into correctness/methods/infra tiers, ending with the decisions waiting on Josh. Use when asked for status, the tracker, "where are we", what to review, or what is blocked; and unprompted after a merge, after an agent reports, or when several things are in flight.
---

# Tracker

Josh's standing view of the project. **Always query live state — never render it from memory.**

## The model

- A **task** is the unit of planning. A **PR is a subtask**. Implementing a task means shipping a stack of PRs.
- **Work that belongs together stays in one task and one tier.** "Implement the bilevel planning model" and "consume it here" are one task with two subtasks.
- Tiers, in priority order:
  0. **Public PRs** — anything open on a repo Josh does not solely own (`Princeton-Robot-Planning-and-Learning/*`, and any third-party repo). These come first *regardless of their content*, because someone else's attention is already committed to them: a maintainer may be reviewing, CI is burning, and a stale or wrong public PR costs another person's time rather than ours. A one-line docs fix upstream outranks a correctness bug of ours. Note this is about **where the PR lives**, not who wrote it — a PR on `joshnroy/*` is never tier 0 no matter how important.
  1. **Correctness** — the environment, its skills, or the measurement pipeline are *wrong*, so anything measured on them is suspect. First regardless of size.
  2. **Methods and experiments** — the long-running scientific work.
  3. **Infra** — tooling, docs, ergonomics. **Exception:** infra that materially speeds up tier-2 work is promoted to tier 2.
- When several tasks are ready, **start the longest-running agent work first**, so expensive jobs run while cheap ones are discussed.

## Gather state

Three repos. Ours, plus two lab-owned upstreams.

```bash
cd <repo-root>
git fetch -q origin && git log --oneline -1 origin/main
gh pr list --state open --json number,title,baseRefName,mergeStateStatus,isDraft,reviewDecision,reviewRequests \
  --jq '.[]|"#\(.number) base=\(.baseRefName) \(.mergeStateStatus) \(.reviewDecision // "none") \(.title)"'
gh pr checks <n>            # per open PR

R=Princeton-Robot-Planning-and-Learning
gh pr list --repo $R/kinder-baselines --state open --json number,title,isDraft,mergeStateStatus
gh pr list --repo $R/kindergarden     --state open --json number,title,isDraft,mergeStateStatus
```

Check `mergeStateStatus` on every PR. `BEHIND` means it has never been tested against
what it would merge into — get it rebased **before** surfacing it, and prefer resuming
the agent that owns the branch (a stale checkout of an agent's branch has nearly
clobbered work). `BLOCKED` on a draft usually just means draft + review-required, not a
problem; say which.

**Exception: an agent far into an experiment finishes first, then rebases** — provided
the incoming changes cannot affect the result. Check that rather than assume it; if
`main` moved the dynamics, the sampler or the analysis module, the numbers are stale and
the rebase means a re-run. Surface such a PR as `BEHIND, rebasing after results` rather
than holding it back.

A `lint`/`test` failure is sometimes GitHub infrastructure, not the diff — read the log
before reporting a red check as a real one, and re-run the job if it died in setup.

**Pull GitHub's own review state, don't infer it.** `reviewDecision` and `reviewRequests`
say whether a PR is `APPROVED`, `CHANGES_REQUESTED`, awaiting review, and **who** it is
waiting on. A PR sitting on a named reviewer is a different row from one nobody has been
asked to look at, and only GitHub knows which. Say the reviewer's name when there is one.

Note that `main` moving does **not** move `reference/*` checkouts — they are git
submodules pinned to a fork branch, so the pin changes only when someone commits a new
gitlink. `update_reference_repos.sh --check` reports the pin and any drift without
touching anything; fetch explicitly if you need what a fork branch has since gained.

## Render

Four tables, one per tier, then the decisions. Per row: the task, its subtask PRs with
**raw URLs**, and state.

**Tier 0 is rendered even when empty** — print "No public PRs open." rather than dropping
the table. Its absence should mean "nothing is outstanding on someone else's repo", which
is information; a missing table just looks like it was forgotten. Tier 0 rows carry two
things the other tiers don't: **who is waiting** (a named reviewer, or "unreviewed"), and
whether the PR is a **draft**, since a draft still notifies maintainers but asserts
nothing. A tier-0 row that has sat unreviewed is worth saying so about explicitly.

Include upstream PRs *this project* owns, not every PR on those repos — a lab colleague's
unrelated PR is not our tier 0. When in doubt, a PR opened by or on behalf of this project
counts.

**"Done" is a task-level status only.** A task (the row itself) is ✅ done / 🔄 running /
⏸️ not started / 👤 waiting on Josh — done means every subtask PR in it has merged.
**A PR or stack is never "done"** — once merged it is dropped from the render (see
below), not marked done. A PR/stack still open is exactly one of, each paired with its
emoji (emoji **and** the text label together, never the emoji alone — it's a scan aid,
not a replacement):
- 👤 **waiting on human** — needs Josh's review or a decision only he can make
- 👀 **ready for review** — checks green, nothing blocking, just needs eyes
- 🔄 **in progress / running** — an agent is actively working it right now
- 🚧 **blocked on `<x>`** — name the specific blocker (a failing check, a dependency PR,
  a decision pending elsewhere), never a bare "blocked"

**Drop merged PRs/stacks once already reported merged.** If an earlier render this
session already said a PR merged, don't restate it — leave it out entirely rather than
re-listing it as done. Only bring it back if something changed (e.g. a rebase or
`git log` check reveals it didn't actually land where reported, the way #189 turned out
to be merged into a stale branch rather than `main`).

End with **"Waiting on you"** — every open decision, one line each, each naming the
specific thing to decide rather than "review X". This section is the point of the
tracker; if it is empty, say so explicitly.

**Running work belongs in its own workstream row, not only in a footer.** A row whose
subtask is an agent mid-flight is 🔄 **in progress** and must say *what specifically* is
being worked — "measuring the tossing goal-region bounds", not "an agent is running". A
reader scanning a workstream needs to see that something is already moving on it, or they
will ask for it again. Name the artifact it will produce when there is one (a PR, a
figure, a served page).

Then list every running agent in one line at the end as well, as a roll-call. Never
predict what a running agent will find.

## Rules that apply to every render

- **Raw URLs**, written as `thing (https://...)` or a markdown link — both render as
  visible text now that `FORCE_HYPERLINK=0` is set. Never a bare `#89` with no link:
  three repos use overlapping PR numbers, so a bare number is ambiguous.
- **Counts as `x/y`**, never a bare percentage. Write "null result" in full.
- Say what is *actually* true of CI at the moment of rendering. Do not wait for it.
- Keep it scannable. The tracker is read at a glance, not studied.

## Standing context

- `main` for this repo; the kinder-baselines Tossing3D port is a stack of draft PRs
  against that repo's `main`; kindergarden PRs are usually one-file fixes.
- Josh is in the PRPL lab and owns both upstreams, so upstream merge latency is a
  scheduling detail, not a risk — do not weight it when recommending. **The exception is
  a PR sitting with a named lab reviewer** — that is someone else's time, not Josh's, so
  it is genuinely out of his hands and belongs under "waiting on <name>", never under
  "waiting on you".
- Anything touching a public repo (opening, commenting, pushing) needs his permission
  first; the tracker should surface those as decisions rather than actions taken.
  **Marking a PR ready is Josh's alone** — agent PRs stay drafts, so "ready for review"
  is never a state the tracker puts one into, only one it reports.
