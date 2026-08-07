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
  1. **Correctness** — the environment, its skills, or the measurement pipeline are *wrong*, so anything measured on them is suspect. First regardless of size.
  2. **Methods and experiments** — the long-running scientific work.
  3. **Infra** — tooling, docs, ergonomics. **Exception:** infra that materially speeds up tier-2 work is promoted to tier 2.
- When several tasks are ready, **start the longest-running agent work first**, so expensive jobs run while cheap ones are discussed.

## Gather state

Three repos. Ours, plus two lab-owned upstreams.

```bash
cd <repo-root>
git fetch -q origin && git log --oneline -1 origin/main
gh pr list --state open --json number,title,baseRefName,mergeStateStatus \
  --jq '.[]|"#\(.number) base=\(.baseRefName) \(.mergeStateStatus)  \(.title)"'
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

Note that `main` moving does **not** move `reference/*` checkouts — they are git
submodules pinned to a fork branch, so the pin changes only when someone commits a new
gitlink. `update_reference_repos.sh --check` reports the pin and any drift without
touching anything; fetch explicitly if you need what a fork branch has since gained.

## Render

Three tables, one per tier, then the decisions. Per row: the task, its subtask PRs with
**raw URLs**, and state. Mark tasks ✅ done / 🔄 running / ⏸️ not started / 👤 waiting on Josh.

End with **"Waiting on you"** — every open decision, one line each, each naming the
specific thing to decide rather than "review X". This section is the point of the
tracker; if it is empty, say so explicitly.

If agents are running, list them in one line at the end. Never predict what a running
agent will find.

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
  scheduling detail, not a risk — do not weight it when recommending.
- Anything touching a public repo (opening, commenting, pushing) needs his permission
  first; the tracker should surface those as decisions rather than actions taken.
  **Marking a PR ready is Josh's alone** — agent PRs stay drafts, so "ready for review"
  is never a state the tracker puts one into, only one it reports.
