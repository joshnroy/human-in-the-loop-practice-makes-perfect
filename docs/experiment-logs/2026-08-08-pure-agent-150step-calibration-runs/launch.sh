#!/bin/bash
set -u
WORKTREE=/home/josh/Documents/repos/research/human-in-the-loop-practice-makes-perfect/.claude/worktrees/agent-a4c98caaa4dbd7f36
SCRATCH=/tmp/claude-1000/-home-josh-Documents-repos-research-human-in-the-loop-practice-makes-perfect/2ad84fb0-74aa-46a3-b3e1-ecae6aa0f3ab/scratchpad
cd "$WORKTREE" || exit 1
exec scripts/with_env.sh python -m hitl_pmp.cli \
  --env tossingroom \
  --method pure-agent \
  --seed 0 \
  --num-cycles 1 \
  --max-steps-per-interaction 150 \
  --num-test-tasks 4 \
  --output-dir "$SCRATCH/calib/out" \
  --pure-agent-sandbox-dir "$SCRATCH/calib/sandbox" \
  --pure-agent-model opus \
  --pure-agent-max-total-cost-usd 25 \
  --pure-agent-max-cost-usd-per-query 2.0
