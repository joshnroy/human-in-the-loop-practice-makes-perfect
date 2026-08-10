#!/usr/bin/env bash
#
# Fail if a link in a committed experiment log is one that will silently die.
#
#     scripts/check_doc_links.sh
#     scripts/check_doc_links.sh --repo-root /somewhere/else   # for the tests
#
# Runs as a step of CI's `lint` job, and takes about 50 ms. Two checks, both reported in
# one run so that fixing a log is one pass rather than two CI rounds:
#
#   1. a link back at *this* repository by URL -- see WHY below
#   2. a relative link whose target does not exist -- see check 2's own comment
#
# WHY. `main` allows squash-merge only, so merging a PR mints a new commit and
# **orphans every SHA that PR pinned**. A `raw.githubusercontent.com` URL sitting in a
# committed file therefore dies the moment the PR lands -- and dies *silently*, because
# GitHub keeps orphaned commits reachable until it garbage-collects, so the dead URL
# still returns `200` with byte-correct content for months afterwards. Measured
# 2026-08-07: 3/5 distinct pinned SHAs in merged `docs/experiment-logs/` were already
# orphaned. PR #173 converted them to repo-relative links -- which resolve against
# whatever ref the reader is on, so they cannot orphan -- and wrote the rule into
# `CLAUDE.md`. Nothing enforced it, so a future log could reintroduce one and pass every
# check. That gap is what this closes.
#
# WHAT IS BANNED, and why the scope is narrow on both axes:
#
#   * **Only self-referencing URLs.** A URL into `kindergarden` or `predicators` has no
#     relative equivalent, so banning every raw URL would be wrong and would push people
#     toward workarounds. Only a link into this repo has a relative alternative, so only
#     that is banned. The owner is matched as `[^/]+` rather than a literal `joshnroy`:
#     a fork or a transfer to the lab org orphans exactly the same way, and no other
#     repository shares this one's name.
#   * **Only `docs/experiment-logs/*.md`.** `CLAUDE.md` has to keep saying
#     `raw.githubusercontent.com` in prose -- it is where the rule is documented -- so a
#     repo-wide guard would fail on the file that defines the rule. Committed run
#     artefacts (`*.json`) under the log directory are data, not prose, and are skipped.
#
# A `blob/<sha>/` or `tree/<sha>/` link is banned alongside the raw one because it
# orphans identically; a PR link (`/pull/173`) and a branch-name `blob/main/` link are
# not, since neither pins a commit.
#
# THE TRAP. `grep` exits `1` on *no* match, so the success path here is the `else`
# branch, and an inverted test yields a guard that passes unconditionally. Worse, `grep`
# exits `2` on an error of its own (a missing directory, a bad pattern), which a naive
# `if grep ...; then fail; fi` also treats as success. Both cases are handled explicitly
# below, and `tests/scripts/test_check_doc_links.py` asserts the failing direction.
#
# Exit codes:
#   0  both checks clean
#   1  at least one banned URL, or at least one dangling relative target
#   2  a check could not run (missing scan directory, grep error) -- deliberately not
#      0, because a guard with nothing to scan reporting success is the failure mode
#      this whole script exists to prevent

set -uo pipefail

EXIT_OK=0
EXIT_VIOLATION=1
EXIT_ERROR=2

# The directory whose Markdown is scanned, relative to the repo root.
SCAN_DIR="docs/experiment-logs"

# Where a reader of a red CI log should go to find out what to write instead.
RULE_REFERENCE='CLAUDE.md, "Where a figure or video lives"'

# A raw URL into this repo, pinned to anything at all -- a branch name dies when the
# branch is deleted, a SHA dies at the squash.
PATTERN_RAW='https://raw\.githubusercontent\.com/[^/]+/human-in-the-loop-practice-makes-perfect/'
# A commit-pinned blob/tree link into this repo. Seven hex characters is GitHub's own
# minimum abbreviation.
PATTERN_BLOB='https://github\.com/[^/]+/human-in-the-loop-practice-makes-perfect/(blob|tree)/[0-9a-f]{7,40}/'

# --- argument parsing ------------------------------------------------------
# Resolve this script's directory even through a symlink, then take the repo root as
# its parent. Deliberately not `git rev-parse`: $PWD is wrong the moment a caller cd's,
# and this should still work from an archive.
script_source="${BASH_SOURCE[0]}"
while [ -L "$script_source" ]; do
    script_dir="$(cd -P "$(dirname "$script_source")" && pwd)"
    script_source="$(readlink "$script_source")"
    [[ "$script_source" != /* ]] && script_source="$script_dir/$script_source"
done
REPO_ROOT="$(cd -P "$(dirname "$script_source")/.." && pwd)"

usage() {
    echo "usage: $(basename "$0") [--repo-root PATH]"
    echo
    echo "Fail if a committed experiment log links back at this repository by URL, or"
    echo "carries a relative link whose target does not exist."
    echo "Exit 0 clean, 1 violation found, 2 a check could not run."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --repo-root)
            if [ $# -lt 2 ]; then
                echo "error: --repo-root needs a path" >&2
                exit "$EXIT_ERROR"
            fi
            REPO_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit "$EXIT_OK"
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit "$EXIT_ERROR"
            ;;
    esac
done

if ! cd "$REPO_ROOT" 2>/dev/null; then
    echo "error: repo root does not exist: $REPO_ROOT" >&2
    exit "$EXIT_ERROR"
fi

if [ ! -d "$SCAN_DIR" ]; then
    # Not a pass. If the directory this guard exists to watch is gone, the guard is
    # vacuous, and a vacuous guard must say so rather than report success.
    echo "error: nothing to scan -- $SCAN_DIR/ does not exist under $REPO_ROOT" >&2
    exit "$EXIT_ERROR"
fi

annotate() {
    # A GitHub annotation, so a red CI log links straight to the offending line and
    # names the rule. Suppressed locally, where it is noise duplicating the human line.
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        echo "::error file=$1,line=$2::$3"
    fi
}

total_files="$(find "$SCAN_DIR" -type f -name '*.md' | wc -l | tr -d ' ')"
violations=0

# --- check 1: a URL pointing back at this repository -----------------------
# Paths are printed relative to the repo root, which is what a GitHub annotation needs
# (it resolves `file=` against the workspace) and what a human can paste into an editor.
url_matches="$(grep -rEn --include='*.md' -e "$PATTERN_RAW" -e "$PATTERN_BLOB" -- "$SCAN_DIR")"
grep_status=$?

if [ "$grep_status" -gt 1 ]; then
    echo "error: grep failed (exit $grep_status) while scanning $SCAN_DIR/" >&2
    exit "$EXIT_ERROR"
fi

if [ "$grep_status" -eq 1 ]; then
    echo "check_doc_links: 0/$total_files file(s) under $SCAN_DIR/ contain a URL into this repo."
else
    # Report every offending line, so fixing this is one pass rather than an iterative
    # game against CI.
    offending_lines="$(printf '%s\n' "$url_matches" | wc -l | tr -d ' ')"
    offending_files="$(printf '%s\n' "$url_matches" | cut -d: -f1 | sort -u | wc -l | tr -d ' ')"
    violations=$((violations + offending_lines))

    echo "check_doc_links: $offending_files/$total_files file(s) under $SCAN_DIR/ contain a URL into this repo ($offending_lines offending line(s))."
    echo
    printf '%s\n' "$url_matches"
    echo
    echo "A committed file must link to this repo's own content by *repo-relative* path, not"
    echo "by URL: squash-merge orphans every pinned SHA, and the dead URL still returns 200"
    echo "for months, so the breakage is silent. Write ![alt](2026-08-07-foo.png) instead."
    echo "The full rule, including why a PR *body* is the exception: $RULE_REFERENCE."
    echo

    while IFS= read -r match; do
        file="${match%%:*}"
        rest="${match#*:}"
        annotate "$file" "${rest%%:*}" "URL into this repository in a committed file. Use a repo-relative link -- squash-merge orphans every pinned SHA, silently. See $RULE_REFERENCE."
    done <<< "$url_matches"
fi

# --- check 2: a relative link whose target does not exist ------------------
# The other half of the same rule. Making every figure reference relative moved the
# failure mode rather than removing it: a relative link cannot orphan, but it can point
# at a filename that was never committed or was later renamed, and GitHub renders that
# as a dead link with no warning anywhere.
#
# Deliberately not a Markdown parser. The awk pass below drops fenced code blocks and
# inline code spans first -- `CLAUDE.md` writes the rule as `![alt](2026-08-07-foo.png)`
# inside backticks, so a log quoting it would otherwise be flagged for a figure it never
# claimed to have. Measured on the tree at the time of writing: 97 relative targets, 0
# of them inside a fence or a span, so this is prevention rather than a fix. Known
# limits, all currently unused here: reference-style (`[x]: path`) links, angle-bracket
# targets (`](<a b.png>)`), and percent-encoded paths are not resolved.
extract_targets() {
    awk '
        /^[[:space:]]*```/ { fence = !fence; next }
        fence { next }
        {
            line = $0
            gsub(/`[^`]*`/, "", line)
            while (match(line, /\]\([^()]*\)/)) {
                print NR "\t" substr(line, RSTART + 2, RLENGTH - 3)
                line = substr(line, RSTART + RLENGTH)
            }
        }
    ' "$1"
}

dangling=""
targets_checked=0

while IFS= read -r file; do
    while IFS=$'\t' read -r line_number target; do
        # A title (`](fig.png "caption")`) is not part of the path, and neither is a
        # fragment (`](other.md#results)`) -- which also makes a bare `#anchor` empty.
        target="${target%% *}"
        target="${target%%#*}"
        case "$target" in
            "") continue ;;
            # A scheme-qualified or protocol-relative URL has no local target. Check 1
            # owns URLs; this one must not try to resolve them on disk.
            *://*|mailto:*|//*) continue ;;
        esac
        targets_checked=$((targets_checked + 1))
        # GitHub resolves a leading `/` against the repo root, everything else against
        # the linking file's own directory.
        case "$target" in
            /*) resolved="${target#/}" ;;
            *) resolved="$(dirname "$file")/$target" ;;
        esac
        if [ ! -e "$resolved" ]; then
            dangling+="$file:$line_number: $target"$'\n'
        fi
    done < <(extract_targets "$file")
done < <(find "$SCAN_DIR" -type f -name '*.md')

if [ -z "$dangling" ]; then
    echo "check_doc_links: 0/$targets_checked relative link target(s) under $SCAN_DIR/ are missing."
else
    dangling_lines="$(printf '%s' "$dangling" | wc -l | tr -d ' ')"
    violations=$((violations + dangling_lines))

    echo "check_doc_links: $dangling_lines/$targets_checked relative link target(s) under $SCAN_DIR/ are missing."
    echo
    printf '%s' "$dangling"
    echo
    echo "A relative link is only correct if it resolves. Check the filename against what"
    echo "is actually committed beside the log -- a renamed or never-committed figure"
    echo "renders as a dead link on GitHub with no warning anywhere."
    echo

    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        file="${entry%%:*}"
        rest="${entry#*:}"
        annotate "$file" "${rest%%:*}" "Relative link target does not exist: ${rest#*: }. Check it against what is committed beside the log. See $RULE_REFERENCE."
    done <<< "$dangling"
fi

if [ "$violations" -gt 0 ]; then
    exit "$EXIT_VIOLATION"
fi
exit "$EXIT_OK"
