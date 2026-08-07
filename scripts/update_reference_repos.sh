#!/usr/bin/env bash
#
# Sync `reference/` -- the third-party checkouts this project reads but never builds
# on -- with the commits this repository pins, and report anything that has drifted.
# Idempotent: run it as often as you like.
#
#     scripts/update_reference_repos.sh
#     scripts/update_reference_repos.sh --check          # report only, clone nothing
#     scripts/update_reference_repos.sh --repo-root /somewhere/else
#     scripts/update_reference_repos.sh --no-filter      # full-blob clones
#
# `reference/` used to be gitignored, so there was no way to pin these checkouts
# through this repo's own history and this script was the substitute: it fast-forwarded
# each checkout to its own remote's default branch. **They are git submodules now**, so
# `.gitmodules` plus a gitlink per path *is* the pin, and this script's job changed with
# it -- from "bring everything to the tip of its default branch" to "make the checkouts
# match what is recorded, and say so when they do not".
#
# Two of the old contract's three rules went with the premise. A submodule tracks a
# *commit*, so there is no per-repo default branch to resolve (`predicators` being on
# `master` no longer matters to anything here) and there is nothing to fast-forward.
# They were deleted rather than ported.
#
# The rule that survives is the one that mattered, and a submodule makes it *more*
# important, not less:
#
#   **Local work is never clobbered.** A dirty tree or a checkout sitting on some other
#   commit means someone is mid-investigation in there. Those are reported with a
#   reason and left exactly as they are. `git submodule update` would silently detach
#   such a checkout back onto the recorded gitlink, so this script only ever
#   initialises a submodule that is **not populated yet**; anything already populated is
#   read, never written. It never stashes, resets, force-checks-out or discards.
#
# `git worktree add` does not populate submodules, which is deliberate leverage: a
# worktree starts with empty `reference/` directories and stays that way unless someone
# runs this script *inside it*. KINDER-backed tests gate on `importlib.util.find_spec`,
# so a worktree without them skips those tests rather than failing -- the same thing CI
# does. Most worktrees should therefore never run this at all; `--check` answers "am I
# in sync?" without paying kindergarden's 1.08 GiB pack to find out.
#
# Exit codes, so this is usable as a pre-flight check:
#   0  every submodule already current, or freshly initialised
#   1  at least one submodule FAILED (not a git repo, clone error, merge conflict)
#   2  nothing failed, but at least one was skipped or has drifted from its pin

set -uo pipefail

EXIT_OK=0
EXIT_FAILED=1
EXIT_SKIPPED=2

# --- argument parsing ------------------------------------------------------
# Resolve this script's directory even through a symlink, then take the repo root
# as its parent. Deliberately not `git rev-parse`: $PWD is wrong the moment a
# caller cd's, and the wrapper should still work from an archive.
script_source="${BASH_SOURCE[0]}"
while [ -L "$script_source" ]; do
    script_dir="$(cd -P "$(dirname "$script_source")" && pwd)"
    script_source="$(readlink "$script_source")"
    [[ "$script_source" != /* ]] && script_source="$script_dir/$script_source"
done
REPO_ROOT="$(cd -P "$(dirname "$script_source")/.." && pwd)"

# Submodules belong to the checkout you are standing in, so -- unlike the old script,
# which redirected every worktree at the main checkout's one shared `reference/` --
# this resolves no further. A worktree that wants KINDER populates its own.
CHECK_ONLY=false

# kindergarden's pack is 1.08 GiB. A blobless clone keeps full history for `git log`
# and `git blame` and fetches blobs lazily, which is the difference between seconds and
# several minutes. Applied to all three; `--no-filter` opts out.
FILTER_ARG="--filter=blob:none"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --no-filter)
            FILTER_ARG=""
            shift
            ;;
        -h|--help)
            sed -n '2,45p' "$script_source" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "update_reference_repos.sh: unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "update_reference_repos.sh: FAILED -- not a git repository: $REPO_ROOT" >&2
    exit "$EXIT_FAILED"
fi

# --- helpers ---------------------------------------------------------------

initialised=0
current=0
drifted=0
skipped=0
failed=0
total=0

note() {
    echo "$1"
}

short() {
    printf '%s' "${1:0:7}"
}

# The commit this repository records for a path. Read from the index rather than HEAD so
# it agrees with what `git submodule status` compares against.
recorded_sha() {
    git -C "$REPO_ROOT" rev-parse ":$1" 2>/dev/null || echo "unknown"
}

# --- main loop -------------------------------------------------------------
# `git submodule status` prints one line per submodule:
#
#     " <sha> <path> (<describe>)"   in sync with the recorded gitlink
#     "-<sha> <path>"                not initialised (no working tree)
#     "+<sha> <path> (<describe>)"   checked out at some *other* commit
#     "U<sha> <path>"                unresolved merge conflict
#
# The prefix is the whole classification, which is why this parses it rather than
# re-deriving the comparison by hand.
status_output="$(git -C "$REPO_ROOT" submodule status 2>/dev/null)"

while IFS= read -r line; do
    [ -z "$line" ] && continue

    prefix="${line:0:1}"
    rest="${line:1}"
    sha="${rest%% *}"
    path="${rest#* }"
    path="${path%% (*}"

    total=$((total + 1))
    sub_dir="$REPO_ROOT/$path"

    case "$prefix" in
        U)
            failed=$((failed + 1))
            note "  FAILED           $path: unresolved merge conflict on the gitlink"
            ;;

        -)
            # Not populated. This is the fresh-clone and fresh-worktree case: the
            # gitlink is recorded but the directory is empty.
            if [ "$CHECK_ONLY" = true ]; then
                skipped=$((skipped + 1))
                note "  not initialised  $path (pinned $(short "$sha")); --check made no changes"
                continue
            fi
            # shellcheck disable=SC2086 -- FILTER_ARG is one flag or the empty string.
            if update_log="$(git -C "$REPO_ROOT" submodule update --init $FILTER_ARG -- "$path" 2>&1)"; then
                initialised=$((initialised + 1))
                note "  initialised      $path @ $(short "$(recorded_sha "$path")")"
            else
                failed=$((failed + 1))
                note "  FAILED           $path: could not initialise"
                echo "$update_log" | tail -3 >&2
            fi
            ;;

        +)
            # Populated, but sitting on a different commit than the one recorded. That
            # commit is somebody's work -- report both ends and move on.
            drifted=$((drifted + 1))
            note "  DRIFTED          $path: checked out $(short "$sha"), pinned $(short "$(recorded_sha "$path")")"
            ;;

        *)
            # At the recorded commit. Dirtiness is a separate axis: `git submodule
            # status` compares commits only, so an uncommitted edit shows up as in-sync
            # here and has to be asked for directly.
            if [ -n "$(git -C "$sub_dir" status --porcelain 2>/dev/null)" ]; then
                skipped=$((skipped + 1))
                note "  skipped          $path: working tree is dirty (uncommitted local changes)"
                continue
            fi
            branch="$(git -C "$sub_dir" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
            if [ -n "$branch" ]; then
                current=$((current + 1))
                note "  already current  $path @ $(short "$sha") (on branch '$branch')"
            else
                current=$((current + 1))
                note "  already current  $path @ $(short "$sha")"
            fi
            ;;
    esac
done <<< "$status_output"

# --- summary ---------------------------------------------------------------
ok=$((initialised + current))
echo
echo "reference/ submodule summary ($REPO_ROOT)"
echo "  ok       $ok/$total  (initialised $initialised, already current $current)"
echo "  drifted  $drifted/$total"
echo "  skipped  $skipped/$total"
echo "  failed   $failed/$total"

if [ "$failed" -gt 0 ]; then
    exit "$EXIT_FAILED"
fi
if [ "$((drifted + skipped))" -gt 0 ]; then
    exit "$EXIT_SKIPPED"
fi
exit "$EXIT_OK"
