#!/usr/bin/env bash
#
# Keep `reference/` -- the third-party checkouts this project reads but never
# builds on -- current. Idempotent: run it as often as you like.
#
#     scripts/update_reference_repos.sh
#     scripts/update_reference_repos.sh --reference-dir /somewhere/else
#     scripts/update_reference_repos.sh --manifest my-repos.tsv
#
# `reference/` is gitignored and never committed (1.9 GB, and an embedded git
# repo inside this one confuses `git add -A`), so there is no way to pin these
# checkouts through this repo's own history. This script is the substitute: one
# command that brings every reference checkout to its upstream default branch.
#
# Three rules it will not break, because a reference checkout is somewhere people
# actually work:
#
#   1. **The default branch is per-repo.** `predicators` is on `master`, the
#      KINDER repos on `main`. The branch is resolved from the remote's own HEAD,
#      never assumed -- assuming `main` would make `predicators` look permanently
#      skipped-by-choice rather than broken.
#   2. **Local work is never clobbered.** A dirty tree, a non-default branch or a
#      detached HEAD means someone is mid-investigation in there. Those are
#      skipped with a reason. The script never stashes, resets, force-checks-out
#      or otherwise discards anything.
#   3. **`--ff-only`.** A merge commit or a rebase in a reference checkout is
#      never what anyone wanted.
#
# Exit codes, so this is usable as a pre-flight check:
#   0  every repo cloned, fast-forwarded, or already current
#   1  at least one repo FAILED (clone/fetch/merge error)
#   2  no failures, but at least one repo was SKIPPED to protect local work

set -uo pipefail

EXIT_OK=0
EXIT_FAILED=1
EXIT_SKIPPED=2

# --- the default manifest --------------------------------------------------
# Format: name<TAB>url<TAB>extra-clone-flags
# kindergarden's pack is 1.08 GiB, so it is cloned blobless: full history for
# `git log`/`git blame`, blobs fetched lazily on demand.
read -r -d '' DEFAULT_MANIFEST <<'EOF'
kindergarden	https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git	--filter=blob:none
kinder-baselines	https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines.git
predicators	https://github.com/Learning-and-Intelligent-Systems/predicators.git
EOF

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

# `reference/` belongs to the *main* checkout, not to each worktree: it is 1.9 GB
# of third-party source that nothing here builds on, and cloning it per worktree
# would be absurd. Resolve it through the git common dir (the same trick
# with_env.sh uses for FD_EXEC_PATH) so a worktree updates the one real
# `reference/` rather than growing its own empty copy.
REFERENCE_DIR="$REPO_ROOT/reference"
git_common_dir="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
if [ -n "$git_common_dir" ]; then
    REFERENCE_DIR="$(dirname "$git_common_dir")/reference"
fi
MANIFEST_FILE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --reference-dir)
            REFERENCE_DIR="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST_FILE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,32p' "$script_source" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "update_reference_repos.sh: unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

if [ -n "$MANIFEST_FILE" ]; then
    if [ ! -f "$MANIFEST_FILE" ]; then
        echo "update_reference_repos.sh: no such manifest: $MANIFEST_FILE" >&2
        exit 1
    fi
    manifest_body="$(cat "$MANIFEST_FILE")"
else
    manifest_body="$DEFAULT_MANIFEST"
fi

mkdir -p "$REFERENCE_DIR"

# --- helpers ---------------------------------------------------------------

# The remote's own idea of its default branch. Prefer the cached
# refs/remotes/origin/HEAD; if it is missing (older clones never set it), ask the
# remote once and cache it. Printing nothing means "could not determine".
resolve_default_branch() {
    local repo_dir="$1" head_ref
    head_ref="$(git -C "$repo_dir" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)"
    if [ -z "$head_ref" ]; then
        git -C "$repo_dir" remote set-head origin --auto >/dev/null 2>&1
        head_ref="$(git -C "$repo_dir" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)"
    fi
    # "origin/main" -> "main"
    echo "${head_ref#origin/}"
}

cloned=0
updated=0
current=0
skipped=0
failed=0
total=0
declare -a summary_lines=()

note() {
    summary_lines+=("$1")
    echo "$1"
}

# --- main loop -------------------------------------------------------------
while IFS=$'\t' read -r name url extra_flags; do
    # Tolerate blank lines and comments in a hand-written manifest.
    [ -z "${name// /}" ] && continue
    case "$name" in \#*) continue ;; esac

    total=$((total + 1))
    repo_dir="$REFERENCE_DIR/$name"

    # ---- clone, if it is not there yet ----
    if [ ! -d "$repo_dir/.git" ]; then
        # shellcheck disable=SC2086 -- extra_flags is deliberately word-split.
        if git clone $extra_flags "$url" "$repo_dir" >/dev/null 2>&1; then
            sha="$(git -C "$repo_dir" rev-parse --short HEAD)"
            branch="$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD)"
            cloned=$((cloned + 1))
            note "  cloned      $name -> $branch @ $sha"
        else
            failed=$((failed + 1))
            note "  FAILED      $name: clone from $url failed"
        fi
        continue
    fi

    # ---- refuse to touch anything that looks like live work ----
    if [ -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" ]; then
        skipped=$((skipped + 1))
        note "  skipped     $name: working tree is dirty (uncommitted local changes)"
        continue
    fi

    if ! branch="$(git -C "$repo_dir" symbolic-ref --quiet --short HEAD 2>/dev/null)"; then
        sha="$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo '?')"
        skipped=$((skipped + 1))
        note "  skipped     $name: detached HEAD at $sha"
        continue
    fi

    if ! git -C "$repo_dir" fetch --quiet origin 2>/dev/null; then
        failed=$((failed + 1))
        note "  FAILED      $name: fetch from origin failed"
        continue
    fi

    default_branch="$(resolve_default_branch "$repo_dir")"
    if [ -z "$default_branch" ]; then
        failed=$((failed + 1))
        note "  FAILED      $name: could not resolve the remote's default branch"
        continue
    fi

    if [ "$branch" != "$default_branch" ]; then
        skipped=$((skipped + 1))
        note "  skipped     $name: on '$branch', not the default branch '$default_branch'"
        continue
    fi

    # ---- fast-forward only ----
    before="$(git -C "$repo_dir" rev-parse HEAD)"
    target="$(git -C "$repo_dir" rev-parse "origin/$default_branch" 2>/dev/null)"
    if [ "$before" = "$target" ]; then
        current=$((current + 1))
        note "  already current  $name ($default_branch @ $(git -C "$repo_dir" rev-parse --short HEAD))"
        continue
    fi

    if git -C "$repo_dir" merge --ff-only --quiet "origin/$default_branch" >/dev/null 2>&1; then
        updated=$((updated + 1))
        note "  updated     $name ($default_branch): $(git -C "$repo_dir" rev-parse --short "$before") -> $(git -C "$repo_dir" rev-parse --short HEAD)"
    else
        failed=$((failed + 1))
        note "  FAILED      $name: cannot fast-forward $branch to origin/$default_branch (diverged)"
    fi
done <<< "$manifest_body"

# --- summary ---------------------------------------------------------------
ok=$((cloned + updated + current))
echo
echo "reference/ update summary ($REFERENCE_DIR)"
echo "  ok       $ok/$total  (cloned $cloned, updated $updated, already current $current)"
echo "  skipped  $skipped/$total"
echo "  failed   $failed/$total"

if [ "$failed" -gt 0 ]; then
    exit "$EXIT_FAILED"
fi
if [ "$skipped" -gt 0 ]; then
    exit "$EXIT_SKIPPED"
fi
exit "$EXIT_OK"
