"""`scripts/check_doc_links.sh` is the CI guard that stops a committed experiment log
from reintroducing a URL that points back at *this* repository.

`main` allows squash-merge only, so merging a PR mints a new commit and **orphans every
SHA that PR pinned**. A `raw.githubusercontent.com` URL in a committed file therefore
dies at merge -- and dies *silently*, because GitHub keeps orphaned commits reachable
until it garbage-collects, so the dead URL still returns `200` with byte-correct content
for months. Measured on 2026-08-07, 3/5 distinct pinned SHAs in merged
`docs/experiment-logs/` were already orphaned; PR #173 converted them to relative links
and wrote the rule into `CLAUDE.md`. Nothing enforced it. This does.

The tests that matter here are the ones asserting the guard **fails**. `grep` exits `1`
on *no* match, so the guard's success path is its `else` branch, and getting that
inverted yields a check that passes unconditionally -- the same silent pass the guard
exists to prevent. A test that only proves "the current tree passes" would pass against
`exit 0` as the entire implementation.

Two boundaries are pinned deliberately, because both are scope decisions rather than
oversights:

* A URL into **another** repo (`kindergarden`, `predicators`) passes. Those have no
  relative equivalent, so banning them would be wrong and would push people toward
  workarounds.
* A file **outside** `docs/experiment-logs/` passes. `CLAUDE.md` has to keep saying
  `raw.githubusercontent.com` in prose -- it is where the rule is written down -- so a
  repo-wide guard would fail on the file that defines it.

Every test builds a synthetic tree in a tmpdir and points the script at it with
`--repo-root`. Nothing here touches the real `docs/` except the one test that
deliberately asserts the committed tree is clean.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_doc_links.sh"

# Exit codes the script contracts on. Both non-zero values fail CI; they are kept
# distinct so "the check found something" is never confused with "the check could not
# run", which is how a guard ends up passing vacuously.
EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2

# The directory the guard scans, relative to the repo root.
LOG_DIR = "docs/experiment-logs"

# One banned URL of each shape, plus the near-misses that must stay allowed.
RAW_SELF = (
    "https://raw.githubusercontent.com/joshnroy/"
    "human-in-the-loop-practice-makes-perfect/"
    "b5dd17d0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6/docs/experiment-logs/fig.png"
)
BLOB_SELF_SHA = (
    "https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/blob/"
    "b5dd17d0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6/docs/experiment-logs/fig.png"
)
TREE_SELF_SHA = (
    "https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/tree/"
    "b5dd17d/docs/experiment-logs"
)
RAW_OTHER_REPO = (
    "https://raw.githubusercontent.com/joshnroy/kindergarden/"
    "4113237ab0000000000000000000000000000000/src/kinder/__init__.py"
)
RAW_PREDICATORS = (
    "https://raw.githubusercontent.com/Learning-and-Intelligent-Systems/"
    "predicators/5bd3f5b/predicators/envs/grid_row.py"
)
PR_URL_SELF = "https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/173"


def _make_tree(*, tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a synthetic repo root containing `files`, keyed by repo-relative path.

    The scanned directory is always created, so a test that means "clean" is never
    silently testing "nothing to scan" instead.
    """
    root = tmp_path / "repo"
    (root / LOG_DIR).mkdir(parents=True)
    for relative_path, text in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return root


def _run(*, repo_root: Path, github_actions: bool = False) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    if github_actions:
        env["GITHUB_ACTIONS"] = "true"
    else:
        env.pop("GITHUB_ACTIONS", None)
    return subprocess.run(
        [str(SCRIPT), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_clean_tree_passes(*, tmp_path: Path) -> None:
    root = _make_tree(
        tmp_path=tmp_path,
        files={
            f"{LOG_DIR}/2026-08-07-log.md": (
                "# A log\n\n![a figure](2026-08-07-fig.png)\n\n"
                "[seed 3, stranded](2026-08-07-seed3.mp4)\n"
            )
        },
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_raw_self_repo_url_fails(*, tmp_path: Path) -> None:
    """The shape that actually died: a raw URL pinned to a SHA the squash orphans."""
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"![fig]({RAW_SELF})\n"}
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_VIOLATION, result.stdout + result.stderr
    assert f"{LOG_DIR}/2026-08-07-log.md:1" in result.stdout


def test_sha_pinned_blob_url_fails(*, tmp_path: Path) -> None:
    """A `blob/<sha>/` link orphans identically to a raw one, so it is banned too."""
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"[fig]({BLOB_SELF_SHA})\n"}
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_VIOLATION, result.stdout + result.stderr
    assert f"{LOG_DIR}/2026-08-07-log.md:1" in result.stdout


def test_sha_pinned_tree_url_fails(*, tmp_path: Path) -> None:
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"[dir]({TREE_SELF_SHA})\n"}
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_VIOLATION, result.stdout + result.stderr


def test_every_offending_line_is_reported(*, tmp_path: Path) -> None:
    """A guard that stops at the first hit makes the fix an iterative game with CI."""
    root = _make_tree(
        tmp_path=tmp_path,
        files={
            f"{LOG_DIR}/a.md": f"line one\n![fig]({RAW_SELF})\n",
            f"{LOG_DIR}/b.md": f"[dir]({TREE_SELF_SHA})\n",
        },
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_VIOLATION, result.stdout + result.stderr
    assert f"{LOG_DIR}/a.md:2" in result.stdout
    assert f"{LOG_DIR}/b.md:1" in result.stdout
    # Counts as x/y, never a bare percentage.
    assert "2/2" in result.stdout


def test_other_repo_urls_pass(*, tmp_path: Path) -> None:
    """Scope limit, not an oversight: a KINDER/predicators URL has no relative form."""
    root = _make_tree(
        tmp_path=tmp_path,
        files={
            f"{LOG_DIR}/2026-08-07-log.md": (
                f"[kinder]({RAW_OTHER_REPO})\n[predicators]({RAW_PREDICATORS})\n"
            )
        },
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_self_repo_pull_request_url_passes(*, tmp_path: Path) -> None:
    """A PR link carries no SHA and has no relative form, so it cannot orphan."""
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"See {PR_URL_SELF}.\n"}
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_files_outside_the_log_directory_pass(*, tmp_path: Path) -> None:
    """`CLAUDE.md` documents the rule, so it must keep naming the banned host."""
    root = _make_tree(
        tmp_path=tmp_path,
        files={
            "CLAUDE.md": f"Bodies use {RAW_SELF} pinned to a full SHA.\n",
            "docs/some-design-doc.md": f"![fig]({RAW_SELF})\n",
        },
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_non_markdown_files_in_the_log_directory_pass(*, tmp_path: Path) -> None:
    """Documented limit: committed run artefacts (`*.json`) are data, not prose."""
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/traces.json": f'{{"url": "{RAW_SELF}"}}\n'}
    )
    result = _run(repo_root=root)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_missing_scan_directory_is_an_error_not_a_pass(*, tmp_path: Path) -> None:
    """A guard with nothing to scan must say so rather than report success."""
    root = tmp_path / "empty-repo"
    root.mkdir()
    result = _run(repo_root=root)
    assert result.returncode == EXIT_ERROR, result.stdout + result.stderr


def test_violation_emits_a_github_annotation_pointing_at_the_rule(*, tmp_path: Path) -> None:
    """The annotation is how someone reading a red CI log finds the fix."""
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"![fig]({RAW_SELF})\n"}
    )
    result = _run(repo_root=root, github_actions=True)
    assert result.returncode == EXIT_VIOLATION
    assert f"::error file={LOG_DIR}/2026-08-07-log.md,line=1::" in result.stdout
    assert "CLAUDE.md" in result.stdout


def test_no_annotation_outside_github_actions(*, tmp_path: Path) -> None:
    root = _make_tree(
        tmp_path=tmp_path, files={f"{LOG_DIR}/2026-08-07-log.md": f"![fig]({RAW_SELF})\n"}
    )
    result = _run(repo_root=root)
    assert "::error" not in result.stdout


def test_committed_tree_is_clean() -> None:
    """The guard's own subject. Run against the real repo, with no `--repo-root`."""
    result = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, cwd=os.sep)
    assert result.returncode == EXIT_OK, result.stdout + result.stderr


def test_ci_runs_the_guard_in_the_lint_job() -> None:
    """A script nothing invokes is not a guard. Pin that CI actually calls it."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    lint_job = workflow.split("\n  lint:\n", 1)[1].split("\n  typecheck:\n", 1)[0]
    assert "scripts/check_doc_links.sh" in lint_job
