# Tossing3D launch provenance backfill

Backfilled from original Della manifests and saved run snapshots on 2026-10-26.
This adds provenance without changing published results. Original flags are kept
verbatim; resolved defaults come from the historical snapshots, not today's parser.

| Experiment | Manifest rows | Snapshots verified | Source commits |
| --- | ---: | ---: | --- |
| t3-dp-ees-20261025 | 24 | 0/24 | `a5ea1863a73d4f502035b333c9f08ee10e4a622f` |
| t3-ees-human-scored-relaunch-20261025 | 12 | 0/12 | `a5ea1863a73d4f502035b333c9f08ee10e4a622f` |
| t3-final-20261023 | 36 | 36/36 | `1f209df652576d1f6f52afd30b2def4ef840495a` |
| t3-relaunch-20261023 | 36 | 36/36 | `8c2d664a4bbd8521cdd7823cc30943b450624b5d` |
| t3-relaunch-curtis-20261023 | 12 | 12/12 | `78fb1a75a6b7b1a60f0309ca485bd16b30f4119a` |
| t3-relaunch2-20261024 | 36 | 36/36 | `2cd74021bd6e37b5fe87ef95cdcb6ca2412e4189` |
| t3-pour-pilot-20261025 | 6 | 6/6 | `0fa7c204209ad2f32102cd5b40b37cd785398c6c` |

The first relaunch used a separate Curtis launch tag; its records belong to the
same comparison. DP/EES records describe the original and replacement submissions,
including runs later cancelled; these are not new outcome or completion claims.

## Exact commands and resolved settings

[runs.json](2026-10-26-tossing3d-launch-provenance/runs.json) contains every exact argv,
copyable shell-quoted CLI command, source commit, original manifest path, task ID,
and output path. Verified records additionally include the saved resolved flags,
dependency SHAs, dirty flags, interpreter and numerical-library versions.
The `sources` list hashes each byte-preserved [manifest and Slurm script](2026-10-26-tossing3d-launch-provenance/manifests).
Slurm scripts preserve resource requests and launch environment.
A missing runtime snapshot is labelled, rather than reconstructed from present defaults.

## Changes between experiments

To compare source, use the full `source_commit` values with `git diff OLD NEW -- src scripts configs`.
Compare `argv` for explicit flag changes and `snapshot.args` for changed defaults.
Compare KINDER and Fast Downward commits separately; a hitl-pmp SHA alone does not
identify the simulator or motion-planning code. A dirty flag means the SHA alone
cannot reconstruct the executed tree.

## Earlier local mass probes

The 3.8 kg and 1 kg local validation notes predate the frozen Della pilot. They
identify KINDER dependencies, but do not retain an exact source tree or original
launch argv. Their source commit and exact invocation are therefore **unknown**.
They must not be tagged with the current HEAD or treated as reproductions of it.
The scripts and output files cited in those notes remain the available evidence.

## Required fields for subsequent log entries

Every experiment entry must name its run ID, full launch commit, dependency
commits and dirty status, exact argv/command, resolved defaults, seed/arm matrix,
budget/evaluation/reset settings, cost-file contents or immutable hash, output
location, and the manifest/Slurm provenance. Record analysis-code commits separately
from run-code commits; record every resume revision rather than overwriting history.
Planned manifests must be labelled planned until the worker verifies the revision.

The Tossing3D worker now writes `launch.json` before dispatch, linking the exact
manifest row and hash to `physics.json`. The physics record captures the verified
code/dependencies, resolved flags, exact CLI command and cost-file contents/hash.
`config_snapshot.json` remains the per-run environment record.
