from pathlib import Path

from hitl_pmp.methods.pure_agent.types import AuthoringTranscript


class TranscriptStore:
    """Reads and writes the run artifact record-then-replay turns on. A static-method
    container, never instantiated, same as every other business-logic class in this
    project.

    **A directory, not a single file**, and the redundancy is deliberate.
    `transcript.json` is the machine-readable record and the only thing `read` consumes;
    beside it, each round's `policy.py` is also written out verbatim as
    `round_<n>_policy.py`. That second copy exists to be *read by a human*, because the
    single most informative output of this whole baseline is what the agent actually
    wrote, and nobody reads it out of a JSON string with `\\n` escapes in it. It is
    never parsed back -- `read` uses `transcript.json` alone -- so the two cannot
    disagree about what will be replayed.

    Written where the run's other artifacts go (`--output-dir`), so a replay and the
    `stats.json` it produced sit together."""

    TRANSCRIPT_NAME = "transcript.json"

    @staticmethod
    def write(*, transcript: AuthoringTranscript, directory: Path) -> Path:
        """Write the transcript and its per-round sources; return the JSON's path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TranscriptStore.TRANSCRIPT_NAME
        path.write_text(transcript.model_dump_json(indent=2))
        for round_ in transcript.rounds:
            if round_.policy_source is None:
                continue
            (directory / f"round_{round_.round_index:03d}_policy.py").write_text(
                round_.policy_source
            )
        return path

    @staticmethod
    def read(*, path: Path) -> AuthoringTranscript:
        """Load a transcript from its JSON, accepting either the file itself or the
        directory holding it -- a caller who has the run's `--output-dir` to hand should
        not have to remember the filename."""
        if path.is_dir():
            path = path / TranscriptStore.TRANSCRIPT_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no pure-agent transcript at {path}. A replay needs one authored first; "
                "see methods/pure_agent/README.md."
            )
        return AuthoringTranscript.model_validate_json(path.read_text())
