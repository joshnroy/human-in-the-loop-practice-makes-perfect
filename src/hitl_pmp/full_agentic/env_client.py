"""Practice-only client. Each connection reaches the SAME persistent world."""

import json
import socket
import uuid
from typing import Any


class PracticeEnvironment:
    def __init__(self) -> None:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        self.version = None
        self.last_request: dict[str, Any] | None = None
        self.last_response = self.observe()

    def request(self, operation: Any, **arguments: Any) -> Any:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        payload = dict(
            operation=operation,
            request_id=uuid.uuid4().hex,
            expected_version=self.version,
            **arguments,
        )
        self.last_request = payload
        return self.retry_last_request()

    def retry_last_request(self) -> Any:
        """Retry only this identical request after a lost connection."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(180)
            client.connect("/run/hitl-live/world.sock")
            client.sendall(json.dumps(self.last_request, allow_nan=False).encode() + b"\n")
            with client.makefile("rb") as stream:
                response = json.loads(stream.readline(4_000_000))
        if "error" in response:
            raise RuntimeError(response["error"])
        self.last_response = response["result"]
        self.version = self.last_response["world_version"]
        return self.last_response

    def observe(self) -> Any:
        return self.request("observe")

    def begin_trial(self) -> Any:
        """Snapshot current Python files; charge one robot attempt (max 1000 steps)."""
        return self.request("begin_trial")

    def step(self, action: Any) -> Any:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        return self.request("step", action=action)

    def end_trial(self) -> Any:
        return self.request("end_trial")

    def request_help(self, intervention_id: Any) -> Any:  # noqa: PLR0917 -- supplied RoboCode client/interface contract
        return self.request("request_help", intervention_id=intervention_id)

    def finish_adaptation(self) -> Any:
        return self.request("finish_adaptation")


def make_env() -> Any:
    return PracticeEnvironment()
