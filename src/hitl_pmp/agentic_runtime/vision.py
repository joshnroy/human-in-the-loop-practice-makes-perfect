"""Model judgments from declared observations, with caller-owned prompts."""

import http.client
import importlib
import json
import math
import socket
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol

from .observations import ObservationEvidence
from .types import ClusterJudgment, ContractJudgment, OptionContract, RuntimeManifest


class VisionClient(Protocol):
    """A single multimodal completion; no tools or arbitrary network capabilities."""

    def complete(self, *, messages: list[dict[str, Any]]) -> str: ...


class VLMJudge:
    """Unknown evidence remains unknown for visual and object-state conditions."""

    def __init__(
        self, *, client: VisionClient, classification_prompt: str, option_check_prompt: str
    ) -> None:
        self._client = client
        self._classification_prompt = classification_prompt
        self._option_check_prompt = option_check_prompt

    def classify(
        self, *, observation: dict[str, Any], manifest: RuntimeManifest
    ) -> ClusterJudgment:
        text = self._complete(
            observation=observation,
            prompt=self._classification_prompt,
            request={"clusters": [item.model_dump() for item in manifest.clusters]},
        )
        result = ClusterJudgment.model_validate_json(text)
        if result.cluster_id is not None and result.cluster_id not in {
            item.cluster_id for item in manifest.clusters
        }:
            raise ValueError("VLM returned an unknown cluster ID")
        return result

    def check(
        self,
        *,
        observation: dict[str, Any],
        option: OptionContract,
        phase: Literal["initiation", "termination", "success"],
    ) -> ContractJudgment:
        text = self._complete(
            observation=observation,
            prompt=self._option_check_prompt,
            request={
                "option_id": option.option_id,
                "phase": phase,
                "description": getattr(option, phase),
            },
        )
        return ContractJudgment.model_validate_json(text)

    def _complete(
        self, *, observation: dict[str, Any], prompt: str, request: dict[str, Any]
    ) -> str:
        images: list[Any] = []
        if observation.get("observation_mode") == "object_state":
            self._validate_object_state(observation=observation)
        else:
            image_observations = observation.get("images")
            if not isinstance(image_observations, list) or not image_observations:
                raise ValueError("VLM judgments require actual image observations")
            images = image_observations
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        **request,
                        "observation": {k: v for k, v in observation.items() if k != "images"},
                    },
                    allow_nan=False,
                ),
            }
        ]
        for image in images:
            if not isinstance(image, str) or not image.startswith("data:image/png;base64,"):
                raise ValueError("Only inline PNG observations are accepted")
            content.append({"type": "image_url", "image_url": {"url": image}})
        return self._client.complete(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}]
        )

    @staticmethod
    def _validate_object_state(*, observation: dict[str, Any]) -> None:
        allowed = {
            "observation_mode",
            "objects",
            "proprioception",
            "action_spec",
            "control_step",
            "simulation_time_s",
            "state_spec",
            "execution_seed",
            "execution",
            "recent_observations",
        }
        if observation.get("observation_mode") != "object_state":
            raise ValueError("Numeric observations require explicit object_state mode")
        if observation.keys() - allowed:
            raise ValueError("Object-state observations contain undeclared fields")
        if "simulation_time_s" in observation:
            timestamp = observation["simulation_time_s"]
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError("simulation_time_s must be a finite nonnegative number")
        objects = observation.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("Object-state judgments require nonempty objects")
        names: set[str] = set()
        for item in objects:
            if not isinstance(item, dict) or set(item) != {"name", "type", "features"}:
                raise ValueError("Each object requires only name, type, and numeric features")
            if any(
                not isinstance(item[key], str) or not item[key].strip() for key in ("name", "type")
            ):
                raise ValueError("Object names and types must be nonempty strings")
            if item["name"] in names:
                raise ValueError("Object names must be unique")
            names.add(item["name"])
            VLMJudge._validate_measurements(values=item["features"])
        if "proprioception" in observation:
            VLMJudge._validate_measurements(values=observation["proprioception"])
        if "execution" in observation:
            execution = observation["execution"]
            if not isinstance(execution, dict) or execution.keys() - {
                "steps",
                "controller_done",
                "error",
                "trajectory_path",
                "seed",
            }:
                raise ValueError("Execution metadata may describe control, not skill success")
        if "recent_observations" in observation:
            history = observation["recent_observations"]
            if (
                not isinstance(history, list)
                or not 1 <= len(history) <= ObservationEvidence.MAX_HISTORY
            ):
                raise ValueError("Recent observations must contain one to eight physical samples")
            for sample in history:
                if not isinstance(sample, dict) or sample.keys() & {
                    "recent_observations",
                    "execution",
                    "execution_seed",
                }:
                    raise ValueError(
                        "Recent observations cannot contain nested or execution metadata"
                    )
                VLMJudge._validate_object_state(observation=sample)

    @staticmethod
    def _validate_measurements(*, values: Any) -> None:
        if not isinstance(values, dict) or not values:
            raise ValueError("Object-state measurements must be a nonempty numeric dictionary")
        # Descriptions are judged from measurements, never simulator-provided answers.
        forbidden = {
            "success",
            "failure",
            "reward",
            "done",
            "terminated",
            "truncated",
            "predicates",
            "goal_reached",
        }
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip() or key.strip().casefold() in forbidden:
                raise ValueError("Object-state features must be labeled measurements, not outcomes")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "Object-state measurements must be finite numbers, not Boolean flags"
                )


class RobocodeVisionClient:
    """Multimodal completion through Robocode's API client or fixed host broker.

    The broker provider reuses the coding agent's authenticated provider without
    running an agent or enabling tools. This is a multimodal extension to the
    integration: Robocode's existing plain-text CLI client cannot preserve images.
    """

    def __init__(self, *, config: dict[str, Any]) -> None:
        self._broker_client: _BrokerVisionClient | None = None
        if config.get("provider") == "robocode_broker":
            self._broker_client = _BrokerVisionClient(config=config)
            return
        if config.get("provider") != "openai_compatible":
            raise ValueError("Vision provider must be robocode_broker or openai_compatible")
        try:
            factory = importlib.import_module("robocode.utils.llm")
            omega = importlib.import_module("omegaconf")
        except ImportError as exc:
            raise RuntimeError(
                "Install the configured Robocode checkout to enable live VLM calls"
            ) from exc
        self._client = factory.create_llm_client(omega.DictConfig(config))

    def complete(self, *, messages: list[dict[str, Any]]) -> str:
        if self._broker_client is not None:
            return self._broker_client.complete(messages=messages)
        return str(self._client.complete(messages).text)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP framing over Robocode's private Unix socket, never arbitrary TCP."""

    def __init__(self, *, path: Path, timeout: float) -> None:
        super().__init__("robocode-model-broker", timeout=timeout)
        self._path = path
        self._request_socket: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._request_socket = self.sock
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))

    def abort(self) -> None:
        """Interrupt a response read even after HTTP/1.0 hands socket ownership off."""
        if self._request_socket is not None:
            with suppress(OSError):
                self._request_socket.shutdown(socket.SHUT_RDWR)


class _BrokerVisionClient:
    """Host-side VLM adapter using only Robocode's public broker interfaces."""

    _MAX_RESPONSE = 8_000_000

    def __init__(self, *, config: dict[str, Any]) -> None:
        self._backend = config["backend"]
        if self._backend not in {"codex", "claude"}:
            raise ValueError("The Robocode vision broker supports codex or claude")
        self._model = config["model"]
        if not isinstance(self._model, str) or not self._model.strip():
            raise ValueError("Choose an explicit vision model")
        self._timeout = float(config.get("request_timeout_s", 120))
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("Vision timeout must be finite and positive")
        self._reasoning_effort = config.get("reasoning_effort", "")
        self._artifact_dir = Path(config["artifact_dir"])
        try:
            self._broker = importlib.import_module("robocode.utils.model_broker")
        except ImportError as exc:
            raise RuntimeError("Install the configured Robocode checkout for VLM calls") from exc
        expected = (
            Path(config["robocode_checkout"]).resolve() / "src/robocode/utils/model_broker.py"
        )
        actual = self._broker.__file__
        if actual is None or Path(actual).resolve() != expected:
            raise RuntimeError("Imported Robocode differs from the configured vision checkout")

    def complete(self, *, messages: list[dict[str, Any]]) -> str:
        protocol, path, body = self._request(messages=messages)
        # Validate before loading credentials or opening any provider connection.
        raw = json.dumps(body, allow_nan=False).encode()
        self._broker.validate_request(protocol, path, raw)
        call_dir = self._artifact_dir / uuid.uuid4().hex
        call_dir.mkdir(parents=True, exist_ok=False)
        (call_dir / "request.json").write_bytes(raw)
        metadata: dict[str, Any] = {
            "backend": self._backend,
            "model": self._model,
            "protocol": protocol,
            "path": path,
            "status": "started",
            "usage": None,
        }
        started = time.monotonic()
        try:
            provider = self._broker.load_broker_upstream(self._backend)
            if provider.protocol != protocol:
                raise RuntimeError("Robocode broker protocol does not match the chosen backend")
            with (
                tempfile.TemporaryDirectory(prefix="hitl-vlm-") as temporary,
                self._broker.model_broker(
                    Path(temporary), provider, call_dir / "broker.jsonl"
                ) as endpoint,
            ):
                connection = _UnixHTTPConnection(path=endpoint, timeout=self._timeout)
                expired = threading.Event()

                def expire() -> None:
                    expired.set()
                    connection.abort()

                timer = threading.Timer(self._timeout, expire)
                timer.daemon = True
                timer.start()
                try:
                    connection.request(
                        "POST", path, body=raw, headers={"Content-Type": "application/json"}
                    )
                    response = connection.getresponse()
                    metadata["http_status"] = response.status
                    data = response.read(self._MAX_RESPONSE + 1)
                    (call_dir / "response.raw").write_bytes(data)
                    if expired.is_set():
                        raise TimeoutError("Robocode VLM request exceeded its total time limit")
                    if response.status != 200:
                        raise RuntimeError(
                            f"Robocode VLM request failed with HTTP {response.status}"
                        )
                    if len(data) > self._MAX_RESPONSE:
                        raise RuntimeError("Robocode VLM response exceeds the size limit")
                    result = self._response_text(protocol=protocol, raw=data)
                    metadata["usage"] = self._usage(protocol=protocol, raw=data)
                    metadata["status"] = "completed"
                    (call_dir / "judgment.txt").write_text(result)
                    return result
                finally:
                    timer.cancel()
                    connection.close()
        except Exception as exc:
            metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            metadata["elapsed_seconds"] = time.monotonic() - started
            (call_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @staticmethod
    def _usage(*, protocol: str, raw: bytes) -> Any:
        if protocol == "messages":
            return json.loads(raw).get("usage")
        for chunk in raw.decode().replace("\r\n", "\n").split("\n\n"):
            lines = [line[5:].lstrip() for line in chunk.splitlines() if line.startswith("data:")]
            if not lines or lines == ["[DONE]"]:
                continue
            event = json.loads("\n".join(lines))
            if event.get("type") == "response.completed":
                return event.get("response", {}).get("usage")
        return None

    def _request(self, *, messages: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        systems: list[str] = []
        inputs: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system" and isinstance(content, str):
                systems.append(content)
                continue
            if role != "user" or not isinstance(content, list):
                raise ValueError("Vision requests require system text and multimodal user content")
            blocks: list[dict[str, Any]] = []
            for block in content:
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({
                        "type": "input_text" if self._backend == "codex" else "text",
                        "text": block["text"],
                    })
                elif block.get("type") == "image_url":
                    image = block.get("image_url", {}).get("url", "")
                    if not isinstance(image, str) or not image.startswith("data:image/png;base64,"):
                        raise ValueError("Only inline PNG observations are accepted")
                    if self._backend == "codex":
                        blocks.append({"type": "input_image", "image_url": image})
                    else:
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image.split(",", 1)[1],
                            },
                        })
                else:
                    raise ValueError("Unsupported vision content block")
            inputs.append({"role": "user", "content": blocks})
        if not systems or not inputs:
            raise ValueError("Vision requests require a system prompt and observation")
        if self._backend == "codex":
            body: dict[str, Any] = {
                "model": self._model,
                "instructions": "\n\n".join(systems),
                "input": inputs,
                "tools": [],
                "tool_choice": "none",
                "store": False,
                "stream": True,
            }
            if self._reasoning_effort:
                body["reasoning"] = {"effort": self._reasoning_effort}
            return "responses", "/v1/responses", body
        return (
            "messages",
            "/v1/messages",
            {
                "model": self._model,
                "system": "\n\n".join(systems),
                "messages": inputs,
                "tools": [],
                "max_tokens": 2048,
                "stream": False,
            },
        )

    @staticmethod
    def _response_text(*, protocol: str, raw: bytes) -> str:
        if protocol == "messages":
            data = json.loads(raw)
            if data.get("type") != "message" or data.get("stop_reason") != "end_turn":
                raise RuntimeError("Robocode VLM did not complete its judgment")
            blocks = data.get("content", [])
            if not blocks or any(block.get("type") != "text" for block in blocks):
                raise RuntimeError("Robocode VLM returned non-text content")
            return "".join(block["text"] for block in blocks)
        # A completed event is required. Partial deltas and EOF cannot become a
        # guessed judgment when inference fails or its stream is interrupted.
        finalized: dict[int, dict[str, Any]] = {}
        for event in raw.decode().replace("\r\n", "\n").split("\n\n"):
            lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")]
            if not lines or lines == ["[DONE]"]:
                continue
            data = json.loads("\n".join(lines))
            if data.get("type") in {"error", "response.failed", "response.incomplete"}:
                raise RuntimeError("Robocode VLM inference did not complete")
            if data.get("type") == "response.output_item.done":
                index, item = data.get("output_index"), data.get("item")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index in finalized
                    or not isinstance(item, dict)
                ):
                    raise RuntimeError("Robocode VLM returned invalid finalized output")
                if item.get("type") == "message" and item.get("status") != "completed":
                    raise RuntimeError("Robocode VLM finalized message is incomplete")
                _BrokerVisionClient._output_text(output=item)
                finalized[index] = item
                continue
            if data.get("type") != "response.completed":
                continue
            response = data["response"]
            if response.get("status") != "completed":
                raise RuntimeError("Robocode VLM judgment is incomplete")
            outputs = response.get("output", [])
            if not isinstance(outputs, list):
                raise RuntimeError("Robocode VLM returned invalid completed output")
            if not outputs:
                outputs = [finalized[index] for index in sorted(finalized)]
            text = "".join(_BrokerVisionClient._output_text(output=output) for output in outputs)
            if text.strip():
                return text
            raise RuntimeError("Robocode VLM returned an empty judgment")
        raise RuntimeError("Robocode VLM stream ended without a completed judgment")

    @staticmethod
    def _output_text(*, output: dict[str, Any]) -> str:
        if output.get("type") == "reasoning":
            return ""
        if output.get("type") != "message":
            raise RuntimeError("Robocode VLM returned non-text output")
        if output.get("status", "completed") != "completed":
            raise RuntimeError("Robocode VLM message is incomplete")
        texts: list[str] = []
        for block in output.get("content", []):
            if block.get("type") != "output_text" or not isinstance(block.get("text"), str):
                raise RuntimeError("Robocode VLM returned a refusal or non-text judgment")
            texts.append(block["text"])
        return "".join(texts)
