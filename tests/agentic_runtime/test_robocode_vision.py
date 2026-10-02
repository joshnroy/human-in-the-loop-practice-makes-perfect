"""Offline contracts for multimodal judgments through Robocode's fixed broker."""

import base64
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hitl_pmp.agentic_runtime import vision
from hitl_pmp.agentic_runtime.types import OptionContract

PNG = "data:image/png;base64," + base64.b64encode(b"unchanged-image-bytes").decode()


def messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Return an unknown judgment if the cube is hidden."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '{"cluster":"near"}'},
                {"type": "image_url", "image_url": {"url": PNG}},
            ],
        },
    ]


def completed(*, text: str) -> bytes:
    return (
        "data: "
        + json.dumps({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 42, "output_tokens": 12},
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": text}]},
                ],
            },
        })
        + "\n\n"
    ).encode()


def output_done(*, text: str, index: int = 0) -> bytes:
    return (
        "data: "
        + json.dumps({
            "type": "response.output_item.done",
            "output_index": index,
            "item": {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        })
        + "\n\n"
    ).encode()


def empty_completion() -> bytes:
    return b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'


@pytest.fixture
def broker_setup(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "order": [],
        "response": completed(text='{"value":"unknown"}'),
        "abort_event": threading.Event(),
    }

    def validate(protocol: str, path: str, raw: bytes) -> None:  # noqa: PLR0917
        calls["order"].append("validate")
        calls["protocol"] = protocol
        calls["path"] = path
        calls["body"] = json.loads(raw)

    def upstream(backend: str) -> Any:  # noqa: PLR0917
        calls["order"].append("credentials")
        calls["backend"] = backend
        return SimpleNamespace(protocol="responses" if backend == "codex" else "messages")

    @contextmanager
    def broker(directory: Path, provider: Any, log_path: Path) -> Any:  # noqa: PLR0917
        calls["log_path"] = log_path
        calls["order"].append("broker")
        yield directory / "model.sock"

    module = SimpleNamespace(
        __file__=str(tmp_path / "src/robocode/utils/model_broker.py"),
        validate_request=validate,
        load_broker_upstream=upstream,
        model_broker=broker,
    )
    monkeypatch.setattr(vision.importlib, "import_module", lambda _: module)

    class Connection:
        def __init__(self, *, path: Path, timeout: float) -> None:
            calls["socket"] = path
            calls["timeout"] = timeout

        def request(self, method: str, path: str, **kwargs: Any) -> None:  # noqa: PLR0917
            calls["request"] = (method, path, kwargs)

        def getresponse(self) -> Any:
            return SimpleNamespace(
                status=calls.get("status", 200), read=calls.get("read", lambda _: calls["response"])
            )

        def abort(self) -> None:
            calls["abort_event"].set()

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(vision, "_UnixHTTPConnection", Connection)
    calls["config"] = {
        "provider": "robocode_broker",
        "backend": "codex",
        "model": "configured-test-model",
        "robocode_checkout": str(tmp_path),
        "artifact_dir": str(tmp_path / "audit"),
        "reasoning_effort": "high",
        "request_timeout_s": 30,
    }
    return calls


def test_codex_keeps_images_system_prompt_and_fixed_protocol(
    *, broker_setup: dict[str, Any]
) -> None:
    client = vision.RobocodeVisionClient(config=broker_setup["config"])
    assert client.complete(messages=messages()) == '{"value":"unknown"}'
    body = broker_setup["body"]
    assert body["instructions"] == messages()[0]["content"]
    assert body["model"] == "configured-test-model"
    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": '{"cluster":"near"}'},
        {"type": "input_image", "image_url": PNG},
    ]
    assert body["tools"] == [] and body["tool_choice"] == "none"
    assert body["store"] is False and body["stream"] is True
    assert body["reasoning"] == {"effort": "high"}
    assert broker_setup["path"] == "/v1/responses"
    assert broker_setup["order"] == ["validate", "credentials", "broker"]
    assert broker_setup["request"][2]["headers"] == {"Content-Type": "application/json"}
    assert broker_setup["closed"]
    call_dir = broker_setup["log_path"].parent
    assert json.loads((call_dir / "request.json").read_text()) == body
    assert (call_dir / "response.raw").read_bytes() == broker_setup["response"]
    metadata = json.loads((call_dir / "metadata.json").read_text())
    assert metadata["usage"] == {"input_tokens": 42, "output_tokens": 12}
    assert metadata["status"] == "completed"


def test_claude_keeps_png_bytes_and_uses_messages(*, broker_setup: dict[str, Any]) -> None:
    broker_setup["config"]["backend"] = "claude"
    broker_setup["response"] = json.dumps({
        "type": "message",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": '{"value":"true"}'}],
    }).encode()
    client = vision.RobocodeVisionClient(config=broker_setup["config"])
    assert client.complete(messages=messages()) == '{"value":"true"}'
    body = broker_setup["body"]
    assert body["system"] == messages()[0]["content"]
    assert body["messages"][0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": PNG.split(",")[1],
        },
    }
    assert body["tools"] == [] and body["stream"] is False
    assert broker_setup["path"] == "/v1/messages"


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_object_state_judgments_use_text_only_broker_requests(
    *, broker_setup: dict[str, Any], backend: str
) -> None:
    broker_setup["config"]["backend"] = backend
    result = '{"value":null,"reason":"The grip cannot be established from these measurements."}'
    broker_setup["response"] = (
        completed(text=result)
        if backend == "codex"
        else json.dumps({
            "type": "message",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": result}],
        }).encode()
    )
    observation = {
        "observation_mode": "object_state",
        "objects": [{"name": "cube0", "type": "cube", "features": {"x": 0.4, "z": 0.3}}],
        "proprioception": {"pos_gripper": 0.7},
        "control_step": 15,
    }
    option = OptionContract(
        option_id="pick",
        belief_skill="pick",
        initiation="The cube is reachable.",
        termination="The controller stops.",
        success="The cube is held above the floor.",
        controller="pick.py",
        cost=1,
    )
    judge = vision.VLMJudge(
        client=vision.RobocodeVisionClient(config=broker_setup["config"]),
        classification_prompt="classify",
        option_check_prompt="Check the language condition.",
    )
    assert judge.check(observation=observation, option=option, phase="success").value is None
    body = broker_setup["body"]
    blocks = body["input" if backend == "codex" else "messages"][0]["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == ("input_text" if backend == "codex" else "text")
    assert json.loads(blocks[0]["text"]) == {
        "option_id": "pick",
        "phase": "success",
        "description": option.success,
        "observation": observation,
    }
    assert body["tools"] == []
    assert (broker_setup["log_path"].parent / "judgment.txt").read_text() == result


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_provider_failure_never_becomes_a_judgment(
    *, broker_setup: dict[str, Any], status: int
) -> None:
    broker_setup["status"] = status
    client = vision.RobocodeVisionClient(config=broker_setup["config"])
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        client.complete(messages=messages())
    assert broker_setup["closed"]
    metadata = json.loads((broker_setup["log_path"].parent / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["http_status"] == status


def test_whole_request_deadline_interrupts_a_continuing_stream(
    *, broker_setup: dict[str, Any]
) -> None:
    broker_setup["config"]["request_timeout_s"] = 0.01

    def wait_for_abort(_: int) -> bytes:  # noqa: PLR0917
        assert broker_setup["abort_event"].wait(2)
        return broker_setup["response"]

    broker_setup["read"] = wait_for_abort
    client = vision.RobocodeVisionClient(config=broker_setup["config"])
    with pytest.raises(TimeoutError, match="total time limit"):
        client.complete(messages=messages())
    assert broker_setup["closed"]
    metadata = json.loads((broker_setup["log_path"].parent / "metadata.json").read_text())
    assert metadata["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.output_text.delta", "delta": '{"value":"true"}'},
        {"type": "response.incomplete"},
        {"type": "response.failed"},
        {"type": "error"},
    ],
)
def test_incomplete_stream_never_returns_partial_json(*, event: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError):
        vision._BrokerVisionClient._response_text(
            protocol="responses", raw=("data: " + json.dumps(event) + "\n\n").encode()
        )


def test_remote_image_is_rejected_before_credentials(*, broker_setup: dict[str, Any]) -> None:
    client = vision.RobocodeVisionClient(config=broker_setup["config"])
    request = messages()
    request[1]["content"][1]["image_url"]["url"] = "https://example.com/image.png"
    with pytest.raises(ValueError, match="inline PNG"):
        client.complete(messages=request)
    assert broker_setup["order"] == []


def test_wrong_robocode_checkout_is_rejected(*, broker_setup: dict[str, Any]) -> None:
    broker_setup["config"]["robocode_checkout"] += "/different"
    with pytest.raises(RuntimeError, match="configured vision checkout"):
        vision.RobocodeVisionClient(config=broker_setup["config"])


def test_plain_cli_is_not_silently_used_for_images() -> None:
    with pytest.raises(ValueError, match="Vision provider"):
        vision.RobocodeVisionClient(config={"provider": "cli"})


def test_claude_token_limit_is_not_treated_as_completion() -> None:
    with pytest.raises(RuntimeError, match="did not complete"):
        vision._BrokerVisionClient._response_text(
            protocol="messages",
            raw=json.dumps({
                "type": "message",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": '{"value":"true"}'}],
            }).encode(),
        )


def test_finalized_items_fill_empty_completed_output_in_index_order() -> None:
    raw = (
        output_done(text='null,"reason":"Ambiguous."}', index=2)
        + output_done(text='{"value":', index=1)
        + empty_completion()
    )
    assert (
        vision._BrokerVisionClient._response_text(protocol="responses", raw=raw)
        == '{"value":null,"reason":"Ambiguous."}'
    )


def test_full_completed_output_remains_authoritative() -> None:
    raw = output_done(text="Do not concatenate this.") + completed(text='{"value":false}')
    assert (
        vision._BrokerVisionClient._response_text(protocol="responses", raw=raw)
        == '{"value":false}'
    )


@pytest.mark.parametrize("terminal", [None, "error", "response.failed", "response.incomplete"])
def test_finalized_message_without_successful_completion_is_not_a_judgment(
    *, terminal: str | None
) -> None:
    raw = output_done(text='{"value":true}')
    if terminal is not None:
        raw += ("data: " + json.dumps({"type": terminal}) + "\n\n").encode()
    with pytest.raises(RuntimeError):
        vision._BrokerVisionClient._response_text(protocol="responses", raw=raw)


@pytest.mark.parametrize(
    "item",
    [
        {"type": "function_call", "name": "pretend_success"},
        {
            "type": "message",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "No."}],
        },
        {
            "type": "message",
            "status": "incomplete",
            "content": [{"type": "output_text", "text": '{"value":true}'}],
        },
    ],
)
def test_finalized_nontext_refusal_or_incomplete_item_is_rejected(*, item: dict[str, Any]) -> None:
    raw = (
        "data: "
        + json.dumps({"type": "response.output_item.done", "output_index": 0, "item": item})
        + "\n\n"
    ).encode() + empty_completion()
    with pytest.raises(RuntimeError):
        vision._BrokerVisionClient._response_text(protocol="responses", raw=raw)


def test_deltas_are_not_a_fallback_for_missing_finalized_message() -> None:
    raw = (
        b'data: {"type":"response.output_text.delta","delta":"{\\"value\\":true}"}\n\n'
        + empty_completion()
    )
    with pytest.raises(RuntimeError, match="empty judgment"):
        vision._BrokerVisionClient._response_text(protocol="responses", raw=raw)
