"""Authentication parity: the adapter delegates policy to Robocode."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hitl_pmp.agentic_runtime.broker_auth import BrokerAuth


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_native_auth_is_used_without_expiry_gate(*, backend):
    native = Mock(return_value=object())
    broker = SimpleNamespace(load_broker_upstream=native)
    assert BrokerAuth.load_upstream(backend=backend, broker=broker) is native.return_value
    native.assert_called_once_with(backend)


def test_native_auth_failure_propagates():
    broker = SimpleNamespace(load_broker_upstream=Mock(side_effect=RuntimeError("auth failed")))
    with pytest.raises(RuntimeError, match="auth failed"):
        BrokerAuth.load_upstream(backend="claude", broker=broker)
