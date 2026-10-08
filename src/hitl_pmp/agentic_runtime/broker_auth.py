"""Delegate authentication to the configured Robocode broker."""

from pathlib import Path
from typing import Any


class BrokerAuth:
    """Compatibility facade; no independent OAuth renewal or expiry policy."""

    @staticmethod
    def load_upstream(
        *,
        backend: str,
        broker: Any,
        timeout_seconds: float | None = None,
        auth_cli: Path | None = None,
    ) -> Any:
        """Use Robocode's own credential selection unchanged."""
        return broker.load_broker_upstream(backend)
