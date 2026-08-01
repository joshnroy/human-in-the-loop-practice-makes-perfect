"""The structural shapes the global CLI (`cli.py`) wires together: what an
environment-CLI (`ENVIRONMENTS` value) and a method-CLI (`METHODS` value) each must
expose. Kept in their own module, below both `environments/` and `methods/` in the
import layering, so a method-CLI can be typed against `EnvironmentCli` (it drives the
selected env's `run_method`) without importing `cli.py` -- which would be a cycle,
since `cli.py` imports every concrete method-CLI. Both are Protocols (structural), so
a concrete CLI satisfies them just by having matching staticmethods, no subclassing.
"""

import argparse
from collections.abc import Callable
from typing import Protocol

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext


class EnvironmentCli(Protocol):
    """The shape every environments/<domain>/cli.py entry in `ENVIRONMENTS` must
    expose: `add_arguments` (cli.py registers the selected env's config flags) and
    `run_method` (its composition root -- builds env/tasks/problem plus the domain's
    SkillProvider/OraclePolicyProvider, then calls the method-CLI's method_factory)."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None: ...


class MethodCli(Protocol):
    """The shape every methods/<name>/cli.py entry in `METHODS` must expose:
    `add_arguments` (its method-specific flags) and `run` (which drives the selected
    `env_cli.run_method` with a method_factory that builds this method from the
    env's DomainContext). A method-CLI never imports a specific environment -- it
    receives `env_cli` from cli.py, the one place allowed to import both sides."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None: ...
