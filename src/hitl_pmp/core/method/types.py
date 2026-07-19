from collections.abc import Callable
from typing import Any

from hitl_pmp.core.problem.environment.types import Action, State

Policy = Callable[[State], Action]
Rollout = Any
Skill = Any
SetupCommand = Any
