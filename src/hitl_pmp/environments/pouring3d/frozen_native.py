"""Individually frozen cooked-shrimp interpretation of the rigid native proxy.

The geometry, mass, contact solver and reset remain those of shrimp_native.
Freezing is not simulated: temperature, ice glaze, adhesion and thawing are not
modeled. Sliding friction is an explicit sensitivity assumption, not a measured
coefficient for frozen shrimp against this pitcher. Merely relabeling the same
rigid model does not change its dynamics.

Physical context: https://www.fao.org/4/x5931e/x5931e01.htm
Ice/polymer friction varies with conditions; no value below is inferred from:
https://doi.org/10.1016/j.triboint.2012.05.001
"""

from typing import Any, cast

from kinder.envs.dynamic3d.objects.base import register_object
from pydantic import Field

from .feedback_controller import FeedbackPourParameters
from .shrimp_native import CookedShrimp, ShrimpPourEnv, ShrimpPourScene


@register_object
class FrozenShrimp(CookedShrimp):
    """Same curled geometry as cooked shrimp, assumed frozen and not ice-bound."""


class FrozenShrimpScene(ShrimpPourScene):
    def physical_snapshot(self) -> dict[str, Any]:
        snapshot = super().physical_snapshot()
        snapshot.update(payload="individually frozen cooked-shrimp rigid proxy")
        return snapshot


class FrozenShrimpEnv(ShrimpPourEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> FrozenShrimpScene:
        return FrozenShrimpScene(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        scene = cast(FrozenShrimpScene, self._object_centric_env)
        return (
            f"Pour exactly {scene.requested_count} of three individually frozen shrimp "
            "proxies; retain the remainder, settle all pieces, and avoid spills. "
            "No ice bridges, melting or adhesion are modeled."
        )


class ExtendedFeedbackPourParameters(FeedbackPourParameters):
    """Same control law, with up to one minute of feeding for duration checks."""

    duration_seconds: float = Field(default=30, ge=0, le=60, allow_inf_nan=False)
