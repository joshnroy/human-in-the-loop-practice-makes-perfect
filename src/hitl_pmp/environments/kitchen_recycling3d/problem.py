"""The kitchen's `Problem` is Tossing3D's."""

from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem


class KitchenRecyclingProblem(Tossing3DProblem):
    """Tossing3D's episode loop and horizon, over the kitchen environment."""
