"""The kitchen's tasks are Tossing3D's: train/test scene-seed streams, goal InBin."""

from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks


class KitchenRecyclingTasks(Tossing3DTasks):
    """Tossing3D's task streams over the kitchen environment."""
