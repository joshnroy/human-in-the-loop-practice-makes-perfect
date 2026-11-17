"""pybullet_helpers' `ikfast_closest_inverse_kinematics`, value for value, faster.

Upstream measures every IKFast candidate's joint-wise distance from the current
configuration, a Python loop over the joints, only to keep those within
`max_distance`. The kitchens' planners leave that at infinity, where the test keeps
every candidate whose distance is not NaN; with finite candidates and a finite
configuration it is never NaN, so the measurement is skipped there, and any other
candidate takes upstream's own check. It was a quarter of each IK call (about 9 s of
one RackToServing3D robot batch return). The free-joint samples are drawn a batch at
a time (`FreeJointSamples`), the same stream of doubles. The candidate shuffle, the
joint-limit check and the final sort are upstream's own code, in upstream's order, so
the random stream and the returned list are unchanged.
"""

import math
from functools import partial
from itertools import islice
from typing import Any, ClassVar

import numpy as np
from pybullet_helpers.geometry import Pose, matrix_from_quat
from pybullet_helpers.ikfast.load import import_ikfast
from pybullet_helpers.ikfast.utils import (
    IKFastHyperparameters,
    get_base_from_ee,
    get_ikfast_joints,
    ikfast_closest_inverse_kinematics,
    violates_joint_limits,
)
from pybullet_helpers.joint import (
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_jointwise_difference,
)


class IKFastClosest:
    """A static-method container, never instantiated."""

    @staticmethod
    def solve(
        *,
        robot: Any,
        world_from_target: Pose,
        hyperparameters: IKFastHyperparameters,
        seed: int = 0,
    ) -> list[Any]:
        """`ikfast_closest_inverse_kinematics(robot, world_from_target, seed,
        hyperparameters)`; upstream itself unless no time limit and no distance limit."""
        if not (math.isinf(hyperparameters.max_time) and math.isinf(hyperparameters.max_distance)):
            return ikfast_closest_inverse_kinematics(
                robot, world_from_target, seed=seed, hyperparameters=hyperparameters
            )
        if not hyperparameters.max_attempts < np.inf and not (
            hyperparameters.max_candidates < np.inf
        ):
            raise ValueError(
                "At least one of max_time, max_attempts, max_candidates must be finite."
            )
        ik_joint_infos, free_joint_infos = get_ikfast_joints(robot)
        ik_joints = [joint_info.jointIndex for joint_info in ik_joint_infos]
        current = get_joint_positions(robot.robot_id, ik_joints, robot.physics_client_id)
        generator = IKFastClosest.candidates(
            robot=robot,
            world_from_target=world_from_target,
            ik_joint_infos=ik_joint_infos,
            free_joint_infos=free_joint_infos,
            current=current,
            hyperparameters=hyperparameters,
            rng=np.random.default_rng(seed),
        )
        if hyperparameters.max_candidates < np.inf:
            generator = islice(generator, hyperparameters.max_candidates)
        difference = partial(get_jointwise_difference, ik_joint_infos)
        norm = hyperparameters.norm

        def difference_from_current(joint_positions: Any) -> float:  # noqa: PLR0917 -- sort key
            return np.linalg.norm(difference(current, joint_positions), ord=norm)  # type: ignore[return-value]

        return sorted(list(generator), key=difference_from_current)

    @staticmethod
    def candidates(
        *,
        robot: Any,
        world_from_target: Pose,
        ik_joint_infos: list[Any],
        free_joint_infos: list[Any],
        current: list[float],
        hyperparameters: IKFastHyperparameters,
        rng: np.random.Generator,
    ) -> Any:
        """Upstream's `ikfast_inverse_kinematics` generator with no time limit."""
        ikfast = import_ikfast(robot.ikfast_info())
        base_from_ee = get_base_from_ee(robot, world_from_target)
        position = list(base_from_ee.position)
        rotation = matrix_from_quat(base_from_ee.orientation).tolist()
        sampler = FreeJointSamples(
            robot=robot,
            free_joint_infos=free_joint_infos,
            max_distance=hyperparameters.max_distance,
            rng=rng,
        )
        samples: Any = sampler.samples()
        if hyperparameters.max_attempts < np.inf:
            samples = islice(samples, hyperparameters.max_attempts)
        difference = partial(get_jointwise_difference, ik_joint_infos)
        current_finite = all(math.isfinite(v) for v in current)
        for free_positions in samples:
            found = ikfast.get_ik(rotation, position, list(free_positions))
            if found is None:
                continue
            sampler.sync()
            rng.shuffle(found)
            for joint_positions in found:
                if violates_joint_limits(ik_joint_infos, joint_positions):
                    continue
                # Upstream's own distance test only where it could fail.
                if (current_finite and all(math.isfinite(v) for v in joint_positions)) or (
                    np.linalg.norm(difference(current, joint_positions), ord=hyperparameters.norm)
                    <= hyperparameters.max_distance
                ):
                    yield joint_positions


class FreeJointSamples:
    """Upstream's `free_joints_generator` samples, value for value, drawn a batch at
    a time.

    Upstream draws one `rng.uniform(lower, upper)` per attempt, and nearly every
    attempt of an unreachable pose (414 of 562 solves in one robot batch return, all
    1500 attempts each) finds nothing, so the draw's Python overhead was a fifth of
    the solve. A batch of draws is the same stream of doubles as that many single
    draws. The candidate shuffle also draws from `rng`, so before each shuffle `sync`
    rewinds the generator to just after the samples handed out so far, and the next
    sample starts a new batch from there: every draw sees the state upstream's does.
    """

    BATCH: ClassVar[int] = 256

    def __init__(
        self, *, robot: Any, free_joint_infos: list[Any], max_distance: float, rng: Any
    ) -> None:
        free_joints = [joint_info.jointIndex for joint_info in free_joint_infos]
        client = robot.physics_client_id
        self.current = get_joint_positions(robot.robot_id, free_joints, client)
        self.lower = np.maximum(
            get_joint_lower_limits(robot.robot_id, free_joints, client),
            np.array(self.current) - max_distance,
        )
        self.upper = np.minimum(
            get_joint_upper_limits(robot.robot_id, free_joints, client),
            np.array(self.current) + max_distance,
        )
        assert np.less_equal(self.lower, self.upper).all()
        for index, joint_info in enumerate(free_joint_infos):
            if joint_info.is_circular:
                assert np.isinf(self.lower[index]) and np.isinf(self.upper[index])
                self.lower[index] = -np.pi
                self.upper[index] = np.pi
        self.rng = rng
        self._state: Any = None
        self._batch: np.ndarray | None = None
        self._used = 0

    def samples(self) -> Any:
        yield self.current
        if np.equal(self.lower, self.upper).all():
            yield self.lower
            return
        while True:
            if self._batch is None or self._used == len(self._batch):
                self._state = self.rng.bit_generator.state
                self._batch = self.rng.uniform(
                    self.lower, self.upper, size=(self.BATCH, len(self.lower))
                )
                self._used = 0
            row = self._batch[self._used]
            self._used += 1
            yield row

    def sync(self) -> None:
        """Leave `rng` exactly where upstream's single draws would have left it."""
        if self._batch is None:
            return
        self.rng.bit_generator.state = self._state
        self.rng.uniform(self.lower, self.upper, size=(self._used, len(self.lower)))
        self._batch = None
