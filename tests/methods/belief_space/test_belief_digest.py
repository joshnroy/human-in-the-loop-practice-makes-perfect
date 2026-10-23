"""The search memo's belief identifier must be the very digest it always was.

Expectimax memoizes on these bytes, so any change to them -- a collision, or two
encodings of one posterior -- would change which nodes the search shares. A faster
identifier is admissible only if it is the same identifier.
"""

import gc
import hashlib
import weakref
from enum import Enum

import numpy as np
import pytest
from pydantic import BaseModel

from hitl_pmp.methods.belief_space.belief_digest import BeliefDigest
from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    InferenceConfig,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    create_broad_particle_prior,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import (
    WeightedHypothesisBelief,
)


class ReferenceDigest:
    """The SHA-256 identifier as `Tossing3DPracticeModel._belief_signature` computed it
    before it was made faster, kept verbatim as the specification."""

    @staticmethod
    def of(*, value: object) -> bytes:
        digest = hashlib.sha256()

        def update(*, value: object) -> None:
            if isinstance(value, tuple):
                digest.update(b"tuple" + len(value).to_bytes(8, "big"))
                for item in value:
                    update(value=item)
                return
            if isinstance(value, BaseModel):
                digest.update(b"model")
                update(
                    value=(
                        "pydantic",
                        type(value).__module__,
                        type(value).__qualname__,
                        value.model_dump_json(),
                    )
                )
                return
            if isinstance(value, bytes):
                tag, payload = b"bytes", value
            elif isinstance(value, str):
                tag, payload = b"string", value.encode("utf-8")
            elif isinstance(value, bool):
                tag, payload = b"bool", bytes([value])
            elif isinstance(value, int):
                tag, payload = b"int", str(value).encode("ascii")
            elif isinstance(value, float):
                tag, payload = b"float", (0.0 if value == 0 else value).hex().encode("ascii")
            elif value is None:
                tag, payload = b"none", b""
            else:
                raise TypeError(f"unsupported belief signature component: {type(value).__name__}")
            digest.update(tag + len(payload).to_bytes(8, "big"))
            digest.update(payload)

        update(value=value)
        return digest.digest()

    @staticmethod
    def belief(*, belief: object) -> bytes:
        return ReferenceDigest.of(
            value=(type(belief).__module__, type(belief).__qualname__, belief.signature())
        )


class Flavour(str, Enum):
    SWEET = "sweet"


class Shape(BaseModel):
    side: float
    name: str = "cube"


def _beliefs() -> list[object]:
    grid_b = create_bayesian_prior(model="local_trend", engine="grid", seed=1, num_particles=8)
    grid_a = create_bayesian_prior(model="global_curve", engine="grid", seed=2, num_particles=8)
    particle = create_bayesian_prior(
        model="local_trend", engine="particle", seed=3, num_particles=64
    )
    hypotheses = WeightedHypothesisBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=0.25, learning_rate=0.0), probability=0.5
            ),
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=-0.0, learning_rate=0.1), probability=0.5
            ),
        )
    )
    return [
        grid_b,
        grid_b.condition_outcome(success=False).advance_cycle(training_examples=2),
        grid_a,
        particle,
        particle.condition_cost(observed_cost=3.0),
        create_broad_particle_prior(num_particles=32, seed=4),
        hypotheses,
        BayesianSkillBelief.model_validate_json(grid_b.model_dump_json()),
    ]


@pytest.mark.parametrize("index", range(8))
def test_each_belief_digest_is_byte_identical_to_the_reference(*, index: int) -> None:
    belief = _beliefs()[index]
    assert BeliefDigest.of(belief=belief) == ReferenceDigest.belief(belief=belief)
    assert Tossing3DPracticeModel._belief_signature(belief=belief) == ReferenceDigest.belief(
        belief=belief
    )


@pytest.mark.parametrize(
    "value",
    [
        (),
        (True, False, 0, 1, -7, 2**70, 0.0, -0.0, 1.5, float("inf"), None),
        ("", "é", b"", b"\x00\x01", ((), ((),))),
        (np.float64(-0.0), np.float64(0.1)),
        (Flavour.SWEET, Shape(side=0.5), (Shape(side=-0.0, name="ü"),)),
        (1, np.int64(5)),
        (np.bool_(True),),
        ([1.0],),
    ],
)
def test_every_component_type_encodes_to_the_reference_bytes(*, value: tuple) -> None:
    """Subclasses (enums, numpy floats) take the reference's isinstance route, and
    whatever it refuses (numpy ints and bools, lists) is refused here too."""
    try:
        expected = ReferenceDigest.of(value=value)
    except TypeError:
        with pytest.raises(TypeError):
            BeliefDigest.digest(value=value)
        return
    assert BeliefDigest.digest(value=value) == expected


def test_a_belief_is_encoded_once_and_its_shared_config_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search key digests every skill's posterior at every node, though most nodes
    share most posteriors with their parent; and every grid posterior carries the
    same config, which was re-serialized to JSON each time."""
    belief = create_bayesian_prior(model="local_trend", engine="grid", seed=5, num_particles=8)
    first = BeliefDigest.of(belief=belief)
    children = [belief.condition_outcome(success=bool(i % 2)) for i in range(4)]
    assert all(child.config is belief.config for child in children)
    dumps = []
    real_dump = InferenceConfig.model_dump_json
    monkeypatch.setattr(
        InferenceConfig,
        "model_dump_json",
        lambda self, **kwargs: dumps.append(1) or real_dump(self, **kwargs),
    )
    signatures = []
    real_signature = BayesianSkillBelief.signature
    monkeypatch.setattr(
        BayesianSkillBelief,
        "signature",
        lambda self: signatures.append(1) or real_signature(self),
    )
    assert BeliefDigest.of(belief=belief) == first
    assert signatures == []
    digests = [BeliefDigest.of(belief=child) for child in children]
    assert digests == [ReferenceDigest.belief(belief=child) for child in children]
    assert len(signatures) == 2 * len(children)  # ours, then the reference's
    assert len(dumps) == len(children)  # the reference's only


def test_remembered_digests_neither_keep_beliefs_alive_nor_outlive_them() -> None:
    belief = create_bayesian_prior(model="local_trend", engine="grid", seed=6, num_particles=8)
    BeliefDigest.of(belief=belief)
    reference = weakref.ref(belief)
    remembered = len(BeliefDigest._digests)
    del belief
    gc.collect()
    assert reference() is None
    assert len(BeliefDigest._digests) == remembered - 1
