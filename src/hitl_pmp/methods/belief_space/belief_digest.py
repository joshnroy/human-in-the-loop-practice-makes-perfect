"""A posterior's SHA-256 search identifier, computed once per object."""

import hashlib
import weakref
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from .types.skill_belief import SkillBelief
from .types.weighted_hypothesis_belief import WeightedHypothesisBelief


class BeliefDigest:
    """Identify a posterior without retaining its potentially large buffers.

    Search memo tables live for one solve. A persistent signature-to-integer table
    once kept every hypothetical posterior alive across solves, including hundreds of
    KB per fixed-grid posterior; a SHA-256 identifier is constant-size, needs no
    interning table, and includes all the same signature information. Type tags and
    lengths prevent ambiguous joins.

    The byte stream hashed is fixed: memo hits and misses are decided by these
    digests, so a cheaper route must reach the identical bytes. Two routes are taken.
    A search key digests every skill's posterior at every node, though a child shares
    all but one posterior with its parent, so each digest is remembered against its
    frozen belief -- weakly, dropped when the belief dies, so it neither keeps the
    belief alive nor outlives it (which would let a later object reuse its id). And
    a nested model's encoding (every grid posterior carries the same config) is
    remembered the same way instead of re-serialized.
    """

    _digests: ClassVar[dict[int, tuple[weakref.ref[SkillBelief], bytes]]] = {}
    _models: ClassVar[dict[int, tuple[weakref.ref[BaseModel], bytes]]] = {}

    @staticmethod
    def of(*, belief: SkillBelief) -> bytes:
        entry = BeliefDigest._digests.get(id(belief))
        if entry is not None and entry[0]() is belief:
            return entry[1]
        if isinstance(belief, WeightedHypothesisBelief):
            digest = BeliefDigest._weighted_digest(belief=belief)
        else:
            digest = BeliefDigest.digest(
                value=(type(belief).__module__, type(belief).__qualname__, belief.signature())
            )
        BeliefDigest._remember(table=BeliefDigest._digests, owner=belief, encoded=digest)
        return digest

    # The per-hypothesis float encodings of a weighted belief's shared columns, remembered
    # against the column (weakly, like `_digests`): conditioning keeps the competences
    # and learning rates and makes only new weights, so most of a child's encoding is its
    # parent's. The weights are encoded afresh each time and never remembered.
    _columns: ClassVar[dict[int, tuple[weakref.ref[np.ndarray], list[bytes]]]] = {}
    _ROW: ClassVar[bytes] = b"tuple" + (3).to_bytes(8, "big")

    @staticmethod
    def _weighted_digest(*, belief: WeightedHypothesisBelief) -> bytes:
        """The general route's digest of `belief.signature()` -- a tuple of float
        triples -- to the byte, built from the columns."""
        competences, learning_rates, probabilities = belief.signature_columns()
        chunks: list[bytes] = []
        BeliefDigest._encode(
            value=(type(belief).__module__, type(belief).__qualname__), chunks=chunks
        )
        # The 2-tuple's header is rewritten as the 3-tuple's: the signature follows.
        chunks[0] = b"tuple" + (3).to_bytes(8, "big")
        chunks.append(b"tuple" + len(probabilities).to_bytes(8, "big"))
        row = BeliefDigest._ROW
        for c, r, p in zip(
            BeliefDigest._shared_column(column=competences),
            BeliefDigest._shared_column(column=learning_rates),
            BeliefDigest._float_column(values=probabilities.tolist()),
            strict=True,
        ):
            chunks.append(row + c + r + p)
        return hashlib.sha256(b"".join(chunks)).digest()

    @staticmethod
    def _shared_column(*, column: np.ndarray) -> list[bytes]:
        entry = BeliefDigest._columns.get(id(column))
        if entry is not None and entry[0]() is column:
            return entry[1]
        encoded = BeliefDigest._float_column(values=column.tolist())
        key = id(column)

        def forget(reference: weakref.ref) -> None:  # noqa: PLR0917 -- weakref calls it so
            current = BeliefDigest._columns.get(key)
            if current is not None and current[0] is reference:
                del BeliefDigest._columns[key]

        BeliefDigest._columns[key] = (weakref.ref(column, forget), encoded)
        return encoded

    @staticmethod
    def _float_column(*, values: list[float]) -> list[bytes]:
        encodings: list[bytes] = []
        for value in values:
            tag, payload = BeliefDigest._leaf(value=value)
            encodings.append(tag + len(payload).to_bytes(8, "big") + payload)
        return encodings

    @staticmethod
    def content_key(*, belief: SkillBelief) -> object:
        """Equal for two beliefs exactly when the beliefs are equal, and cheap to hash.

        The digest, where the signature reads every field; otherwise the belief itself,
        compared and hashed by pydantic as before (a weighted belief's signature leaves
        out each hypothesis's cost)."""
        if type(belief).SIGNATURE_COVERS_EVERY_FIELD:
            return BeliefDigest.of(belief=belief)
        return belief

    @staticmethod
    def digest(*, value: object) -> bytes:
        chunks: list[bytes] = []
        BeliefDigest._encode(value=value, chunks=chunks)
        return hashlib.sha256(b"".join(chunks)).digest()

    @staticmethod
    def _encode(*, value: object, chunks: list[bytes]) -> None:
        if isinstance(value, tuple):
            chunks.append(b"tuple" + len(value).to_bytes(8, "big"))
            if BeliefDigest._encode_float_rows(rows=value, chunks=chunks):
                return
            for item in value:
                BeliefDigest._encode(value=item, chunks=chunks)
            return
        if isinstance(value, BaseModel):
            chunks.append(BeliefDigest._model_encoding(model=value))
            return
        tag, payload = BeliefDigest._leaf(value=value)
        chunks.append(tag + len(payload).to_bytes(8, "big"))
        chunks.append(payload)

    # A float's encoding, by value: a weighted belief's signature is thousands of float
    # triples, and a grid's learning rates and weights repeat across hypotheses and
    # across the posteriors of one search. 0.0 and -0.0 share an entry, and an encoding.
    _float_encodings: ClassVar[dict[float, bytes]] = {}
    FLOAT_ENCODINGS_SIZE: ClassVar[int] = 1 << 20

    @staticmethod
    def _encode_float_rows(*, rows: tuple, chunks: list[bytes]) -> bool:
        """Encode `rows` -- already announced as a tuple -- if it is a tuple of tuples of
        exact floats, to the bytes the general route gives; else encode nothing."""
        for row in rows:
            if type(row) is not tuple:
                return False
            for item in row:
                if type(item) is not float:
                    return False
        known = BeliefDigest._float_encodings
        if len(known) > BeliefDigest.FLOAT_ENCODINGS_SIZE:
            known.clear()
        for row in rows:
            chunks.append(b"tuple" + len(row).to_bytes(8, "big"))
            for item in row:
                encoded = known.get(item)
                if encoded is None:
                    tag, payload = BeliefDigest._leaf(value=item)
                    encoded = known[item] = tag + len(payload).to_bytes(8, "big") + payload
                chunks.append(encoded)
        return True

    @staticmethod
    def _model_encoding(*, model: BaseModel) -> bytes:
        entry = BeliefDigest._models.get(id(model))
        if entry is not None and entry[0]() is model:
            return entry[1]
        chunks = [b"model"]
        BeliefDigest._encode(
            value=(
                "pydantic",
                type(model).__module__,
                type(model).__qualname__,
                model.model_dump_json(),
            ),
            chunks=chunks,
        )
        encoded = b"".join(chunks)
        if model.model_config.get("frozen"):
            BeliefDigest._remember(table=BeliefDigest._models, owner=model, encoded=encoded)
        return encoded

    @staticmethod
    def _leaf(*, value: object) -> tuple[bytes, bytes]:
        if isinstance(value, bytes):
            return b"bytes", value
        if isinstance(value, str):
            return b"string", value.encode("utf-8")
        if isinstance(value, bool):
            return b"bool", bytes([value])
        if isinstance(value, int):
            return b"int", str(value).encode("ascii")
        if isinstance(value, float):
            # Python's old tuple keys treat positive and negative zero equally.
            return b"float", (0.0 if value == 0 else value).hex().encode("ascii")
        if value is None:
            return b"none", b""
        raise TypeError(f"unsupported belief signature component: {type(value).__name__}")

    @staticmethod
    def _remember(*, table: dict, owner: object, encoded: bytes) -> None:
        key = id(owner)

        def forget(reference: weakref.ref) -> None:  # noqa: PLR0917 -- weakref calls it so
            entry = table.get(key)
            if entry is not None and entry[0] is reference:
                del table[key]

        table[key] = (weakref.ref(owner, forget), encoded)
