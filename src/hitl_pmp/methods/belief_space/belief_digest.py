"""A posterior's SHA-256 search identifier, computed once per object."""

import hashlib
import weakref
from typing import ClassVar

from pydantic import BaseModel

from .types.skill_belief import SkillBelief


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
        digest = BeliefDigest.digest(
            value=(type(belief).__module__, type(belief).__qualname__, belief.signature())
        )
        BeliefDigest._remember(table=BeliefDigest._digests, owner=belief, encoded=digest)
        return digest

    @staticmethod
    def digest(*, value: object) -> bytes:
        chunks: list[bytes] = []
        BeliefDigest._encode(value=value, chunks=chunks)
        return hashlib.sha256(b"".join(chunks)).digest()

    @staticmethod
    def _encode(*, value: object, chunks: list[bytes]) -> None:
        if isinstance(value, tuple):
            chunks.append(b"tuple" + len(value).to_bytes(8, "big"))
            for item in value:
                BeliefDigest._encode(value=item, chunks=chunks)
            return
        if isinstance(value, BaseModel):
            chunks.append(BeliefDigest._model_encoding(model=value))
            return
        tag, payload = BeliefDigest._leaf(value=value)
        chunks.append(tag + len(payload).to_bytes(8, "big"))
        chunks.append(payload)

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
