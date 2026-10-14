"""Which event advances a skill's training clock."""

from enum import Enum


class PracticeExampleSource(Enum):
    OUTCOME = "outcome"
    SAMPLER = "sampler"
