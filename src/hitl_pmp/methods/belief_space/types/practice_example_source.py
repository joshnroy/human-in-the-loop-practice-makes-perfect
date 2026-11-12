"""Which event advances a skill's training clock."""

from enum import Enum


class PracticeExampleSource(Enum):
    OUTCOME = "outcome"
    SAMPLER = "sampler"
    # A human skill: each attempt conditions its competence, but none trains it, so a
    # cycle's refit never moves it (a person's reliability does not rise with asking).
    NO_TRAINING = "no_training"
