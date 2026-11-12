"""When the planner's learning curve steps."""

from typing import Literal

# "session": at the session boundary, where a session-fitted sampler retrains.
# "example": after every training example, for a sampler that refits on each one.
LearningUpdate = Literal["session", "example"]
