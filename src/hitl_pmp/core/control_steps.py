"""External control-step boundaries must survive controller failure handling."""


class ControlStepLimitReached(BaseException):
    """The external physical interaction budget has ended."""


class ControlStepObserverFailed(BaseException):
    """An instrumentation failure is not a failed robot controller."""
