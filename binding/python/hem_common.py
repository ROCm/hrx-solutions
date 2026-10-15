"""Small shared command contracts; no persistent state or lifecycle engine."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_TIMEOUT = 10.0


class HemError(Exception):
    """A user-facing failure with a stable automation exit code."""

    exit_code = 1

    def __init__(self, message: str, *, accepted: Optional[bool] = None):
        super().__init__(message)
        # True: acknowledged; False: known rejected; None: not established.
        self.accepted = accepted


class InvalidInput(HemError):
    exit_code = 2


class DeadlineExpired(HemError):
    """Deadline exceeded with no uncertain external outcome."""

    exit_code = 3


class UncertainOutcome(HemError):
    """Inspect ownership or a submitted mutation before retrying."""

    exit_code = 4


@dataclass(frozen=True)
class Deadline:
    """One monotonic budget, passed unchanged through a lifecycle command.

    Call remaining() before locks, waits and external actions. A lost mutation
    response must raise UncertainOutcome instead of DeadlineExpired, even if
    the deadline also expired. Expiry itself performs no cancellation.
    """

    expires_at: float
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def after(cls, seconds: float = DEFAULT_TIMEOUT, *,
              clock: Callable[[], float] = time.monotonic) -> Deadline:
        if not math.isfinite(seconds) or seconds <= 0:
            raise InvalidInput("timeout must be a finite positive number")
        return cls(clock() + seconds, clock)

    def remaining(self) -> float:
        remaining = self.expires_at - self.clock()
        if remaining <= 0:
            raise DeadlineExpired("deadline expired; no automatic cancellation was requested")
        return remaining


def result(state: str, *, completed: bool, **fields) -> dict:
    """Current command output only; callers never persist a result history."""
    return dict(fields, state=state, completed=completed)


def unimplemented(resource: str, command: str) -> dict:
    raise HemError(f"{resource} {command} is not implemented yet")
