"""Simulated time.

The simulation is discrete: the clock moves forward by `step` simulated minutes per tick.
Times are plain integers, minutes since midnight of the simulated day; values of 1440 and
above mean the next day.
"""

from __future__ import annotations

import re

MINUTES_PER_DAY = 1440
_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def parse_hhmm(value: str | int) -> int:
    """Convert "18:30" to 1110 minutes. Integers pass through unchanged.

    Hours above 23 mean the next day: "25:00" is 1:00 tomorrow.
    """
    if isinstance(value, bool):
        raise TypeError("a time cannot be a boolean")
    if isinstance(value, int):
        return value
    match = _HHMM.match(value)
    if match is None:
        raise ValueError(f"expected a time like '18:30', got {value!r}")
    hours, minutes = int(match[1]), int(match[2])
    if minutes >= 60:
        raise ValueError(f"invalid minutes in {value!r}")
    return hours * 60 + minutes


def format_hhmm(minutes: int) -> str:
    """Convert 1110 to "18:30". Times on a later day get a suffix: "01:00+1d"."""
    day, rest = divmod(minutes, MINUTES_PER_DAY)
    text = f"{rest // 60:02d}:{rest % 60:02d}"
    return text if day == 0 else f"{text}+{day}d"


class SimClock:
    """Discrete simulation clock running from `start` to `end` (inclusive)."""

    def __init__(self, start: int, end: int, step: int = 1) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        if end < start:
            raise ValueError("end must not be before start")
        self.start = start
        self.end = end
        self.step = step
        self.now = start

    @property
    def finished(self) -> bool:
        return self.now > self.end

    def advance(self) -> int:
        self.now += self.step
        return self.now

    def __str__(self) -> str:
        return format_hhmm(self.now)
