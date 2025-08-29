"""Time range value object."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class TimeRange:
    """Represents a time range for predictions and questions."""

    start: datetime
    end: datetime

    def __post_init__(self):
        """Validate time range."""
        if self.start >= self.end:
            raise ValueError(
                f"Start time {self.start} must be before end time {self.end}"
            )

    @classmethod
    def from_now_plus_days(cls, days: int) -> "TimeRange":
        """Create time range from now to now + days."""
        now = datetime.now()
        return cls(start=now, end=now + timedelta(days=days))

    @classmethod
    def from_now_plus_hours(cls, hours: int) -> "TimeRange":
        """Create time range from now to now + hours."""
        now = datetime.now()
        return cls(start=now, end=now + timedelta(hours=hours))

    @classmethod
    def from_date_strings(
        cls, start_str: str, end_str: str, date_format: str = "%Y-%m-%d"
    ) -> "TimeRange":
        """Create time range from date strings."""
        start = datetime.strptime(start_str, date_format)
        end = datetime.strptime(end_str, date_format)
        return cls(start=start, end=end)

    def duration(self) -> timedelta:
        """Get the duration of the time range."""
        return self.end - self.start

    def duration_days(self) -> float:
        """Get duration in days."""
        return self.duration().total_seconds() / (24 * 3600)

    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration().total_seconds() / 3600

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within the range."""
        return self.start <= dt <= self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and other.start <= self.end

    def is_future(self, reference_time: Optional[datetime] = None) -> bool:
        """Check if the range is in the future."""
        ref = reference_time or datetime.now()
        return self.start > ref

    def is_past(self, reference_time: Optional[datetime] = None) -> bool:
        """Check if the range is in the past."""
        ref = reference_time or datetime.now()
        return self.end < ref

    def is_current(self, reference_time: Optional[datetime] = None) -> bool:
        """Check if the current time is within the range."""
        ref = reference_time or datetime.now()
        return self.contains(ref)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.start.strftime('%Y-%m-%d %H:%M')} to {self.end.strftime('%Y-%m-%d %H:%M')}"

    def __repr__(self) -> str:
        """Representation for debugging."""
        return f"TimeRange(start={self.start!r}, end={self.end!r})"
