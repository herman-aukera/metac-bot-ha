"""OpenRouter model catalog helper.

Placed in infrastructure layer (no domain dependencies). For now we keep
it minimal: allow caller to provide an *already obtained* catalog payload
so tests can inject fixtures without performing HTTP.

TODO: Extend with cached HTTP retrieval (respecting offline/DRY_RUN) once
network contract stabilized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class ModelInfo:
    name: str
    available: bool = True


def filter_available(
    requested: Sequence[str], catalog: Iterable[ModelInfo]
) -> List[str]:
    """Return subset of requested models present & available in catalog.

    Silently drops unknown/unavailable entries preserving original order.
    """
    avail = {m.name for m in catalog if m.available}
    return [m for m in requested if m in avail]


__all__ = ["ModelInfo", "filter_available"]
