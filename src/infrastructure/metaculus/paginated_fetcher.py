"""Tournament question pagination helper (infrastructure layer).

Delegates to forecasting_tools.MetaculusApi passed by caller; does not import
that package directly to keep testability (call site provides adapter).
"""
from __future__ import annotations

from typing import Any, Callable, List


def fetch_all(
    *,
    fetch_page: Callable[[int, int], List[Any]],
    limit: int,
    page_size: int,
) -> List[Any]:
    """Fetch up to limit items using page callback returning list.

    Stops early if a page returns fewer than page_size items.
    """
    collected: List[Any] = []
    page = 0
    seen_ids = set()
    while len(collected) < limit:
        items = fetch_page(page, page_size)
        if not items:
            break
        for it in items:
            qid = getattr(it, "id", None)
            if qid is not None and qid in seen_ids:
                continue
            if qid is not None:
                seen_ids.add(qid)
            collected.append(it)
            if len(collected) >= limit:
                break
        if len(items) < page_size:
            break
        page += 1
    return collected


__all__ = ["fetch_all"]
