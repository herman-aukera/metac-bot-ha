"""Subject extraction heuristics for research queries.

Deliberately simple & deterministic; future improvements can add NLP.
"""

from __future__ import annotations

import re
from typing import List, Dict

LEADING_WORDS = {
    "will",
    "who",
    "which",
    "what",
    "is",
    "are",
    "could",
    "should",
}

STOPWORDS = {"the", "of", "to", "on", "in", "a", "an", "be", "before"}
SYNONYMS = {"reelected": "reelection"}


def extract_subject(question: str) -> Dict[str, str]:
    original = question.strip()
    core = original.rstrip("? ")
    tokens = re.split(r"\s+", core)
    simplified_parts: List[str] = []
    for t in tokens:
        low = t.lower()
        if not simplified_parts and low in LEADING_WORDS:
            continue
        word = SYNONYMS.get(low, low)
        if word in STOPWORDS:
            continue
        # Strip year-only trailing segment heuristically (keep if mid-sentence)
        if (
            re.fullmatch(r"20\d{2}", word)
            and len(tokens) > 3
            and tokens[-1].rstrip("? ") == t
        ):
            # keep the year if earlier token indicates election context
            if any(s in simplified_parts for s in ("election", "reelection")):
                simplified_parts.append(word)
            continue
        simplified_parts.append(word)
    simplified = " ".join(simplified_parts)
    expanded = simplified  # placeholder for future synonym expansion
    return {"raw": original, "simplified": simplified, "expanded": expanded}


__all__ = ["extract_subject"]
