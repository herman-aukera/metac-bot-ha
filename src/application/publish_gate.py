"""Final publication gating logic.

Low-risk, pure application layer module evaluating whether a forecast
report should be published. Does not perform I/O; caller handles logging.

Criteria (initial P0 set):
 - Block if rationale contains banned phrases (case-insensitive substrings)
 - Block if binary probability exactly 0.5 and rationale has neutral phrase
 - Provide structured reason codes to support metrics & auditing

Future TODOs (non-breaking): integrate confidence normalization & entropy checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict, Any
import math


BANNED_PHRASES: Sequence[str] = (
    "neutral probability",
    "assigning neutral probability",
    "unable to generate detailed forecast",
    "the research cannot be summarized",
    "all sources failed",
)


@dataclass(slots=True)
class PublishDecision:
    publish: bool
    reason: Optional[str]
    reasons: List[str]
    blocked: bool  # structurally blocked (vs withheld for uncertainty)

    def to_dict(self) -> Dict[str, Any]:  # convenience for metrics serialization
        return {
            "publish": self.publish,
            "reason": self.reason,
            "reasons": self.reasons,
            "blocked": self.blocked,
        }


def _contains_banned(text: str) -> Optional[str]:
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def evaluate_publish(
    *,
    rationale: Optional[str],
    probabilities: Optional[Iterable[float]] = None,
    is_binary: bool = False,
    confidence: Optional[float] = None,
    min_confidence: float = 0.0,
    uniform_epsilon: float = 5e-4,
    min_options_for_uniform_block: int = 4,
    rationale_min_chars: int = 60,
    # Additional tuning params (advanced heuristics)
    min_entropy_allow: float = 0.02,  # if entropy >= this, allow even if near uniform
    min_variance_allow: float = 1e-4,  # numeric variance threshold to distinguish true uniform
    treat_low_info_as_withhold: bool = True,  # convert some blocks to withhold if borderline
    fallback_flag: Optional[bool] = None,  # external signal that distribution came from fallback repair
) -> PublishDecision:
    """Return a publish decision.

    Parameters
    ----------
    rationale: textual reasoning (may be None)
    probabilities: iterable of probabilities (0-1); None means not applicable
    is_binary: whether the question is binary (enables special neutral guard)
    confidence: optional model-reported confidence (0-1)
    min_confidence: threshold below which we WITHHOLD (not blocked) with LOW_CONFIDENCE
    """
    reasons: List[str] = []
    blocked = False

    # Phrase-based hard block
    if rationale:
        banned_hit = _contains_banned(rationale)
        if banned_hit:
            reasons.append(f"BANNED_PHRASE:{banned_hit}")
            blocked = True

    # Probability-based structural checks
    if probabilities is not None:
        probs = list(probabilities)
        # Basic normalization safeguard (non-negative, sum ~1)
        try:
            if probs and (abs(sum(probs) - 1.0) > 1e-3):
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
        except Exception:
            pass
        if len(probs) == 2:  # binary
            p0 = probs[0]
            if abs(p0 - 0.5) < 1e-9:
                # Treat as blocked if rationale missing / placeholder or contains 'neutral'
                if (not rationale or len(rationale.strip()) < rationale_min_chars or "neutral" in rationale.lower() or "fallback" in rationale.lower()):
                    reasons.append("NEUTRAL_BINARY_PLACEHOLDER")
                    blocked = True
        elif len(probs) >= 3:
            mx, mn = max(probs), min(probs)
            spread = mx - mn
            try:
                ent = -sum(p * math.log(p + 1e-12) for p in probs)
                max_ent = math.log(len(probs)) if probs else 1.0
                norm_entropy = ent / max_ent if max_ent > 0 else 0.0
            except Exception:
                ent = 0.0
                norm_entropy = 0.0

            if spread <= uniform_epsilon:
                # Uniform / near uniform detection
                uniform_detected = len(probs) >= min_options_for_uniform_block
                if uniform_detected:
                    reasons.append("UNIFORM_MC")
                    blocked = True
                else:
                    reasons.append("NEAR_UNIFORM_MC")
                    blocked = True
                # Downgrade ONLY near uniform (not perfectly uniform) if entropy acceptable
                if (not uniform_detected) and norm_entropy >= min_entropy_allow and treat_low_info_as_withhold:
                    blocked = False  # downgrade to withhold semantics
            else:
                # Low-information but not perfectly uniform
                if (max_ent - ent) < 0.01 and len(probs) >= min_options_for_uniform_block:
                    reasons.append("LOW_INFO_MC")
                    blocked = True
                    if norm_entropy >= min_entropy_allow and treat_low_info_as_withhold:
                        blocked = False

            # Fallback distribution signal (e.g., tri-point numeric converted to discrete bins)
            if fallback_flag:
                reasons.append("FALLBACK_DISTRIBUTION")
                # Only block if ALSO uniform; else treat as publishable but flagged
                if blocked and not (spread > uniform_epsilon):
                    # remain blocked; else downgrade if spread acceptable
                    pass
                else:
                    blocked = False

    # Confidence threshold (soft withhold unless already blocked)
    if confidence is not None and confidence < min_confidence and not blocked:
        reasons.append("LOW_CONFIDENCE")

    # Publication decision: allow if no reasons OR only soft reasons.
    soft_only = all(r in {"LOW_CONFIDENCE", "LOW_INFO_MC", "NEAR_UNIFORM_MC", "FALLBACK_DISTRIBUTION"} for r in reasons)
    publish = (not reasons) or (soft_only and not blocked)

    primary_reason = reasons[0] if reasons else None
    return PublishDecision(publish=publish, reason=primary_reason, reasons=reasons, blocked=blocked)


__all__ = ["PublishDecision", "evaluate_publish"]
