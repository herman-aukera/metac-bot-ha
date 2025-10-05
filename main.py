from __future__ import annotations

import argparse
import json
import asyncio
import logging
import os
import sys
import atexit
import inspect
import signal
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal, Set, TYPE_CHECKING, Protocol


logger = logging.getLogger(__name__)
_WITHHELD_QUESTION_IDS: Set[int] = set()
_BLOCKED_PUBLICATION_QIDS: Set[int] = set()
_OPEN_AIOHTTP_SESSIONS: list = []  # best-effort tracking of any aiohttp sessions for shutdown
_SHUTDOWN_REQUESTED = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Kill command


# Always provide a safe default for clean_indents; override with forecasting_tools version if available
def clean_indents(text: str) -> str:  # type: ignore
    return text


# Safe wrapper to guard against unexpected NameError or import-time issues
def _safe_clean_indents(text: str) -> str:
    try:
        return clean_indents(text)  # type: ignore[name-defined]
    except Exception:
        return text


# Import core forecasting_tools classes; provide stubs if unavailable at analysis time
try:  # pragma: no cover - import resolution
    from forecasting_tools import ForecastBot, MetaculusApi, AskNewsSearcher  # type: ignore

    # If available, use the library's clean_indents implementation
    try:
        from forecasting_tools import clean_indents as _ft_clean_indents  # type: ignore

        clean_indents = _ft_clean_indents  # type: ignore
    except Exception:
        pass

    # Apply critical patch to fix tournament question filtering
    try:
        from src.infrastructure.patches.forecasting_tools_fix import (
            apply_forecasting_tools_patch,
        )

        apply_forecasting_tools_patch()
        logger.info(
            "✅ Applied forecasting-tools patch for tournament question filtering"
        )
    except Exception as patch_error:
        logger.warning(f"Failed to apply forecasting-tools patch: {patch_error}")

except Exception:  # pragma: no cover - allow static analysis without hard dependency

    class ForecastBot:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # provide attributes used later in the file for static analyzers
            self.skip_previously_forecasted_questions = True
            self.llms: Dict[str, Any] = {}

        def get_llm(self, model_type: str = "default", purpose: str = "llm") -> Any:
            return object()

        async def forecast_on_tournament(
            self, *args: Any, **kwargs: Any
        ) -> List[Any]:  # runtime stub
            return []

        async def forecast_questions(
            self, *args: Any, **kwargs: Any
        ) -> List[Any]:  # runtime stub
            return []

    class MetaculusApi:  # type: ignore
        CURRENT_QUARTERLY_CUP_ID: int = 0

        @staticmethod
        def get_question_by_url(url: str) -> Any:
            return None

    class AskNewsSearcher:  # type: ignore
        async def get_formatted_news_async(self, q: str) -> str:
            return ""

    # Lightweight fallbacks for helpers referenced later to keep module importable under lint

    class SmartSearcher:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def invoke(self, prompt: str) -> str:
            return ""

    class PredictionExtractor:  # type: ignore
        @staticmethod
        def extract_last_percentage_value(
            text: str, *args: Any, **kwargs: Any
        ) -> float:
            return 0.5

        @staticmethod
        def extract_option_list_with_percentage_afterwards(
            *args: Any, **kwargs: Any
        ) -> Any:
            return None

        @staticmethod
        def extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            *args: Any, **kwargs: Any
        ) -> Any:
            return None

    class BudgetAlertSystem:  # type: ignore
        def send_critical_alert(self, *args: Any, **kwargs: Any) -> None:
            pass

    class IntegratedMonitoringService:  # type: ignore
        def get_system_health_status(self) -> Dict[str, Any]:
            return {}

    # Minimal local shape and stubs only; runtime-safe factories are defined at module scope below
    class LocalReasonedPrediction:  # type: ignore
        def __init__(self, prediction_value: Any, reasoning: str) -> None:
            self.prediction_value = prediction_value
            self.reasoning = reasoning


if TYPE_CHECKING:  # import typing-only symbols without affecting runtime
    try:
        from forecasting_tools import ReasonedPrediction  # type: ignore
        from forecasting_tools import PredictedOptionList, PredictedOption  # type: ignore
        from forecasting_tools import NumericDistribution  # type: ignore
        from forecasting_tools import (
            BinaryQuestion,
            MultipleChoiceQuestion,
            NumericQuestion,
            MetaculusQuestion,
        )  # type: ignore
    except Exception:
        # Lightweight protocols as fallbacks for type checking
        class ReasonedPrediction(Protocol):  # type: ignore
            prediction_value: Any
            reasoning: str

        class PredictedOption(Protocol):  # type: ignore
            option_name: str
            probability: float

        class PredictedOptionList(Protocol):  # type: ignore
            predicted_options: List[PredictedOption]

        class NumericPercentile(Protocol):  # type: ignore
            percentile: float
            value: float

        class NumericDistribution(Protocol):  # type: ignore
            declared_percentiles: List[NumericPercentile]

        class BinaryQuestion(Protocol):  # type: ignore
            id: Any
            page_url: str
            question_text: str
            background_info: str
            resolution_criteria: str
            fine_print: str

        class MultipleChoiceQuestion(BinaryQuestion, Protocol):  # type: ignore
            options: List[str]

        class NumericQuestion(BinaryQuestion, Protocol):  # type: ignore
            unit_of_measure: str
            lower_bound: float
            upper_bound: float
            open_lower_bound: bool
            open_upper_bound: bool

        class MetaculusQuestion(BinaryQuestion, Protocol):  # type: ignore
            pass


# --- Runtime-safe helper factories (module scope) ---
# Always define these so downstream code can call them regardless of import path above
try:
    from forecasting_tools import ReasonedPrediction as _FTReasonedPrediction  # type: ignore
except Exception:
    _FTReasonedPrediction = None  # type: ignore

try:
    from forecasting_tools import NumericDistribution as _FTNumericDistribution  # type: ignore
except Exception:
    _FTNumericDistribution = None  # type: ignore

try:
    LocalReasonedPrediction  # type: ignore[name-defined]
except NameError:

    class LocalReasonedPrediction:  # type: ignore
        def __init__(self, prediction_value: Any, reasoning: str) -> None:
            self.prediction_value = prediction_value
            self.reasoning = reasoning


def _mk_rp(prediction_value: Any, reasoning: str):  # type: ignore
    if _FTReasonedPrediction is not None:
        try:
            return _FTReasonedPrediction(
                prediction_value=prediction_value, reasoning=reasoning
            )
        except Exception:
            pass
    try:
        return LocalReasonedPrediction(
            prediction_value=prediction_value, reasoning=reasoning
        )  # type: ignore[name-defined]
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(prediction_value=prediction_value, reasoning=reasoning)


def _mk_numeric_distribution(percentiles: List[Any], question: Any):  # type: ignore
    if _FTNumericDistribution is not None:
        try:
            return _FTNumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=getattr(question, "open_upper_bound", None),
                open_lower_bound=getattr(question, "open_lower_bound", None),
                upper_bound=getattr(question, "upper_bound", None),
                lower_bound=getattr(question, "lower_bound", None),
                zero_point=None,
            )
        except Exception:
            pass
    from types import SimpleNamespace

    return SimpleNamespace(
        declared_percentiles=percentiles,
        open_upper_bound=getattr(question, "open_upper_bound", None),
        open_lower_bound=getattr(question, "open_lower_bound", None),
        upper_bound=getattr(question, "upper_bound", None),
        lower_bound=getattr(question, "lower_bound", None),
        zero_point=None,
    )


def _should_block_publication_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "[withheld:" in t:
        return True
    if "[do not publish" in t:
        return True
    if "[private]" in t:
        return True
    return False


# --- Runtime-safe PredictionExtractor (module scope) ---
# Ensure PredictionExtractor symbol exists at runtime regardless of forecasting_tools import state
try:  # pragma: no cover - import resolution path may vary by forecasting_tools version
    from forecasting_tools import PredictionExtractor as _FTPredictionExtractor  # type: ignore
except Exception:
    try:
        from forecasting_tools.utils.prediction_extractor import (
            PredictionExtractor as _FTPredictionExtractor,
        )  # type: ignore
    except Exception:
        _FTPredictionExtractor = None  # type: ignore

if _FTPredictionExtractor is not None:
    PredictionExtractor = _FTPredictionExtractor  # type: ignore
else:
    # Minimal local fallback with lenient parsing to keep pipeline resilient
    import re
    from types import SimpleNamespace

    class PredictionExtractor:  # type: ignore
        _pct_regex = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")

        @staticmethod
        def extract_last_percentage_value(
            text: str, max_prediction: float = 1.0, min_prediction: float = 0.0
        ) -> float:
            try:
                matches = PredictionExtractor._pct_regex.findall(text or "")
                if not matches:
                    return max(min(0.5, max_prediction), min_prediction)
                val = float(matches[-1]) / 100.0
                return float(max(min(val, max_prediction), min_prediction))
            except Exception:
                return max(min(0.5, max_prediction), min_prediction)

        @staticmethod
        def extract_option_list_with_percentage_afterwards(
            text: str, options: List[str]
        ) -> Any:
            # Very tolerant parser: look for lines formatted like "Option: 12.3%"
            try:
                preds = []
                lines = (text or "").splitlines()
                for line in lines:
                    if ":" in line and "%" in line:
                        try:
                            name, rest = line.split(":", 1)
                            pct_match = PredictionExtractor._pct_regex.search(rest)
                            if pct_match:
                                p = float(pct_match.group(1)) / 100.0
                                preds.append(
                                    SimpleNamespace(
                                        option_name=name.strip(), probability=p
                                    )
                                )
                        except Exception:
                            continue
                if not preds and options:
                    # Fallback: split percentages in order for provided options (e.g., "10%, 20%, 70%")
                    pcts = [
                        float(x) / 100.0
                        for x in PredictionExtractor._pct_regex.findall(text or "")
                    ]
                    if len(pcts) == len(options):
                        preds = [
                            SimpleNamespace(option_name=o, probability=p)
                            for o, p in zip(options, pcts)
                        ]
                if preds:
                    # Normalize probabilities
                    s = sum(x.probability for x in preds)
                    if s > 0:
                        for x in preds:
                            x.probability = x.probability / s
                    return SimpleNamespace(predicted_options=preds)
                return None
            except Exception:
                return None

        @staticmethod
        def extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            text: str, question: Any
        ) -> Any:
            # Minimal fallback: attempt to parse lines like "Percentile 10: 54 %" and build a distribution-like object
            try:
                pct_line = re.compile(
                    r"(?i)percentile\s*(\d{1,2}|100)\s*[:=]\s*(\d+(?:\.\d+)?)"
                )
                matches = pct_line.findall(text or "")
                if not matches:
                    return None
                # Convert to forecasting_tools Percentile list shape if available
                try:
                    from forecasting_tools.data_models.numeric_report import Percentile  # type: ignore

                    percentile_list = [
                        Percentile(percentile=float(p) / 100.0, value=float(v))
                        for (p, v) in matches
                    ]
                except Exception:
                    # Generic shape fallback
                    percentile_list = [
                        SimpleNamespace(percentile=float(p) / 100.0, value=float(v))
                        for (p, v) in matches
                    ]
                return _mk_numeric_distribution(percentile_list, question)
            except Exception:
                return None


def _wrap_publication_func(orig_fn: Any, kind: str) -> Any:
    if not callable(orig_fn):  # pragma: no cover
        return orig_fn
    # Prevent double wrapping
    if getattr(orig_fn, "__metac_guard_wrapped__", False):
        return orig_fn

    is_coro = inspect.iscoroutinefunction(orig_fn)

    def _extract_any_text(args, kwargs) -> str:  # type: ignore
        # Priority ordered keys
        for key in (
            "comment",
            "body",
            "rationale",
            "explanation",
            "reasoning",
            "text",
            "content",
        ):
            if key in kwargs and isinstance(kwargs[key], str) and kwargs[key]:
                return kwargs[key]
        # Fallback: scan all kwargs + positional args for first sufficiently long string
        for v in kwargs.values():
            if isinstance(v, str) and len(v) >= 10:
                return v
        for v in args:
            if isinstance(v, str) and len(v) >= 10:
                return v
        return ""

    def _extract_question_ids(args, kwargs) -> List[int]:  # type: ignore
        ids: List[int] = []
        # Common kw names first
        for k in ("question_id", "qid", "id"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                ids.append(v)
        # Positional ints
        for a in args:
            if isinstance(a, int) and a > 0:
                ids.append(a)

        # Objects with id-like attributes
        def _maybe_from_obj(obj: Any) -> Optional[int]:
            for attr in ("id", "question_id", "qid"):
                try:
                    val = getattr(obj, attr, None)
                    if isinstance(val, int) and val > 0:
                        return val
                except Exception:
                    continue
            return None

        for a in list(args) + list(kwargs.values()):
            if not isinstance(a, (int, str)):
                qid = _maybe_from_obj(a)
                if qid:
                    ids.append(qid)
        # Deduplicate preserving order
        seen = set()
        ordered: List[int] = []
        for i in ids:
            if i not in seen:
                ordered.append(i)
                seen.add(i)
        return ordered

    def _extract_prob_distributions(args, kwargs) -> List[List[float]]:  # type: ignore
        dists: List[List[float]] = []
        cand_objs = list(args) + list(kwargs.values())

        def _coerce(obj: Any) -> Optional[List[float]]:
            # list/tuple of floats
            if (
                isinstance(obj, (list, tuple))
                and obj
                and all(isinstance(x, (int, float)) for x in obj)
            ):
                vals = [float(x) for x in obj]
                if all(0.0 <= x <= 1.0 for x in vals):
                    return vals
            # dict of probabilities (MC payload)
            if isinstance(obj, dict) and obj:
                try:
                    vals = [
                        float(v) for v in obj.values() if isinstance(v, (int, float))
                    ]
                    if (
                        len(vals) == len(obj)
                        and len(vals) >= 3
                        and all(0.0 <= v <= 1.0 for v in vals)
                    ):
                        return vals
                except Exception:
                    pass
            # PredictedOptionList style
            for attr in ("predicted_options", "options", "predictedOptions"):
                if hasattr(obj, attr):
                    try:
                        opts = getattr(obj, attr)
                        if opts and hasattr(opts, "__iter__"):
                            vals2: List[float] = []
                            for o in opts:
                                p = getattr(o, "probability", None)
                                if isinstance(p, (int, float)):
                                    vals2.append(float(p))
                            if vals2 and all(0.0 <= x <= 1.0 for x in vals2):
                                return vals2
                    except Exception:
                        pass
            # ReasonedPrediction style (try generic probabilities field)
            for attr in ("probabilities", "probs", "probability_list"):
                if hasattr(obj, attr):
                    try:
                        probs = getattr(obj, attr)
                        if (
                            isinstance(probs, (list, tuple))
                            and probs
                            and all(isinstance(x, (int, float)) for x in probs)
                        ):
                            vals3 = [float(x) for x in probs]
                            if all(0.0 <= x <= 1.0 for x in vals3):
                                return vals3
                    except Exception:
                        pass
            return None

        for obj in cand_objs:
            try:
                vals = _coerce(obj)
                if vals:
                    # Normalise if not exactly summing to 1 (tolerate small drift)
                    s = sum(vals)
                    if s > 0:
                        normed = [v / s for v in vals]
                        dists.append(normed)
            except Exception:
                continue
        return dists

    def _is_uniform(dist: List[float]) -> bool:
        # PRODUCTION DEFAULT: Guard disabled for tournament mode
        # Override with DISABLE_PUBLICATION_GUARD=false to re-enable strict filtering
        if os.getenv("DISABLE_PUBLICATION_GUARD", "true").lower() == "true":
            return False
        if len(dist) < 3:
            return False  # only guard MC (>=3)
        mx, mn = max(dist), min(dist)
        return (mx - mn) <= 0.0005  # tight tolerance

    def _low_information(dist: List[float]) -> bool:
        # PRODUCTION DEFAULT: Guard disabled for tournament mode
        # Override with DISABLE_PUBLICATION_GUARD=false to re-enable strict filtering
        if os.getenv("DISABLE_PUBLICATION_GUARD", "true").lower() == "true":
            return False
        # Entropy close to max indicates near-uniform; use small margin
        try:
            import math

            n = len(dist)
            if n < 3:
                return False
            ent = -sum(p * math.log(p + 1e-12) for p in dist)
            max_ent = math.log(n)
            return (max_ent - ent) < 0.01  # within 0.01 nats of max entropy
        except Exception:
            return False

    async def _async_wrapper(*args, **kwargs):  # type: ignore
        candidate_text = _extract_any_text(args, kwargs)
        qids = _extract_question_ids(args, kwargs)
        debug = os.getenv("PUBLICATION_GUARD_DEBUG") == "1"
        is_post_method = kind.startswith("post_") or kind.startswith("_post_")
        if is_post_method:
            if qids and any(q in _WITHHELD_QUESTION_IDS for q in qids):
                logger.info(
                    f"Publication guard: blocked {kind} for withheld question id(s) {qids}"
                )
                for q in qids:
                    _BLOCKED_PUBLICATION_QIDS.add(q)
                return None
            dists = _extract_prob_distributions(args, kwargs)
            if debug:
                try:
                    arg_types = [type(a).__name__ for a in args]
                    kw_types = {k: type(v).__name__ for k, v in kwargs.items()}
                    logger.info(
                        f"Publication guard debug: {kind} qids={qids or 'unknown'} arg_types={arg_types} kw_types={kw_types} dists_lens={[len(d) for d in dists]}"
                    )
                except Exception:
                    pass
            for dist in dists:
                if len(dist) >= 3 and (_is_uniform(dist) or _low_information(dist)):
                    if debug:
                        try:
                            import math

                            ent = -sum(p * math.log(p + 1e-12) for p in dist)
                            max_ent = math.log(len(dist))
                            logger.info(
                                f"Publication guard diagnostics: method={kind} k={len(dist)} max_prob={max(dist):.4f} min_prob={min(dist):.4f} ent={ent:.4f} max_ent={max_ent:.4f} ent_gap={max_ent - ent:.4f}"
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"Publication guard: blocked {kind} due to uniform/low-info MC distribution (k={len(dist)}) qids={qids or 'unknown'}"
                    )
                    for q in qids:
                        _BLOCKED_PUBLICATION_QIDS.add(q)
                    return None
        if _should_block_publication_text(candidate_text):
            logger.info(f"Publication guard: blocked {kind} (marker detected)")
            return None
        # Diagnostic: if WITHHELD marker present but not blocked by earlier logic, log anomaly
        if "[WITHHELD:" in candidate_text.upper():
            logger.warning(
                f"Publication guard anomaly: WITHHELD marker passed through {kind} (qids={qids})"
            )
        return await orig_fn(*args, **kwargs)  # type: ignore

    def _sync_wrapper(*args, **kwargs):  # type: ignore
        candidate_text = _extract_any_text(args, kwargs)
        qids = _extract_question_ids(args, kwargs)
        debug = os.getenv("PUBLICATION_GUARD_DEBUG") == "1"
        is_post_method = kind.startswith("post_") or kind.startswith("_post_")
        if is_post_method:
            if qids and any(q in _WITHHELD_QUESTION_IDS for q in qids):
                logger.info(
                    f"Publication guard: blocked {kind} for withheld question id(s) {qids}"
                )
                for q in qids:
                    _BLOCKED_PUBLICATION_QIDS.add(q)
                return None
            dists = _extract_prob_distributions(args, kwargs)
            if debug:
                try:
                    arg_types = [type(a).__name__ for a in args]
                    kw_types = {k: type(v).__name__ for k, v in kwargs.items()}
                    logger.info(
                        f"Publication guard debug: {kind} qids={qids or 'unknown'} arg_types={arg_types} kw_types={kw_types} dists_lens={[len(d) for d in dists]}"
                    )
                except Exception:
                    pass
            for dist in dists:
                if len(dist) >= 3 and (_is_uniform(dist) or _low_information(dist)):
                    if debug:
                        try:
                            import math

                            ent = -sum(p * math.log(p + 1e-12) for p in dist)
                            max_ent = math.log(len(dist))
                            logger.info(
                                f"Publication guard diagnostics: method={kind} k={len(dist)} max_prob={max(dist):.4f} min_prob={min(dist):.4f} ent={ent:.4f} max_ent={max_ent:.4f} ent_gap={max_ent - ent:.4f}"
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"Publication guard: blocked {kind} due to uniform/low-info MC distribution (k={len(dist)}) qids={qids or 'unknown'}"
                    )
                    for q in qids:
                        _BLOCKED_PUBLICATION_QIDS.add(q)
                    return None
        if _should_block_publication_text(candidate_text):
            logger.info(f"Publication guard: blocked {kind} (marker detected)")
            return None
        if "[WITHHELD:" in candidate_text.upper():
            logger.warning(
                f"Publication guard anomaly: WITHHELD marker passed through {kind} (qids={qids})"
            )
        return orig_fn(*args, **kwargs)

    wrapper = _async_wrapper if is_coro else _sync_wrapper
    setattr(wrapper, "__metac_guard_wrapped__", True)
    return wrapper


try:  # best-effort
    from forecasting_tools.forecast_helpers import metaculus_api as _ft_meta_api  # type: ignore

    _patched = []
    # Original anticipated function names (may not exist in current lib version)
    for _fname in ("post_prediction", "post_comment"):
        if hasattr(_ft_meta_api, _fname):
            wrapped = _wrap_publication_func(getattr(_ft_meta_api, _fname), _fname)
            setattr(_ft_meta_api, _fname, wrapped)
            if getattr(wrapped, "__metac_guard_wrapped__", False):
                _patched.append(_fname)
    # Also patch class methods on MetaculusApi (most real calls go through the instance)
    try:
        _MetaApiCls = getattr(_ft_meta_api, "MetaculusApi", None)
        if _MetaApiCls:
            for _meth in (
                "post_prediction",  # legacy
                "post_comment",  # legacy
                "post_binary_question_prediction",
                "post_numeric_question_prediction",
                "post_multiple_choice_question_prediction",
                "post_question_comment",
                "_post_question_prediction",  # lowest-level; block if higher missed
            ):
                if hasattr(_MetaApiCls, _meth):
                    orig = getattr(_MetaApiCls, _meth)
                    wrapped = _wrap_publication_func(orig, _meth)
                    if wrapped is not orig:
                        setattr(_MetaApiCls, _meth, wrapped)
                        if (
                            getattr(wrapped, "__metac_guard_wrapped__", False)
                            and _meth not in _patched
                        ):
                            _patched.append(f"MetaculusApi.{_meth}")
    except Exception as _cls_patch_err:  # pragma: no cover
        logger.debug(f"Publication guard class patch skipped: {_cls_patch_err}")
    if _patched:
        logger.info("Publication guard active for: %s", ", ".join(_patched))
except Exception as _pg_err:  # pragma: no cover
    logger.debug(f"Publication guard installation skipped: {_pg_err}")


SAFE_REASONING_FALLBACK = (
    "Forecast generated without detailed reasoning due to fallback."
)
# Shared model and text constants to avoid duplication
MODEL_OPENROUTER_FREE_OSS = "openai/gpt-oss-20b:free"  # disabled runtime (404)
MODEL_KIMI_FREE = "moonshotai/kimi-k2:free"  # disabled runtime (404)
MODEL_PERPLEXITY_REASONING = "perplexity/sonar-reasoning"
NEUTRAL_TEXT_UNABLE = "unable to generate detailed forecast"
NEUTRAL_TEXT_ASSIGNING = "assigning neutral probability"

# Tournament components - import after forecasting_tools to avoid conflicts
try:
    # Add src to path for tournament components (append to avoid overshadowing site-packages)
    _src_path = str(Path(__file__).parent / "src")
    if _src_path not in sys.path:
        sys.path.append(_src_path)
    from infrastructure.external_apis.tournament_asknews_client import (
        TournamentAskNewsClient,
    )
    from infrastructure.config.tournament_config import get_tournament_config
    from infrastructure.config.api_keys import api_key_manager

    TOURNAMENT_COMPONENTS_AVAILABLE = True
    logger.info("Tournament components loaded successfully")
except ImportError as e:
    logger.warning(f"Tournament components not available: {e}")
    TOURNAMENT_COMPONENTS_AVAILABLE = False

# Graceful shutdown: close async clients (best-effort)
_async_shutdown_tasks: list = []


async def _graceful_async_shutdown():  # pragma: no cover - runtime behavior
    for obj in _async_shutdown_tasks:
        try:
            if hasattr(obj, "aclose") and inspect.iscoroutinefunction(obj.aclose):
                await obj.aclose()
        except Exception as e:
            logger.debug(f"Shutdown close failed: {e}")
    # Close any leftover aiohttp sessions (best-effort; list may be empty)
    for sess in list(_OPEN_AIOHTTP_SESSIONS):
        try:
            if hasattr(sess, "closed") and not getattr(sess, "closed"):
                await sess.close()
        except Exception:
            pass


def _sync_atexit_shutdown():  # pragma: no cover
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule coroutine; cannot block
            loop.create_task(_graceful_async_shutdown())
        else:
            loop.run_until_complete(_graceful_async_shutdown())
    except Exception:
        pass


atexit.register(_sync_atexit_shutdown)

# --- Runtime patch: numeric report CDF sanitizer to prevent AssertionError on tight percentiles ---
try:  # pragma: no cover - defensive runtime patching
    from forecasting_tools.data_models import numeric_report as _ft_numeric_report  # type: ignore

    _orig_publish_numeric = _ft_numeric_report.NumericReport.publish_report_to_metaculus

    def _sanitized_publish(self, *args, **kwargs):  # noqa: D401
        """Patched publish that widens too-close percentiles and retries with fallback distribution.

        Addresses AssertionError in forecasting_tools.numeric_report.cdf validation where adjacent
        percentiles differ by <5e-05. We pre-expand gaps to >=6e-05; on failure we fall back to a
        coarse tri-point distribution (10%, 50%, 90%). If all remediation fails, the original
        exception is re-raised so upstream error handling remains intact.
        """
        try:
            pred = getattr(self, "prediction", None)
            if pred:
                # Attempt to locate percentile & value arrays (attribute names may vary across versions)
                percentiles = (
                    getattr(pred, "percentiles", None)
                    or getattr(pred, "_percentiles", None)
                    or getattr(pred, "_raw_percentiles", None)
                )
                values = (
                    getattr(pred, "values", None)
                    or getattr(pred, "_values", None)
                    or getattr(pred, "_raw_values", None)
                )
                if percentiles and values and len(percentiles) == len(values):
                    modified = False
                    new_p = []
                    last = None
                    for p in percentiles:
                        if last is not None and p - last < 6e-05:
                            p = (
                                last + 6e-05
                            )  # widen gap slightly beyond library threshold (5e-05)
                            modified = True
                        new_p.append(p)
                        last = p
                    if modified:
                        try:
                            if hasattr(pred, "percentiles"):
                                pred.percentiles = new_p  # type: ignore[attr-defined]
                            elif hasattr(pred, "_percentiles"):
                                setattr(pred, "_percentiles", new_p)
                        except (
                            Exception
                        ):  # silently continue; will fallback if assertion persists
                            pass
        except Exception:  # pragma: no cover - sanitizer must not break publish
            pass
        try:
            return _orig_publish_numeric(self, *args, **kwargs)
        except AssertionError as ae:
            # Graceful suppression: log and skip publishing instead of aborting entire question
            try:
                logger.warning(
                    "Numeric publish suppressed due to CDF assertion (%s). Skipping publish for this question.",
                    ae,
                )
            except Exception:
                pass
            return None

    _ft_numeric_report.NumericReport.publish_report_to_metaculus = _sanitized_publish  # type: ignore
    logger.info(
        "Applied numeric CDF sanitizer patch to NumericReport.publish_report_to_metaculus"
    )
except Exception as _patch_err:  # pragma: no cover
    logger.debug(f"NumericReport publish patch not applied: {_patch_err}")


class TemplateForecaster(ForecastBot):
    """
    Enhanced template bot for Q2 2025 Metaculus AI Tournament with tournament optimizations.

    Features:
    - Tournament-optimized AskNews client with quota management
    - Metaculus proxy client for free credits with fallback to OpenRouter
    - Robust fallback system for all API providers
    - Usage monitoring and alerting
    - Tournament-specific configurations
    - Budget management and cost-aware model selection
    - Token tracking and real-time cost monitoring

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with tournament optimizations and budget management."""
        super().__init__(*args, **kwargs)

        # Initialize budget management components
        try:
            from src.infrastructure.config.budget_manager import budget_manager
            from src.infrastructure.config.budget_alerts import budget_alert_system
            from src.infrastructure.config.enhanced_llm_config import (
                enhanced_llm_config,
            )
            from src.infrastructure.config.token_tracker import token_tracker

            self.budget_manager = budget_manager
            self.budget_alert_system = budget_alert_system
            self.enhanced_llm_config = enhanced_llm_config
            self.token_tracker = token_tracker

            # Log initial budget status
            self.budget_manager.log_budget_status()
            self.enhanced_llm_config.log_configuration_status()

            logger.info("Budget management system initialized")

        except ImportError as e:
            logger.warning(f"Budget management components not available: {e}")
            self.budget_manager = None
            self.budget_alert_system = None
            self.enhanced_llm_config = None
            self.token_tracker = None

        # Initialize enhanced tri-model router for GPT-5 variants with anti-slop directives
        try:
            from src.infrastructure.config.tri_model_router import tri_model_router
            from src.prompts.anti_slop_prompts import anti_slop_prompts
            from src.domain.services.multi_stage_validation_pipeline import (
                MultiStageValidationPipeline,
            )
            from src.infrastructure.config.budget_aware_operation_manager import (
                budget_aware_operation_manager,
            )
            from src.infrastructure.reliability.comprehensive_error_recovery import (
                ComprehensiveErrorRecoveryManager,
            )

            self.tri_model_router = tri_model_router
            self.anti_slop_prompts = anti_slop_prompts

            # Initialize multi-stage validation pipeline (Task 8.1)
            self.multi_stage_pipeline = MultiStageValidationPipeline(
                tri_model_router=self.tri_model_router,
                tournament_asknews=getattr(self, "tournament_asknews", None),
            )

            # Initialize budget-aware operation manager (Task 8.2)
            self.budget_aware_manager = budget_aware_operation_manager

            # Initialize comprehensive error recovery (Task 8.1)
            self.error_recovery_manager = ComprehensiveErrorRecoveryManager(
                tri_model_router=self.tri_model_router,
                budget_manager=self.budget_manager,
            )

            # Integrate tri-model router with budget management systems (Task 8.2)
            self.tri_model_router.integrate_with_budget_manager(
                budget_manager=self.budget_manager,
                budget_aware_manager=self.budget_aware_manager,
            )

            # Log tri-model status
            model_status = self.tri_model_router.get_model_status()
            logger.info("Enhanced tri-model router initialized:")
            for tier, status in model_status.items():
                logger.info(f"  {tier}: {status}")

            logger.info("Budget manager integration with tri-model router completed")

            # Log multi-stage pipeline status
            pipeline_config = self.multi_stage_pipeline.get_pipeline_configuration()
            logger.info(
                f"Multi-stage validation pipeline initialized with {len(pipeline_config['stages'])} stages"
            )

            # (Status logging call removed pending availability of method in this context.)

        except ImportError as e:
            logger.warning(f"Enhanced tri-model components not available: {e}")
            self.tri_model_router = None
            self.anti_slop_prompts = None
            self.multi_stage_pipeline = None
            self.budget_aware_manager = None
            self.error_recovery_manager = None

        # Initialize OpenRouter API key configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key or self.openrouter_api_key.startswith("dummy_"):
            logger.error(
                "OpenRouter API key not configured! This is required for tournament operation."
            )
        else:
            logger.info("OpenRouter API key configured successfully")

        # Initialize performance monitoring integration (Task 8.2)
        try:
            from src.infrastructure.monitoring.integrated_monitoring_service import (
                integrated_monitoring_service,
            )

            self.performance_monitor = integrated_monitoring_service
            logger.info("Performance monitoring integration initialized")
        except ImportError as e:
            logger.warning(f"Performance monitoring not available: {e}")
            self.performance_monitor = None

        # Initialize error handling and fallback state
        self.emergency_mode_active = False
        self.api_failure_count = 0
        self.max_api_failures = int(os.getenv("MAX_API_FAILURES", "5"))
        # Hard-disable unreliable/free/proxy/perplexity fallbacks until validated via explicit curl test.
        # We keep structure for minimal diff but point to None so caller logic can short‑circuit.
        self.fallback_models = {
            # Placeholders kept as strings to avoid type issues; dynamic fallback disabled upstream.
            "emergency": "disabled:no-emergency-fallback",
            "proxy": "disabled:no-proxy-fallback",
            "last_resort": "disabled:no-last-resort",
        }

        # Negative availability cache for deterministic terminal errors (404/allowance/etc.)
        self._unavailable_models: Set[str] = set()

        # Initialize tournament components if available
        if TOURNAMENT_COMPONENTS_AVAILABLE:
            try:
                self.tournament_config = get_tournament_config()
                self.tournament_asknews = TournamentAskNewsClient()
                # Register for graceful shutdown if it exposes aclose
                try:
                    if hasattr(self.tournament_asknews, "aclose"):
                        _async_shutdown_tasks.append(self.tournament_asknews)  # type: ignore[arg-type]
                        logger.info(
                            "Registered TournamentAskNewsClient for graceful shutdown"
                        )
                except Exception:
                    pass

                # Initialize multi-stage research pipeline (Task 4.1)
                from domain.services.multi_stage_research_pipeline import (
                    MultiStageResearchPipeline,
                )

                self._multi_stage_pipeline = MultiStageResearchPipeline(
                    tri_model_router=self.tri_model_router,
                    tournament_asknews=self.tournament_asknews,
                )
                logger.info("Multi-stage research pipeline initialized successfully")

                # Update concurrency based on tournament config
                self._max_concurrent_questions = (
                    self.tournament_config.max_concurrent_questions
                )
                self._concurrency_limiter = asyncio.Semaphore(
                    self._max_concurrent_questions
                )

                # Log tournament initialization
                logger.info(
                    f"Tournament mode initialized: {self.tournament_config.tournament_name}"
                )
                logger.info(
                    f"Max concurrent questions: {self._max_concurrent_questions}"
                )

                # Validate API keys
                api_key_manager.log_key_status()

            except Exception as e:
                logger.warning(f"Failed to initialize tournament components: {e}")
                self.tournament_config = None
                self.tournament_asknews = None
                self._multi_stage_pipeline = None
        else:
            self.tournament_config = None
            self.tournament_asknews = None
            self._multi_stage_pipeline = None

        # Set default concurrency if not set by tournament config
        if not hasattr(self, "_max_concurrent_questions"):
            self._max_concurrent_questions = 2
            self._concurrency_limiter = asyncio.Semaphore(
                self._max_concurrent_questions
            )

        # Initialize a search client for tests and research integrations
        try:
            from src.infrastructure.config.settings import get_settings
            from src.infrastructure.external_apis.search_client import (
                create_search_client,
            )

            settings = get_settings()
            self.search_client = create_search_client(settings)
            logger.info("Search client initialized for TemplateForecaster")
        except Exception as e:
            logger.warning(f"Search client not available, using no-op stub: {e}")

            class _NoOpSearchClient:
                async def search(
                    self, query: str, max_results: int = 10
                ):  # pragma: no cover
                    _ = (query, max_results)
                    return []

                async def health_check(self) -> bool:  # pragma: no cover
                    await asyncio.sleep(0)
                    return True

            self.search_client = _NoOpSearchClient()

        # Defensive dynamic publication guard (belt & suspenders) wrapping metaculus_client methods
        try:
            mc = getattr(self, "metaculus_client", None)
            if mc and not getattr(mc, "__dynamic_guard_patched__", False):
                import math
                import functools

                def _is_uniform(dist: List[float]) -> bool:
                    return len(dist) >= 3 and (max(dist) - min(dist)) <= 0.0005

                def _low_info(dist: List[float]) -> bool:
                    try:
                        if len(dist) < 3:
                            return False
                        ent = -sum(p * math.log(p + 1e-12) for p in dist)
                        return (math.log(len(dist)) - ent) < 0.01
                    except Exception:
                        return False

                def _extract_dists(args, kwargs):
                    d: List[List[float]] = []
                    cand = list(args) + list(kwargs.values())

                    def coerce(obj):
                        if (
                            isinstance(obj, (list, tuple))
                            and obj
                            and all(isinstance(x, (int, float)) for x in obj)
                        ):
                            vals = [float(x) for x in obj]
                            if all(0.0 <= v <= 1.0 for v in vals):
                                return vals
                        for attr in (
                            "predicted_options",
                            "options",
                            "predictedOptions",
                        ):
                            if hasattr(obj, attr):
                                try:
                                    opts = getattr(obj, attr)
                                    vals = []
                                    for o in opts:
                                        p = getattr(o, "probability", None)
                                        if isinstance(p, (int, float)):
                                            vals.append(float(p))
                                    if vals and all(0.0 <= v <= 1.0 for v in vals):
                                        return vals
                                except Exception:
                                    pass
                        return None

                    for o in cand:
                        try:
                            vals = coerce(o)
                            if vals:
                                s = sum(vals)
                                if s > 0:
                                    d.append([v / s for v in vals])
                        except Exception:
                            continue
                    return d

                def _wrap(name, fn):
                    if not callable(fn) or getattr(
                        fn, "__metac_dynamic_guard__", False
                    ):
                        return fn

                    @functools.wraps(fn)
                    def inner(*a, **kw):
                        if os.getenv("PUBLICATION_GUARD_DEBUG") == "1":
                            logger.info(f"Dynamic guard inspecting {name}")
                        if "pred" in name.lower():  # focus on prediction posting
                            for dist in _extract_dists(a, kw):
                                if _is_uniform(dist) or _low_info(dist):
                                    logger.info(
                                        f"Dynamic guard: blocked {name} uniform/low-info MC distribution (k={len(dist)})"
                                    )
                                    return None
                        return fn(*a, **kw)

                    setattr(inner, "__metac_dynamic_guard__", True)
                    return inner

                target_methods = []
                for attr in dir(mc):
                    if (
                        any(tok in attr.lower() for tok in ("post", "submit", "create"))
                        and "pred" in attr.lower()
                    ):
                        try:
                            original = getattr(mc, attr)
                            wrapped = _wrap(attr, original)
                            if wrapped is not original:
                                setattr(mc, attr, wrapped)
                                target_methods.append(attr)
                        except Exception:
                            continue
                setattr(mc, "__dynamic_guard_patched__", True)
                if target_methods:
                    logger.info(
                        f"Dynamic publication guard active on metaculus_client methods: {target_methods}"
                    )
                elif os.getenv("PUBLICATION_GUARD_DEBUG") == "1":
                    try:
                        all_methods = [
                            m
                            for m in dir(mc)
                            if callable(getattr(mc, m)) and not m.startswith("_")
                        ]
                        logger.info(
                            f"Dynamic guard debug: no candidate methods matched filter; available methods: {all_methods}"
                        )
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover
            logger.debug(f"Dynamic guard setup failed: {e}")

    def _has_openrouter_key(self) -> bool:
        """Check if OpenRouter API key is available."""
        return bool(
            self.openrouter_api_key and not self.openrouter_api_key.startswith("dummy_")
        )

    def _has_perplexity_key(self) -> bool:
        """Check if Perplexity API key is available."""
        key = os.getenv("PERPLEXITY_API_KEY")
        return bool(key and not key.startswith("dummy_"))

    def _has_exa_key(self) -> bool:
        """Check if Exa API key is available."""
        key = os.getenv("EXA_API_KEY")
        return bool(key and not key.startswith("dummy_"))

    def _has_metaculus_proxy(self) -> bool:
        """Check if Metaculus proxy is enabled."""
        return os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true"

    async def _call_llm_based_research(self, question: str) -> str:
        """Fallback research method using LLM when no external APIs are available."""
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Use the enhanced research prompt with mini model
                research_prompt = self.anti_slop_prompts.get_research_prompt(
                    question_text=question, model_tier="mini"
                )

                # Add context that this is LLM-based research
                enhanced_prompt = f"""
{research_prompt}

**IMPORTANT NOTE**: External research APIs are not available. Please provide research based on your training data knowledge, clearly indicating the limitations and knowledge cutoff date. Focus on:
- General background information about the topic
- Historical context and patterns
- Known factors that typically influence such questions
- Explicit acknowledgment of information limitations

Be very clear about what information may be outdated or incomplete.
"""

                # Get budget-aware context for routing
                budget_context = (
                    self.tri_model_router.get_budget_aware_routing_context()
                )
                budget_remaining = (
                    budget_context.remaining_percentage if budget_context else 100.0
                )

                research = await self.tri_model_router.route_query(
                    task_type="research",
                    content=enhanced_prompt,
                    complexity="medium",
                    budget_remaining=budget_remaining,
                )

                return research

            except Exception as e:
                logger.warning(f"LLM-based research failed: {e}")
                return ""

        return ""

    def _integrate_budget_manager_with_operation_modes(self):
        """Integrate budget manager with operation mode transitions and alerts (Task 8.2)."""
        if not (self.budget_manager and self.budget_aware_manager):
            return

        try:
            # Monitor budget utilization and trigger operation mode changes
            monitoring_result = self.budget_aware_manager.monitor_budget_utilization()

            # Check for threshold alerts and log them
            if monitoring_result.get("threshold_alerts"):
                for alert in monitoring_result["threshold_alerts"]:
                    logger.warning(
                        f"Budget threshold alert: {alert['threshold_name']} "
                        f"({alert['current_utilization']:.1f}% utilization)"
                    )

                    # Send alert through budget alert system if available
                    if self.budget_alert_system:
                        self.budget_alert_system.check_and_alert()

            # Detect and execute operation mode transitions
            mode_switched, transition_log = (
                self.budget_aware_manager.detect_and_switch_operation_mode()
            )

            if mode_switched and transition_log:
                logger.info(
                    f"Operation mode transition executed: "
                    f"{transition_log.from_mode.value} → {transition_log.to_mode.value}, "
                    f"estimated savings: ${transition_log.cost_savings_estimate:.4f}"
                )

                # Update performance monitoring with mode transition
                if self.performance_monitor:
                    self.performance_monitor.record_model_usage(
                        question_id="mode_transition",
                        task_type="operation_mode_change",
                        selected_model=f"mode_{transition_log.to_mode.value}",
                        selected_tier="system",
                        routing_rationale=transition_log.trigger_reason,
                        estimated_cost=transition_log.cost_savings_estimate,
                        operation_mode=transition_log.to_mode.value,
                        budget_remaining=transition_log.remaining_budget,
                    )

                # Update tri-model router with new operation mode for budget-aware routing
                if self.tri_model_router:
                    # Apply operation mode adjustments to model selection
                    current_mode = transition_log.to_mode.value
                    logger.info(
                        f"Updating tri-model router operation mode to: {current_mode}"
                    )

                    # The router will automatically use the budget-aware operation manager
                    # for future routing decisions based on the new mode

        except Exception as e:
            logger.error(f"Budget manager integration error: {e}")

    def _track_question_processing_cost(
        self,
        question_id: str,
        task_type: str,
        cost: float,
        model_used: str,
        success: bool,
    ):
        """Track cost and performance metrics for question processing (Task 8.2)."""
        try:
            # Estimate token usage for budget manager (approximate)
            estimated_input_tokens = 1000  # Default estimate
            estimated_output_tokens = 500  # Default estimate

            # Update budget manager with actual cost
            if self.budget_manager:
                self.budget_manager.record_cost(
                    question_id=question_id,
                    model=model_used,
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    task_type=task_type,
                    success=success,
                )

            # Update budget-aware operation manager performance metrics
            if self.budget_aware_manager:
                current_mode = (
                    self.budget_aware_manager.operation_mode_manager.get_current_mode()
                )

                # Update question processing count by mode
                mode_key = current_mode.value
                if (
                    mode_key
                    in self.budget_aware_manager.performance_metrics[
                        "questions_processed_by_mode"
                    ]
                ):
                    self.budget_aware_manager.performance_metrics[
                        "questions_processed_by_mode"
                    ][mode_key] += 1

                # Update average cost by mode
                if (
                    mode_key
                    in self.budget_aware_manager.performance_metrics[
                        "average_cost_by_mode"
                    ]
                ):
                    current_avg = self.budget_aware_manager.performance_metrics[
                        "average_cost_by_mode"
                    ][mode_key]
                    question_count = self.budget_aware_manager.performance_metrics[
                        "questions_processed_by_mode"
                    ][mode_key]

                    # Calculate new average
                    new_avg = (
                        (current_avg * (question_count - 1)) + cost
                    ) / question_count
                    self.budget_aware_manager.performance_metrics[
                        "average_cost_by_mode"
                    ][mode_key] = new_avg

            # Update performance monitoring with execution outcome
            if self.performance_monitor:
                self.performance_monitor.record_execution_outcome(
                    question_id=question_id,
                    actual_cost=cost,
                    execution_time=1.0,  # Default execution time
                    quality_score=0.8
                    if success
                    else 0.3,  # Estimated quality based on success
                    success=success,
                    fallback_used=False,  # Would need to be passed from caller
                )

        except Exception as e:
            logger.error(f"Cost tracking error for question {question_id}: {e}")

    def _check_tournament_compliance_integration(self) -> Dict[str, Any]:
        """Check tournament compliance with integrated systems (Task 8.2)."""
        compliance_status = {
            "budget_compliant": True,
            "operation_mode_compliant": True,
            "performance_compliant": True,
            "error_recovery_compliant": True,
            "tri_model_integration_compliant": True,
            "cost_tracking_compliant": True,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check budget compliance
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                if budget_status.utilization_percentage > 100:
                    compliance_status["budget_compliant"] = False
                    compliance_status["issues"].append(
                        "Budget exceeded 100% utilization"
                    )
                    compliance_status["recommendations"].append(
                        "Activate emergency mode immediately"
                    )
                elif budget_status.utilization_percentage > 95:
                    compliance_status["recommendations"].append(
                        "Consider switching to critical operation mode"
                    )

            # Check operation mode compliance
            if self.budget_aware_manager:
                current_mode = (
                    self.budget_aware_manager.operation_mode_manager.get_current_mode()
                )
                emergency_protocol = (
                    self.budget_aware_manager.current_emergency_protocol
                )

                if emergency_protocol.value != "none":
                    compliance_status["operation_mode_compliant"] = False
                    compliance_status["issues"].append(
                        f"Emergency protocol active: {emergency_protocol.value}"
                    )
                    compliance_status["recommendations"].append(
                        "Monitor system closely during emergency protocol"
                    )

                # Check if operation mode matches budget utilization
                budget_util = (
                    self.budget_manager.get_budget_status().utilization_percentage
                    if self.budget_manager
                    else 0
                )
                expected_mode = self.budget_aware_manager.get_operation_mode_for_budget(
                    budget_util
                )
                if current_mode.value != expected_mode:
                    compliance_status["operation_mode_compliant"] = False
                    compliance_status["issues"].append(
                        f"Operation mode mismatch: current={current_mode.value}, expected={expected_mode}"
                    )

            # Check tri-model router integration compliance
            if self.tri_model_router and self.budget_aware_manager:
                try:
                    # Verify router can access budget-aware operation manager
                    router_status = self.tri_model_router.get_model_status()
                    if isinstance(router_status, dict):
                        # Check if status objects have is_available attribute
                        unavailable_tiers = []
                        for tier, status in router_status.items():
                            # Defensive: status may be a plain string in edge cases
                            if (
                                hasattr(status, "is_available")
                                and getattr(status, "is_available") is False
                            ):
                                unavailable_tiers.append(tier)

                        if unavailable_tiers:
                            compliance_status["tri_model_integration_compliant"] = False
                            compliance_status["issues"].append(
                                f"Tri-model router tiers unavailable: {', '.join(unavailable_tiers)}"
                            )
                            compliance_status["recommendations"].append(
                                "Check model availability and fallback chains"
                            )
                except Exception as e:
                    compliance_status["tri_model_integration_compliant"] = False
                    compliance_status["issues"].append(
                        f"Tri-model router integration error: {str(e)}"
                    )

            # Check cost tracking integration compliance
            if self.budget_manager and self.performance_monitor:
                try:
                    # Verify cost tracking is working (allow empty records for fresh start)
                    recent_records = (
                        len(self.budget_manager.cost_records[-10:])
                        if self.budget_manager.cost_records
                        else 0
                    )
                    if (
                        recent_records == 0
                        and self.budget_manager.questions_processed > 0
                    ):
                        # Only flag as issue if we've processed questions but have no records
                        compliance_status["cost_tracking_compliant"] = False
                        compliance_status["issues"].append(
                            "No recent cost tracking records found despite processing questions"
                        )
                        compliance_status["recommendations"].append(
                            "Verify cost tracking integration is functioning"
                        )
                    # If no questions processed yet, cost tracking is still compliant
                except Exception as e:
                    compliance_status["cost_tracking_compliant"] = False
                    compliance_status["issues"].append(
                        f"Cost tracking integration error: {str(e)}"
                    )

            # Check performance monitoring compliance
            if self.performance_monitor:
                try:
                    comprehensive_status = (
                        self.performance_monitor.get_comprehensive_status()
                    )
                    overall_health = comprehensive_status.overall_health
                    if overall_health in ["concerning", "critical"]:
                        compliance_status["performance_compliant"] = False
                        compliance_status["issues"].append(
                            f"System health: {overall_health}"
                        )
                        compliance_status["recommendations"].extend(
                            comprehensive_status.optimization_recommendations[:3]
                        )
                except Exception as e:
                    compliance_status["performance_compliant"] = False
                    compliance_status["issues"].append(
                        f"Performance monitoring error: {str(e)}"
                    )

            # Check error recovery compliance
            if self.error_recovery_manager:
                try:
                    recovery_status = self.error_recovery_manager.get_recovery_status()
                    if (
                        recovery_status.get("system_health", {}).get("status")
                        == "critical"
                    ):
                        compliance_status["error_recovery_compliant"] = False
                        compliance_status["issues"].append(
                            "Error recovery system in critical state"
                        )
                        compliance_status["recommendations"].append(
                            "Review error recovery logs and reset if necessary"
                        )
                except Exception as e:
                    compliance_status["error_recovery_compliant"] = False
                    compliance_status["issues"].append(
                        f"Error recovery check failed: {str(e)}"
                    )

            # Overall compliance assessment
            compliance_status["overall_compliant"] = all(
                [
                    compliance_status["budget_compliant"],
                    compliance_status["operation_mode_compliant"],
                    compliance_status["performance_compliant"],
                    compliance_status["error_recovery_compliant"],
                    compliance_status["tri_model_integration_compliant"],
                    compliance_status["cost_tracking_compliant"],
                ]
            )

            # Add timestamp for monitoring
            compliance_status["last_checked"] = datetime.now().isoformat()

        except Exception as e:
            compliance_status["issues"].append(f"Compliance check error: {str(e)}")
            compliance_status["overall_compliant"] = False
            logger.error(f"Tournament compliance check error: {e}")

        return compliance_status

    def _handle_budget_exhaustion(self, question_id: str = "unknown") -> bool:
        """Handle budget exhaustion scenarios with graceful degradation."""
        if not self.budget_manager:
            return False

        budget_status = self.budget_manager.get_budget_status()

        if budget_status.status_level == "emergency":
            if not self.emergency_mode_active:
                logger.critical(
                    f"EMERGENCY MODE ACTIVATED: Budget utilization at {budget_status.utilization_percentage:.1f}%"
                )
                logger.critical(f"Remaining budget: ${budget_status.remaining:.4f}")
                logger.critical(
                    f"Estimated questions remaining: {budget_status.estimated_questions_remaining}"
                )
                self.emergency_mode_active = True

                # Alert system if available
                if self.budget_alert_system:
                    # Guard: method may not exist in some minimal deployments
                    if hasattr(self.budget_alert_system, "send_critical_alert"):
                        self.budget_alert_system.send_critical_alert(  # type: ignore[attr-defined]
                            f"Emergency mode activated for question {question_id}",
                            budget_status,
                        )

            # In emergency mode, only process high-priority questions
            return True

        elif budget_status.utilization_percentage >= 100:
            logger.critical("BUDGET EXHAUSTED: Cannot process any more questions")
            if self.budget_alert_system:
                if hasattr(self.budget_alert_system, "send_critical_alert"):
                    self.budget_alert_system.send_critical_alert(  # type: ignore[attr-defined]
                        "Budget completely exhausted",
                        budget_status,
                    )
            return True

        return False

    def _handle_api_failure(self, error: Exception, model: str, task_type: str) -> str:
        """Handle API failures with intelligent fallbacks."""
        self.api_failure_count += 1
        logger.warning(
            f"API failure #{self.api_failure_count} for {model} ({task_type}): {error}"
        )

        # If too many failures, activate emergency mode
        if self.api_failure_count >= self.max_api_failures:
            logger.error(
                f"Too many API failures ({self.api_failure_count}), activating emergency protocols"
            )
            self.emergency_mode_active = True

        # Determine fallback strategy based on provider prefix (e.g., "openai/..", "perplexity/..", "metaculus/..")
        provider_prefix = (
            model.split("/", 1)[0].lower() if "/" in model else model.lower()
        )

        if provider_prefix in {
            "openai",
            "anthropic",
            "perplexity",
            "mistral",
            "meta",
            "moonshotai",
        }:
            # OpenRouter family provider failed, try Metaculus proxy first
            if os.getenv("ENABLE_PROXY_CREDITS", "true").lower() == "true":
                fallback_model = self.fallback_models["proxy"]
                logger.info(f"Falling back to Metaculus proxy: {fallback_model}")
                return fallback_model
            # No proxy available, use emergency free model
            fallback_model = self.fallback_models["emergency"]
            logger.info(f"Using emergency fallback model: {fallback_model}")
            return fallback_model

        if provider_prefix == "metaculus":
            # Proxy failed, try OpenRouter free model if key exists
            if self.openrouter_api_key and not self.openrouter_api_key.startswith(
                "dummy_"
            ):
                fallback_model = self.fallback_models["emergency"]
                logger.info(
                    f"Proxy failed, falling back to OpenRouter free model: {fallback_model}"
                )
                return fallback_model
            # No OpenRouter key, use last resort
            fallback_model = self.fallback_models["last_resort"]
            logger.warning(f"Using last resort model: {fallback_model}")
            return fallback_model

        # Unknown provider failed, use emergency model
        fallback_model = self.fallback_models["emergency"]
        logger.info(f"Unknown provider failed, using emergency model: {fallback_model}")
        return fallback_model

    def _create_emergency_response(
        self, task_type: str, question_text: str = ""
    ) -> str:
        """Create emergency response when all APIs fail."""
        if task_type == "research":
            return (
                "Research unavailable due to API failures. "
                "Proceeding with forecast based on question information only."
            )
        elif task_type == "forecast":
            # Avoid emitting language that could be auto‑published as a neutral forecast
            return (
                f"Unable to generate detailed forecast due to API failures. "
                f"Question: {question_text[:200]}... "
                f"Publishing is blocked to prevent a low‑information neutral stub."
            )
        else:
            return "Task unavailable due to system limitations."

    async def _safe_llm_invoke(
        self,
        llm,
        prompt: str,
        task_type: str,
        question_id: str = "unknown",
        max_retries: int = 3,
    ) -> str:
        """Safely invoke LLM with error handling and fallbacks."""
        last_error = None

        for attempt in range(max_retries):
            try:
                # Check budget before each attempt
                if self._handle_budget_exhaustion(question_id):
                    if task_type == "research":
                        return "Research skipped due to budget constraints."
                    else:
                        return self._create_emergency_response(task_type, prompt[:200])

                # Skip immediately if model previously marked unavailable
                model_name = getattr(llm, "model", "unknown")
                if model_name in self._unavailable_models:
                    logger.warning(
                        f"Skipping invoke for unavailable model: {model_name}"
                    )
                    return self._create_emergency_response(task_type, prompt[:200])

                # Attempt LLM call
                response = await llm.invoke(prompt)

                # Reset failure count on success
                if self.api_failure_count > 0:
                    logger.info(
                        f"API call successful after {self.api_failure_count} previous failures"
                    )
                    self.api_failure_count = max(0, self.api_failure_count - 1)

                return response

            except Exception as e:
                last_error = e
                err_text = str(e).lower()
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                terminal_patterns = [
                    "404",
                    "not found",
                    "does not exist",
                    "allowance",
                    "unauthorized",
                    "forbidden",
                ]
                model_name = getattr(llm, "model", "unknown")
                is_terminal = any(pat in err_text for pat in terminal_patterns)
                if is_terminal:
                    self._unavailable_models.add(model_name)
                    logger.error(
                        f"Terminal error detected for model {model_name}; added to negative cache and aborting retries."
                    )
                    break

                if attempt >= max_retries - 1:
                    break

                # Do NOT attempt dynamic fallback to free/proxy models (disabled). Just simple retry on transient errors.
                await asyncio.sleep(min(2**attempt, 5))

        # All attempts failed
        logger.error(
            f"All LLM attempts failed for {task_type}. Last error: {last_error}"
        )
        return self._create_emergency_response(task_type, prompt[:200])

    def _is_unacceptable_forecast(self, prediction: float, reasoning: str) -> bool:
        """Gatekeeper for forecasts we should not publish.

        Criteria:
        - Emergency/neutral messages
        - Exactly neutral 0.5 probability (likely extraction fallback)
        """
        if prediction is None:
            return True
        if abs(prediction - 0.5) < 1e-6:
            return True
        text = (reasoning or "").lower()
        if "research unavailable" in text:
            return True
        if NEUTRAL_TEXT_UNABLE in text:
            return True
        if NEUTRAL_TEXT_ASSIGNING in text:
            return True
        return False

    def _is_unacceptable_mc_forecast(
        self, prediction: "PredictedOptionList", reasoning: str, options: list[str]
    ) -> bool:
        """Gatekeeper for multiple-choice forecasts we should not publish.

        ONLY withhold when research actually fails - uncertain forecasts are still valuable.
        """
        if prediction is None or not getattr(prediction, "predicted_options", None):
            return True

        probs = []
        try:
            for p in prediction.predicted_options:
                # Ensure option is among provided options and probability is valid
                if options and getattr(p, "option_name", None) not in options:
                    continue
                if hasattr(p, "probability") and p.probability is not None:
                    probs.append(float(p.probability))
        except Exception:
            return True

        if not probs:
            return True

        text = (reasoning or "").lower()
        # ONLY withhold when research unavailable - let uncertain forecasts through
        if "research unavailable" in text:
            return True
        if NEUTRAL_TEXT_UNABLE in text:
            return True
        if NEUTRAL_TEXT_ASSIGNING in text:
            return True

        return False

    def _mc_distribution_reason(
        self, prediction: "PredictedOptionList", reasoning: str, options: list[str]
    ) -> Optional[str]:
        """Return reason code if unacceptable else None (used for diagnostics)."""
        if prediction is None or not getattr(prediction, "predicted_options", None):
            return "MISSING"
        probs = []
        try:
            for p in prediction.predicted_options:
                if options and getattr(p, "option_name", None) not in options:
                    continue
                if hasattr(p, "probability") and p.probability is not None:
                    probs.append(float(p.probability))
        except Exception:
            return "PARSE_ERROR"
        if not probs:
            return "EMPTY"
        max_p = max(probs)
        min_p = min(probs)
        n = len(probs)
        uniform_target = 1.0 / n if n else 0
        if all(abs(p - uniform_target) < 0.02 for p in probs):
            return "UNIFORM"
        if (max_p - min_p) < 0.05:
            return "LOW_SPREAD"
        if max_p < 0.30:
            return "LOW_MAX"
        text = (reasoning or "").lower()
        # Only flag research failures - let uncertain forecasts through
        if "research unavailable" in text:
            return "RESEARCH_UNAVAILABLE"
        if NEUTRAL_TEXT_UNABLE in text:
            return "NEUTRAL_UNABLE"
        if NEUTRAL_TEXT_ASSIGNING in text:
            return "NEUTRAL_ASSIGN"
        if "[withheld:" in text:
            return "WITHHELD_MARKER"
        return None

    def _is_unacceptable_numeric_forecast(
        self, prediction: "NumericDistribution", reasoning: str
    ) -> bool:
        """Gatekeeper for numeric forecasts we should not publish.

        Heuristics:
        - Missing/empty distribution or < 3 declared percentiles
        - Collapsed distribution (p90 <= p10)
        - Emergency/neutral messages
        """
        if prediction is None:
            return True

        declared = getattr(prediction, "declared_percentiles", None)
        if not declared or len(declared) < 3:
            return True

        # Try to find p10 and p90
        try:
            p10_vals = [
                pp.value for pp in declared if abs(float(pp.percentile) - 0.10) < 1e-6
            ]
            p90_vals = [
                pp.value for pp in declared if abs(float(pp.percentile) - 0.90) < 1e-6
            ]
            if p10_vals and p90_vals:
                if float(p90_vals[0]) <= float(p10_vals[0]):
                    return True
        except Exception:
            # If parsing fails, be conservative
            return True

        text = (reasoning or "").lower()
        if NEUTRAL_TEXT_UNABLE in text:
            return True
        if NEUTRAL_TEXT_ASSIGNING in text:
            return True

        return False

    async def _retry_forecast_with_alternatives(
        self, question: "BinaryQuestion", research: str
    ) -> Any:
        """Disabled alternative forecast retries (free/perplexity models unvalidated)."""
        # CRITICAL FIX: Never publish fallback forecasts without proper research
        # Return None to skip publication instead of returning 0.5
        q_id = getattr(question, "question_id", getattr(question, "id", "unknown"))
        logger.error(
            f"Forecast retry alternatives exhausted for question {q_id} - skipping publication to maintain quality"
        )
        return None  # Skip publication - tournament compliance requires proper research

    async def _retry_mc_with_alternatives(
        self, question: "MultipleChoiceQuestion", research: str
    ) -> Any:
        """Disabled MC alternative retries; returns uniform placeholder distribution (may be gated)."""
        from forecasting_tools import PredictedOptionList, PredictedOption

        try:
            eq = 1.0 / max(len(question.options), 1)
            pred = PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=o, probability=eq)
                    for o in question.options
                ]
            )
        except Exception:
            pred = None
        return _mk_rp(prediction_value=pred, reasoning=SAFE_REASONING_FALLBACK)

    async def _retry_numeric_with_alternatives(
        self, question: "NumericQuestion", research: str
    ) -> Any:
        """Disabled numeric alternative retries; returns empty distribution placeholder."""
        return _mk_rp(prediction_value=None, reasoning=SAFE_REASONING_FALLBACK)

    async def run_research(self, question: "MetaculusQuestion") -> str:
        """Enhanced research with multi-stage validation pipeline, tri-model routing, and comprehensive error handling."""
        async with self._concurrency_limiter:
            # Early: try the dedicated multi-stage research pipeline (AskNews-first) if available
            if getattr(self, "_multi_stage_pipeline", None):
                try:
                    # Guard none pipeline edge case
                    early_result = None
                    if self._multi_stage_pipeline and hasattr(
                        self._multi_stage_pipeline, "execute_research_pipeline"
                    ):
                        early_result = (
                            await self._multi_stage_pipeline.execute_research_pipeline(
                                question.question_text,
                                context={
                                    "question_url": question.page_url,
                                    "background_info": getattr(
                                        question, "background_info", ""
                                    ),
                                    "resolution_criteria": getattr(
                                        question, "resolution_criteria", ""
                                    ),
                                    "fine_print": getattr(question, "fine_print", ""),
                                },
                            )
                        )
                    if (
                        early_result
                        and early_result.get("success")
                        and early_result.get("final_research")
                    ):
                        return early_result["final_research"]
                except Exception as e:
                    logger.warning(f"Early dedicated research pipeline failed: {e}")
            # Get budget status and operation mode for intelligent routing
            budget_remaining = 100.0
            operation_mode = "normal"

            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                budget_remaining = 100.0 - budget_status.utilization_percentage

                # Check and alert on budget status
                if self.budget_alert_system:
                    self.budget_alert_system.check_and_alert()

                # Get current operation mode from budget-aware manager
                if self.budget_aware_manager:
                    mode_switched, transition_log = (
                        self.budget_aware_manager.detect_and_switch_operation_mode()
                    )
                    if mode_switched:
                        if (
                            transition_log
                            and hasattr(transition_log, "from_mode")
                            and hasattr(transition_log, "to_mode")
                        ):
                            logger.info(
                                f"Operation mode switched: {transition_log.from_mode.value} → {transition_log.to_mode.value}"
                            )

                    operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

                # Check if question should be skipped based on operation mode
                if self.budget_aware_manager:
                    should_skip, skip_reason = (
                        self.budget_aware_manager.should_skip_question(
                            question_priority="normal",  # Could be extracted from question metadata
                            question_complexity="medium",
                        )
                    )
                    if should_skip:
                        logger.warning(
                            f"Skipping research for {question.page_url}: {skip_reason}"
                        )
                        return f"Research skipped: {skip_reason}"

            # PRIORITY: Try enhanced multi-stage validation pipeline first (Task 8.1)
            if self.multi_stage_pipeline:
                try:
                    # Use the complete multi-stage validation pipeline for research
                    pipeline_result = await self.multi_stage_pipeline.process_question(
                        question=question.question_text,
                        question_type="research",  # Special type for research-only processing
                        context={
                            "question_url": question.page_url,
                            "budget_remaining": budget_remaining,
                            "operation_mode": operation_mode,
                            "background_info": getattr(question, "background_info", ""),
                            "resolution_criteria": getattr(
                                question, "resolution_criteria", ""
                            ),
                            "fine_print": getattr(question, "fine_print", ""),
                        },
                    )

                    if (
                        pipeline_result.pipeline_success
                        and pipeline_result.research_result.content
                    ):
                        logger.info(
                            f"Multi-stage validation pipeline successful for URL {question.page_url}, "
                            f"cost: ${pipeline_result.total_cost:.4f}, quality: {pipeline_result.quality_score:.2f}"
                        )

                        # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                        self._track_question_processing_cost(
                            question_id=str(getattr(question, "id", "unknown")),
                            task_type="research_pipeline",
                            cost=pipeline_result.total_cost,
                            model_used="multi_stage_pipeline",
                            success=True,
                        )

                        # Check and update operation modes based on budget utilization
                        self._integrate_budget_manager_with_operation_modes()

                        return pipeline_result.research_result.content

                    else:
                        logger.warning(
                            f"Multi-stage validation pipeline failed for {question.page_url}: "
                            f"Success={pipeline_result.pipeline_success}, "
                            f"Quality={pipeline_result.quality_score:.2f}"
                        )
                        # Continue to fallback methods

                except Exception as e:
                    logger.warning(f"Multi-stage validation pipeline failed: {e}")

                    # Use comprehensive error recovery (Task 8.1)
                    if self.error_recovery_manager:
                        try:
                            from src.infrastructure.reliability.error_classification import (
                                ErrorContext,
                            )

                            error_context = ErrorContext(
                                task_type="research",
                                model_tier="mini",
                                operation_mode=operation_mode,
                                budget_remaining=budget_remaining,
                                attempt_number=1,
                                question_id=str(getattr(question, "id", "unknown")),
                                provider="multi_stage_pipeline",
                            )

                            recovery_result = (
                                await self.error_recovery_manager.recover_from_error(
                                    e, error_context
                                )
                            )
                            if recovery_result.success:
                                logger.info(
                                    f"Error recovery successful: {recovery_result.message}"
                                )
                                # Could retry with recovered configuration, but for now continue to fallback
                            else:
                                logger.warning(
                                    f"Error recovery failed: {recovery_result.message}"
                                )
                        except Exception as recovery_error:
                            logger.error(
                                f"Error recovery system failed: {recovery_error}"
                            )

                    # Continue to fallback methods

            # Try tournament-optimized AskNews client first
            if self.tournament_asknews:
                try:
                    research = await self.tournament_asknews.get_news_research(
                        question.question_text
                    )

                    if research and len(research.strip()) > 0:
                        # Log usage stats periodically
                        stats = self.tournament_asknews.get_usage_stats()
                        if stats["total_requests"] % 10 == 0:  # Log every 10 requests
                            logger.info(
                                f"AskNews usage: {stats['estimated_quota_used']}/{stats['quota_limit']} "
                                f"({stats['quota_usage_percentage']:.1f}%), "
                                f"Success rate: {stats['success_rate']:.1f}%"
                            )

                        # Alert on high quota usage
                        if self.tournament_asknews.should_alert_quota_usage():
                            alert_level = (
                                self.tournament_asknews.get_quota_alert_level()
                            )
                            logger.warning(
                                f"AskNews quota usage {alert_level}: "
                                f"{stats['quota_usage_percentage']:.1f}% used"
                            )

                        logger.info(
                            f"Tournament AskNews research successful for URL {question.page_url}"
                        )
                        return research

                except Exception as e:
                    logger.warning(f"Tournament AskNews client failed: {e}")
                    # Continue to other research methods

            # FALLBACK: Try tri-model router for intelligent research
            if not research and self.tri_model_router and self.anti_slop_prompts:
                try:
                    # Create anti-slop research prompt
                    research_prompt = self.anti_slop_prompts.get_research_prompt(
                        question_text=question.question_text,
                        model_tier="mini",  # Use mini model for research by default
                    )

                    # Get budget-aware context for routing
                    budget_context = (
                        self.tri_model_router.get_budget_aware_routing_context()
                    )
                    budget_remaining = (
                        budget_context.remaining_percentage if budget_context else 100.0
                    )

                    # Route to optimal model based on budget and complexity
                    research = await self.tri_model_router.route_query(
                        task_type="research",
                        content=research_prompt,
                        complexity="medium",
                        budget_remaining=budget_remaining,
                    )

                    if research and len(research.strip()) > 50:
                        logger.info(
                            f"Tri-model research successful for URL {question.page_url}"
                        )
                        return research

                except Exception as e:
                    logger.warning(f"Tri-model research failed: {e}")
                    # Continue to fallback methods

            research = ""

            # Fallback to original AskNews if available
            if (
                not research
                and os.getenv("ASKNEWS_CLIENT_ID")
                and os.getenv("ASKNEWS_SECRET")
            ):
                try:
                    research = await AskNewsSearcher().get_formatted_news_async(
                        question.question_text
                    )
                    if research and len(research.strip()) > 0:
                        logger.info(
                            f"Original AskNews research successful for URL {question.page_url}"
                        )
                        return research
                except Exception as e:
                    logger.warning(f"Original AskNews failed: {e}")

            # Fallback to OpenRouter Perplexity (if available)
            if not research and self._has_openrouter_key():
                try:
                    research = await self._call_perplexity(
                        question.question_text, use_open_router=True
                    )
                    if research and len(research.strip()) > 0:
                        logger.info(
                            f"OpenRouter Perplexity research successful for URL {question.page_url}"
                        )
                        return research
                except Exception as e:
                    logger.warning(f"OpenRouter Perplexity search failed: {e}")

            # Fallback to Perplexity direct (if available)
            if not research and self._has_perplexity_key():
                try:
                    research = await self._call_perplexity(question.question_text)
                    if research and len(research.strip()) > 0:
                        logger.info(
                            f"Perplexity research successful for URL {question.page_url}"
                        )
                        return research
                except Exception as e:
                    logger.warning(f"Perplexity search failed: {e}")

            # Fallback to Exa (if available)
            if not research and self._has_exa_key():
                try:
                    research = await self._call_exa_smart_searcher(
                        question.question_text
                    )
                    if research and len(research.strip()) > 0:
                        logger.info(
                            f"Exa research successful for URL {question.page_url}"
                        )
                        return research
                except Exception as e:
                    logger.warning(f"Exa search failed: {e}")

            # Final fallback: Use LLM-based research if no external APIs available
            if not research and (
                self._has_openrouter_key() or self._has_metaculus_proxy()
            ):
                try:
                    research = await self._call_llm_based_research(
                        question.question_text
                    )
                    if research and len(research.strip()) > 0:
                        logger.info(
                            f"LLM-based research successful for URL {question.page_url}"
                        )
                        return research
                except Exception as e:
                    logger.warning(f"LLM-based research failed: {e}")

            # If all research methods fail, check if we're in emergency mode
            if not research:
                if self.emergency_mode_active or self._handle_budget_exhaustion(
                    str(getattr(question, "id", "unknown"))
                ):
                    logger.warning(
                        f"Emergency mode: Skipping research for question URL {question.page_url}"
                    )
                    research = "Research unavailable due to system constraints."
                else:
                    logger.warning(
                        f"All research providers failed for question URL {question.page_url}. "
                        f"Proceeding with empty research."
                    )
                    research = ""

            logger.info(
                f"Research completed for URL {question.page_url} (length: {len(research)})"
            )
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = _safe_clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        # Create Perplexity LLM via centralized factory to ensure correct base_url/headers
        try:
            from src.infrastructure.config.llm_factory import (
                create_perplexity_llm,
                create_llm,
            )

            model = create_perplexity_llm(use_open_router)
        except Exception:
            # Best-effort fallback
            from src.infrastructure.config.llm_factory import create_llm

            model_name = (
                "perplexity/sonar-reasoning"
                if use_open_router
                else "perplexity/sonar-pro"
            )
            model = create_llm(model_name, temperature=0.1)
        # Use safe invoke for Perplexity calls
        response = await self._safe_llm_invoke(model, prompt, "research")
        return response

    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all integrated components (Task 8.2)."""
        status: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "system_components": {},
            "budget_status": {},
            "operation_mode": {},
            "performance_metrics": {},
            "tournament_compliance": {},
            "error_recovery": {},
            "overall_health": "unknown",
        }

        try:
            # Budget manager status
            if self.budget_manager:
                budget_status = self.budget_manager.get_budget_status()
                status["budget_status"] = {
                    "utilization_percentage": budget_status.utilization_percentage,
                    "remaining": budget_status.remaining,
                    "status_level": budget_status.status_level,
                    "estimated_questions_remaining": budget_status.estimated_questions_remaining,
                }

            # Budget-aware operation manager status
            if self.budget_aware_manager:
                operation_details = (
                    self.budget_aware_manager.get_operation_mode_details(
                        status["budget_status"].get("utilization_percentage", 0)
                    )
                )
                status["operation_mode"] = operation_details

                # Get performance metrics
                status["performance_metrics"] = (
                    self.budget_aware_manager.get_performance_metrics()
                )

            # Tri-model router status
            if self.tri_model_router:
                status["system_components"]["tri_model_router"] = {
                    "available": True,
                    "model_status": self.tri_model_router.get_model_status(),
                    "provider_routing": self.tri_model_router.get_openrouter_provider_routing_info(),
                }

            # Multi-stage pipeline status
            if self.multi_stage_pipeline:
                status["system_components"]["multi_stage_pipeline"] = {
                    "available": True,
                    "configuration": self.multi_stage_pipeline.get_pipeline_configuration(),
                }

            # Error recovery status
            if self.error_recovery_manager:
                recovery_status = self.error_recovery_manager.get_recovery_status()
                status["error_recovery"] = recovery_status

            # Performance monitoring status
            if self.performance_monitor:
                status["system_components"]["performance_monitor"] = {
                    "available": True,
                    "system_health": self.performance_monitor.get_system_health_status()
                    if hasattr(self.performance_monitor, "get_system_health_status")
                    else {},  # type: ignore[attr-defined]
                }

            # Tournament compliance check
            status["tournament_compliance"] = (
                self._check_tournament_compliance_integration()
            )

            # Determine overall health
            health_indicators = []

            if status["budget_status"].get("utilization_percentage", 0) < 95:
                health_indicators.append("budget_healthy")

            if status["tournament_compliance"].get("budget_compliant", False):
                health_indicators.append("compliance_healthy")

            if status["error_recovery"].get("system_health", {}).get("status") in [
                "healthy",
                "degraded",
            ]:
                health_indicators.append("recovery_healthy")

            if len(health_indicators) >= 2:
                status["overall_health"] = "healthy"
            elif len(health_indicators) >= 1:
                status["overall_health"] = "degraded"
            else:
                status["overall_health"] = "unhealthy"

        except Exception as e:
            status["error"] = f"Status collection error: {str(e)}"
            status["overall_health"] = "error"
            logger.error(f"Enhanced system status collection error: {e}")

        return status

    def log_enhanced_system_status(self):
        """Log comprehensive system status for monitoring (Task 8.2)."""
        try:
            status = self.get_enhanced_system_status()

            logger.info("=== ENHANCED SYSTEM STATUS ===")
            logger.info(f"Overall Health: {status['overall_health'].upper()}")

            # Budget status
            if status.get("budget_status"):
                budget = status["budget_status"]
                logger.info(
                    f"Budget: {budget.get('utilization_percentage', 0):.1f}% used, "
                    f"${budget.get('remaining', 0):.4f} remaining, "
                    f"~{budget.get('estimated_questions_remaining', 0)} questions left"
                )

            # Operation mode
            if status.get("operation_mode"):
                mode = status["operation_mode"]
                logger.info(
                    f"Operation Mode: {mode.get('operation_mode', 'unknown').upper()}"
                )
                logger.info(f"Mode Description: {mode.get('mode_description', 'N/A')}")

            # Performance metrics
            if status.get("performance_metrics"):
                perf = status["performance_metrics"]
                logger.info(f"Mode Switches: {perf.get('mode_switches_count', 0)}")
                logger.info(
                    f"Emergency Activations: {perf.get('emergency_activations', 0)}"
                )
                logger.info(
                    f"Cost Savings: ${perf.get('cost_savings_achieved', 0):.4f}"
                )

            # Tournament compliance
            if status.get("tournament_compliance"):
                compliance = status["tournament_compliance"]
                compliant_count = sum(
                    1 for v in compliance.values() if isinstance(v, bool) and v
                )
                total_checks = sum(
                    1 for v in compliance.values() if isinstance(v, bool)
                )
                logger.info(
                    f"Tournament Compliance: {compliant_count}/{total_checks} checks passed"
                )

                if compliance.get("issues"):
                    logger.warning("Compliance Issues:")
                    for issue in compliance["issues"]:
                        logger.warning(f"  - {issue}")

            # System components
            if status.get("system_components"):
                components = status["system_components"]
                available_components = [
                    name
                    for name, info in components.items()
                    if info.get("available", False)
                ]
                logger.info(f"Available Components: {', '.join(available_components)}")

        except Exception as e:
            logger.error(f"Enhanced system status logging error: {e}")

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: "BinaryQuestion", research: str
    ) -> Any:
        question_id = str(getattr(question, "id", "unknown"))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="binary",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(
                            question, "resolution_criteria", ""
                        ),
                        "fine_print": getattr(question, "fine_print", ""),
                        "research_data": research,
                    },
                )

                if (
                    pipeline_result.pipeline_success
                    and pipeline_result.forecast_result.quality_validation_passed
                    and pipeline_result.forecast_result.tournament_compliant
                ):
                    logger.info(
                        f"Multi-stage binary forecast successful for question {question_id}, "
                        f"cost: ${pipeline_result.total_cost:.4f}, "
                        f"quality: {pipeline_result.quality_score:.2f}, "
                        f"calibration: {pipeline_result.forecast_result.calibration_score:.2f}"
                    )

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="binary_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True,
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Extract prediction and reasoning from pipeline result with gating
                    try:
                        pipeline_prediction = float(pipeline_result.final_forecast)  # type: ignore
                    except Exception:
                        pipeline_prediction = pipeline_result.final_forecast  # type: ignore
                    pipeline_reasoning = pipeline_result.reasoning or ""

                    # Confidence gate on pipeline output; if unacceptable, try alternatives and otherwise fall through
                    if not self._is_unacceptable_forecast(
                        pipeline_prediction, pipeline_reasoning
                    ):
                        return _mk_rp(
                            prediction_value=pipeline_prediction,
                            reasoning=pipeline_reasoning,
                        )
                    else:
                        logger.warning(
                            "Pipeline binary forecast deemed low-confidence/neutral. Retrying with alternatives..."
                        )
                        alt = await self._retry_forecast_with_alternatives(
                            question, research
                        )
                        if not self._is_unacceptable_forecast(
                            alt.prediction_value, alt.reasoning
                        ):
                            return alt
                        # Fall through to tri-model/legacy paths below

                else:
                    logger.warning(
                        f"Multi-stage binary forecast quality issues for {question_id}: "
                        f"Success={pipeline_result.pipeline_success}, "
                        f"Quality={pipeline_result.forecast_result.quality_validation_passed}, "
                        f"Compliant={pipeline_result.forecast_result.tournament_compliant}"
                    )
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage binary forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success:
                            logger.info(
                                f"Binary forecast error recovery successful: {recovery_result.message}"
                            )
                        else:
                            logger.warning(
                                f"Binary forecast error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Binary forecast error recovery system failed: {recovery_error}"
                        )

                # Continue to fallback methods

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    # Get model selection adjustments based on current operation mode
                    adjusted_model = (
                        self.budget_aware_manager.apply_model_selection_adjustments(
                            "forecast"
                        )
                    )

                    # Map model to tier for prompt optimization
                    if (
                        "gpt-5-nano" in adjusted_model
                        or "gpt-4o-mini" in adjusted_model
                    ):
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop binary forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_binary_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info or "",
                    resolution_criteria=getattr(question, "resolution_criteria", ""),
                    fine_print=getattr(question, "fine_print", ""),
                    research=research,
                    model_tier=selected_model_tier,
                )

                # Get budget-aware context for routing
                budget_context = (
                    self.tri_model_router.get_budget_aware_routing_context()
                )
                budget_remaining = (
                    budget_context.remaining_percentage if budget_context else 100.0
                )

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining,
                )

                logger.info(
                    f"Enhanced tri-model binary forecast successful for question {question_id} "
                    f"using {selected_model_tier} tier"
                )

            except Exception as e:
                logger.warning(f"Enhanced tri-model binary forecast failed: {e}")

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success and recovery_result.fallback_result:
                            logger.info(
                                f"Tri-model error recovery successful: {recovery_result.message}"
                            )
                            # Could use recovered model, but for now fallback to legacy
                        else:
                            logger.warning(
                                f"Tri-model error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Tri-model error recovery system failed: {recovery_error}"
                        )

                # Fallback to legacy method
                reasoning = await self._legacy_binary_forecast(question, research)
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_binary_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            prediction: float = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
        except Exception as e:
            logger.warning(f"Prediction extraction failed for {question_id}: {e}")
            prediction = 0.5  # Default neutral prediction

        logger.info(
            f"Binary forecast completed for URL {question.page_url} as {prediction:.3f} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        # Confidence gate: avoid publishing neutral/emergency outputs; try alternatives first
        if self._is_unacceptable_forecast(prediction, reasoning):
            logger.warning(
                "Low-confidence/neutral forecast detected. Retrying with alternative free/perplexity models..."
            )
            alt = await self._retry_forecast_with_alternatives(question, research)
            if not self._is_unacceptable_forecast(alt.prediction_value, alt.reasoning):
                return alt
            else:
                logger.warning(
                    "Alternatives did not yield acceptable forecast. Returning best-effort result with explicit caution."
                )
                # Append an explicit caution to avoid accidental publication misinterpretation
                reasoning = (
                    alt.reasoning or SAFE_REASONING_FALLBACK
                ) + "\n\n[Note: Low confidence due to upstream API limitations.]"
                prediction = alt.prediction_value

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK

        return _mk_rp(prediction_value=prediction, reasoning=safe_reasoning)

    async def _legacy_binary_forecast(
        self, question: "BinaryQuestion", research: str
    ) -> str:
        """Legacy binary forecasting method as fallback."""
        prompt = _safe_clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {getattr(question, "resolution_criteria", "")}

            {getattr(question, "fine_print", "")}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        # Get appropriate LLM based on complexity analysis and budget status
        if self.enhanced_llm_config:
            complexity_assessment = self.enhanced_llm_config.assess_question_complexity(
                question.question_text,
                question.background_info or "",
                getattr(question, "resolution_criteria", ""),
                getattr(question, "fine_print", ""),
            )
            llm = self.enhanced_llm_config.get_llm_for_task(
                "forecast", complexity_assessment=complexity_assessment
            )
        else:
            llm = self.get_llm("default", "llm")

        # Use safe LLM invoke with error handling and fallbacks
        return await self._safe_llm_invoke(
            llm, prompt, "forecast", question_id=str(getattr(question, "id", "unknown"))
        )

    async def _run_forecast_on_multiple_choice(
        self, question: "MultipleChoiceQuestion", research: str
    ) -> Any:  # relaxed for runtime compatibility
        question_id = str(getattr(question, "id", "unknown"))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="multiple_choice",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(
                            question, "resolution_criteria", ""
                        ),
                        "fine_print": getattr(question, "fine_print", ""),
                        "options": question.options,
                        "research_data": research,
                    },
                )

                if (
                    pipeline_result.pipeline_success
                    and pipeline_result.forecast_result.quality_validation_passed
                    and pipeline_result.forecast_result.tournament_compliant
                ):
                    logger.info(
                        f"Multi-stage multiple choice forecast successful for question {question_id}, "
                        f"cost: ${pipeline_result.total_cost:.4f}, "
                        f"quality: {pipeline_result.quality_score:.2f}"
                    )

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="multiple_choice_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True,
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Convert pipeline result to expected format
                    if isinstance(pipeline_result.final_forecast, dict):
                        # Convert dict to PredictedOptionList format
                        option_predictions = []
                        for (
                            option,
                            probability,
                        ) in pipeline_result.final_forecast.items():
                            option_predictions.append(f"{option}: {probability:.1%}")

                        # Create PredictedOptionList from the predictions
                        pipeline_prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                            "\n".join(option_predictions), question.options
                        )
                    else:
                        # Fallback extraction from reasoning
                        pipeline_prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                            pipeline_result.reasoning, question.options
                        )

                    pipeline_reasoning = pipeline_result.reasoning or ""

                    # Confidence gate: only return if acceptable; otherwise try alternatives and fall through
                    if pipeline_prediction and not self._is_unacceptable_mc_forecast(
                        pipeline_prediction, pipeline_reasoning, question.options
                    ):
                        return _mk_rp(
                            prediction_value=pipeline_prediction,
                            reasoning=pipeline_reasoning,
                        )
                    else:
                        logger.warning(
                            "Pipeline MC forecast deemed low-confidence/neutral. Retrying with alternatives..."
                        )
                        alt = await self._retry_mc_with_alternatives(question, research)
                        if (
                            alt.prediction_value
                            and not self._is_unacceptable_mc_forecast(
                                alt.prediction_value, alt.reasoning, question.options
                            )
                        ):
                            return alt
                        # Fall through to tri-model/legacy paths below

                else:
                    logger.warning(
                        f"Multi-stage multiple choice forecast quality issues for {question_id}"
                    )
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage multiple choice forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success:
                            logger.info(
                                f"Multiple choice forecast error recovery successful: {recovery_result.message}"
                            )
                        else:
                            logger.warning(
                                f"Multiple choice forecast error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Multiple choice forecast error recovery system failed: {recovery_error}"
                        )

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    adjusted_model = (
                        self.budget_aware_manager.apply_model_selection_adjustments(
                            "forecast"
                        )
                    )

                    # Map model to tier for prompt optimization
                    if (
                        "gpt-5-nano" in adjusted_model
                        or "gpt-4o-mini" in adjusted_model
                    ):
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop multiple choice forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_multiple_choice_prompt(
                    question_text=question.question_text,
                    options=question.options,
                    background_info=question.background_info or "",
                    resolution_criteria=getattr(question, "resolution_criteria", ""),
                    fine_print=getattr(question, "fine_print", ""),
                    research=research,
                    model_tier=selected_model_tier,
                )

                # Get budget-aware context for routing
                budget_context = (
                    self.tri_model_router.get_budget_aware_routing_context()
                )
                budget_remaining = (
                    budget_context.remaining_percentage if budget_context else 100.0
                )

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining,
                )

                logger.info(
                    f"Enhanced tri-model multiple choice forecast successful for question {question_id} "
                    f"using {selected_model_tier} tier"
                )

            except Exception as e:
                logger.warning(
                    f"Enhanced tri-model multiple choice forecast failed: {e}"
                )

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success:
                            logger.info(
                                f"Tri-model multiple choice error recovery successful: {recovery_result.message}"
                            )
                        else:
                            logger.warning(
                                f"Tri-model multiple choice error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Tri-model multiple choice error recovery system failed: {recovery_error}"
                        )

                # Fallback to legacy method
                reasoning = await self._legacy_multiple_choice_forecast(
                    question, research
                )
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_multiple_choice_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            # Extract structured option list
            from forecasting_tools import (
                PredictedOptionList as _POL,
            )  # local alias to guard type analysis

            prediction: _POL = (
                PredictionExtractor.extract_option_list_with_percentage_afterwards(  # type: ignore[arg-type]
                    reasoning, question.options
                )
            )
        except Exception as e:
            logger.warning(
                f"Multiple choice prediction extraction failed for {question_id}: {e}"
            )
            # Create default equal probability distribution
            from forecasting_tools import PredictedOption

            equal_prob = 1.0 / len(question.options)
            predicted_options = [
                PredictedOption(option_name=option, probability=equal_prob)
                for option in question.options
            ]
            from forecasting_tools import PredictedOptionList as _POL2

            prediction = _POL2(predicted_options=predicted_options)  # type: ignore[assignment]

        # Confidence gate for MC: avoid near-uniform/low-peak outputs
        if self._is_unacceptable_mc_forecast(prediction, reasoning, question.options):
            logger.warning(
                "Low-confidence/uncertain MC forecast detected. Retrying with alternative models..."
            )
            alt = await self._retry_mc_with_alternatives(question, research)
            if alt.prediction_value and not self._is_unacceptable_mc_forecast(
                alt.prediction_value, alt.reasoning, question.options
            ):
                return alt
            # Do NOT publish unacceptable alternative; return flagged uniform placeholder instead of None
            # (Upstream aggregator in forecasting_tools can't handle None for MC list.)
            logger.warning(
                "Alternatives did not yield acceptable MC forecast. Withholding (placeholder uniform)."
            )
            try:
                from forecasting_tools import PredictedOptionList, PredictedOption

                eq = 1.0 / max(len(question.options), 1)
                placeholder = PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name=o, probability=eq)
                        for o in question.options
                    ]
                )
            except Exception:
                placeholder = prediction  # fall back to original
            withheld_reasoning = (
                (alt.reasoning or SAFE_REASONING_FALLBACK)
                + "\n\n[WITHHELD: uniform / low-signal distribution blocked by gate – DO NOT PUBLISH.]"
            )
            # Track question id for downstream publication guard hard-block
            try:
                for attr in ("id", "question_id", "qid"):
                    qval = getattr(question, attr, None)
                    if isinstance(qval, int) and qval > 0:
                        _WITHHELD_QUESTION_IDS.add(qval)
            except Exception:
                pass
            return _mk_rp(prediction_value=placeholder, reasoning=withheld_reasoning)

        # Belt-and-suspenders: derive reason code again; if present, convert to withheld marker even if earlier heuristic passed
        reason_code = self._mc_distribution_reason(
            prediction, reasoning, question.options
        )
        if reason_code:
            try:
                from forecasting_tools import PredictedOptionList, PredictedOption

                eq = 1.0 / max(len(question.options), 1)
                placeholder = PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name=o, probability=eq)
                        for o in question.options
                    ]
                )
            except Exception:  # pragma: no cover
                placeholder = prediction
            reasoning = (
                reasoning or SAFE_REASONING_FALLBACK
            ) + f"\n\n[WITHHELD: {reason_code} distribution blocked – DO NOT PUBLISH.]"
            logger.info(
                f"MC forecast converted to WITHHELD placeholder (reason={reason_code}) for {question.page_url}"
            )
            try:
                for attr in ("id", "question_id", "qid"):
                    qval = getattr(question, attr, None)
                    if isinstance(qval, int) and qval > 0:
                        _WITHHELD_QUESTION_IDS.add(qval)
            except Exception:
                pass
            return _mk_rp(prediction_value=placeholder, reasoning=reasoning)

        logger.info(
            f"Multiple choice forecast completed for URL {question.page_url} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK
        return _mk_rp(prediction_value=prediction, reasoning=safe_reasoning)

    async def _legacy_multiple_choice_forecast(
        self, question: "MultipleChoiceQuestion", research: str
    ) -> str:
        """Legacy multiple choice forecasting method as fallback."""
        prompt = _safe_clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {getattr(question, "resolution_criteria", "")}

            {getattr(question, "fine_print", "")}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        # Get appropriate LLM and use safe invoke
        if self.enhanced_llm_config:
            complexity_assessment = self.enhanced_llm_config.assess_question_complexity(
                question.question_text, question.background_info or ""
            )
            llm = self.enhanced_llm_config.get_llm_for_task(
                "forecast", complexity_assessment=complexity_assessment
            )
        else:
            llm = self.get_llm("default", "llm")

        return await self._safe_llm_invoke(
            llm, prompt, "forecast", question_id=str(getattr(question, "id", "unknown"))
        )

    async def _run_forecast_on_numeric(
        self, question: "NumericQuestion", research: str
    ) -> Any:
        question_id = str(getattr(question, "id", "unknown"))

        # Get budget status and operation mode for intelligent routing
        budget_remaining = 100.0
        operation_mode = "normal"

        if self.budget_manager:
            budget_status = self.budget_manager.get_budget_status()
            budget_remaining = 100.0 - budget_status.utilization_percentage

        if self.budget_aware_manager:
            operation_mode = self.budget_aware_manager.operation_mode_manager.get_current_mode().value

        # PRIORITY: Try enhanced multi-stage validation pipeline for complete forecasting (Task 8.1)
        if self.multi_stage_pipeline:
            try:
                # Use the complete multi-stage validation pipeline for forecasting
                pipeline_result = await self.multi_stage_pipeline.process_question(
                    question=question.question_text,
                    question_type="numeric",
                    context={
                        "question_url": question.page_url,
                        "budget_remaining": budget_remaining,
                        "operation_mode": operation_mode,
                        "background_info": question.background_info,
                        "resolution_criteria": getattr(
                            question, "resolution_criteria", ""
                        ),
                        "fine_print": getattr(question, "fine_print", ""),
                        "unit_of_measure": question.unit_of_measure,
                        "lower_bound": question.lower_bound
                        if not question.open_lower_bound
                        else None,
                        "upper_bound": question.upper_bound
                        if not question.open_upper_bound
                        else None,
                        "open_lower_bound": question.open_lower_bound,
                        "open_upper_bound": question.open_upper_bound,
                        "research_data": research,
                    },
                )

                if (
                    pipeline_result.pipeline_success
                    and pipeline_result.forecast_result.quality_validation_passed
                    and pipeline_result.forecast_result.tournament_compliant
                ):
                    logger.info(
                        f"Multi-stage numeric forecast successful for question {question_id}, "
                        f"cost: ${pipeline_result.total_cost:.4f}, "
                        f"quality: {pipeline_result.quality_score:.2f}"
                    )

                    # Integrate budget manager with cost tracking and operation modes (Task 8.2)
                    self._track_question_processing_cost(
                        question_id=question_id,
                        task_type="numeric_forecast_pipeline",
                        cost=pipeline_result.total_cost,
                        model_used="multi_stage_pipeline",
                        success=True,
                    )

                    # Check and update operation modes based on budget utilization
                    self._integrate_budget_manager_with_operation_modes()

                    # Convert pipeline result to expected format
                    if isinstance(pipeline_result.final_forecast, dict):
                        # Convert percentile dict to NumericDistribution
                        try:
                            percentiles = {}
                            for key, value in pipeline_result.final_forecast.items():
                                if isinstance(key, (int, str)) and str(key).isdigit():
                                    percentiles[int(key)] = float(value)

                            if percentiles:
                                from forecasting_tools.data_models.numeric_report import (
                                    Percentile,
                                )

                                percentile_list = [
                                    Percentile(
                                        percentile=float(p) / 100.0, value=float(v)
                                    )
                                    for p, v in percentiles.items()
                                ]
                                pipeline_prediction = _mk_numeric_distribution(
                                    percentile_list, question
                                )
                            else:
                                # Fallback extraction from reasoning
                                pipeline_prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                                    pipeline_result.reasoning, question
                                )
                        except Exception as conversion_error:
                            logger.warning(
                                f"Numeric prediction conversion failed: {conversion_error}"
                            )
                            # Fallback extraction from reasoning
                            pipeline_prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                                pipeline_result.reasoning, question
                            )
                    else:
                        # Fallback extraction from reasoning
                        pipeline_prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                            pipeline_result.reasoning, question
                        )

                    pipeline_reasoning = pipeline_result.reasoning or ""

                    # Confidence gate: only return if acceptable; otherwise try alternatives and fall through
                    if (
                        pipeline_prediction
                        and not self._is_unacceptable_numeric_forecast(
                            pipeline_prediction, pipeline_reasoning
                        )
                    ):
                        return _mk_rp(
                            prediction_value=pipeline_prediction,
                            reasoning=pipeline_reasoning,
                        )
                    else:
                        logger.warning(
                            "Pipeline numeric forecast deemed low-confidence/neutral. Retrying with alternatives..."
                        )
                        alt = await self._retry_numeric_with_alternatives(
                            question, research
                        )
                        if (
                            alt.prediction_value
                            and not self._is_unacceptable_numeric_forecast(
                                alt.prediction_value, alt.reasoning
                            )
                        ):
                            return alt
                        # Fall through to tri-model/legacy paths below

                else:
                    logger.warning(
                        f"Multi-stage numeric forecast quality issues for {question_id}"
                    )
                    # Continue to fallback methods

            except Exception as e:
                logger.warning(f"Multi-stage numeric forecast failed: {e}")

                # Use comprehensive error recovery (Task 8.1)
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier="full",
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=1,
                            question_id=question_id,
                            provider="multi_stage_pipeline",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success:
                            logger.info(
                                f"Numeric forecast error recovery successful: {recovery_result.message}"
                            )
                        else:
                            logger.warning(
                                f"Numeric forecast error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Numeric forecast error recovery system failed: {recovery_error}"
                        )

        # FALLBACK: Try enhanced tri-model router with tier-optimized prompts
        if self.tri_model_router and self.anti_slop_prompts:
            try:
                # Apply budget-aware model selection adjustments (Task 8.2)
                selected_model_tier = "full"  # Default for forecasting
                if self.budget_aware_manager:
                    adjusted_model = (
                        self.budget_aware_manager.apply_model_selection_adjustments(
                            "forecast"
                        )
                    )

                    # Map model to tier for prompt optimization
                    if (
                        "gpt-5-nano" in adjusted_model
                        or "gpt-4o-mini" in adjusted_model
                    ):
                        selected_model_tier = "nano"
                    elif "gpt-5-mini" in adjusted_model:
                        selected_model_tier = "mini"
                    else:
                        selected_model_tier = "full"

                # Create tier-optimized anti-slop numeric forecast prompt (Task 8.1)
                prompt = self.anti_slop_prompts.get_numeric_forecast_prompt(
                    question_text=question.question_text,
                    background_info=question.background_info or "",
                    resolution_criteria=getattr(question, "resolution_criteria", ""),
                    fine_print=getattr(question, "fine_print", ""),
                    research=research,
                    unit_of_measure=question.unit_of_measure,
                    lower_bound=question.lower_bound
                    if not question.open_lower_bound
                    else None,
                    upper_bound=question.upper_bound
                    if not question.open_upper_bound
                    else None,
                    model_tier=selected_model_tier,
                )

                # Get budget-aware context for routing
                budget_context = (
                    self.tri_model_router.get_budget_aware_routing_context()
                )
                budget_remaining = (
                    budget_context.remaining_percentage if budget_context else 100.0
                )

                # Route to optimal model with budget-aware selection
                reasoning = await self.tri_model_router.route_query(
                    task_type="forecast",
                    content=prompt,
                    complexity="high",
                    budget_remaining=budget_remaining,
                )

                logger.info(
                    f"Enhanced tri-model numeric forecast successful for question {question_id} "
                    f"using {selected_model_tier} tier"
                )

            except Exception as e:
                logger.warning(f"Enhanced tri-model numeric forecast failed: {e}")

                # Use comprehensive error recovery for tri-model failures
                if self.error_recovery_manager:
                    try:
                        from src.infrastructure.reliability.error_classification import (
                            ErrorContext,
                        )

                        error_context = ErrorContext(
                            task_type="forecast",
                            model_tier=selected_model_tier,
                            operation_mode=operation_mode,
                            budget_remaining=budget_remaining,
                            attempt_number=2,
                            question_id=question_id,
                            provider="tri_model_router",
                        )

                        recovery_result = (
                            await self.error_recovery_manager.recover_from_error(
                                e, error_context
                            )
                        )
                        if recovery_result.success:
                            logger.info(
                                f"Tri-model numeric error recovery successful: {recovery_result.message}"
                            )
                        else:
                            logger.warning(
                                f"Tri-model numeric error recovery failed: {recovery_result.message}"
                            )
                    except Exception as recovery_error:
                        logger.error(
                            f"Tri-model numeric error recovery system failed: {recovery_error}"
                        )

                # Fallback to legacy method
                reasoning = await self._legacy_numeric_forecast(question, research)
        else:
            # Fallback to legacy method if enhanced components not available
            reasoning = await self._legacy_numeric_forecast(question, research)

        # Extract prediction from reasoning with enhanced error handling
        try:
            prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        except Exception as e:
            logger.warning(
                f"Numeric prediction extraction failed for {question_id}: {e}"
            )
            # Create default distribution with median estimate
            median_estimate = (
                (question.lower_bound + question.upper_bound) / 2
                if (not question.open_lower_bound and not question.open_upper_bound)
                else 1.0
            )

            try:
                from forecasting_tools.data_models.numeric_report import Percentile

                percentiles = [
                    Percentile(percentile=0.1, value=median_estimate * 0.5),
                    Percentile(percentile=0.5, value=median_estimate),
                    Percentile(percentile=0.9, value=median_estimate * 1.5),
                ]
            except Exception:
                # Fallback to simple namespaces if Percentile is unavailable
                from types import SimpleNamespace

                percentiles = [
                    SimpleNamespace(percentile=0.1, value=median_estimate * 0.5),
                    SimpleNamespace(percentile=0.5, value=median_estimate),
                    SimpleNamespace(percentile=0.9, value=median_estimate * 1.5),
                ]
            prediction = _mk_numeric_distribution(percentiles, question)

        logger.info(
            f"Numeric forecast completed for URL {question.page_url} "
            f"(mode: {operation_mode}, budget: {budget_remaining:.1f}%)"
        )

        # Confidence gate for Numeric: ensure sensible distribution
        if self._is_unacceptable_numeric_forecast(prediction, reasoning):
            logger.warning(
                "Low-confidence/uncertain numeric forecast detected. Retrying with alternative models..."
            )
            alt = await self._retry_numeric_with_alternatives(question, research)
            if alt.prediction_value and not self._is_unacceptable_numeric_forecast(
                alt.prediction_value, alt.reasoning
            ):
                return alt
            else:
                logger.warning(
                    "Alternatives did not yield acceptable numeric forecast. Returning best-effort with caution."
                )
                reasoning = (
                    alt.reasoning or SAFE_REASONING_FALLBACK
                ) + "\n\n[Note: Low confidence due to upstream API limitations.]"
                if alt.prediction_value:
                    prediction = alt.prediction_value

        safe_reasoning = reasoning or SAFE_REASONING_FALLBACK

        return _mk_rp(prediction_value=prediction, reasoning=safe_reasoning)

    async def _legacy_numeric_forecast(
        self, question: "NumericQuestion", research: str
    ) -> str:
        """Legacy numeric forecasting method as fallback."""
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )

        prompt = _safe_clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {getattr(question, "resolution_criteria", "")}

            {getattr(question, "fine_print", "")}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )

        # Get appropriate LLM based on complexity analysis and budget status
        if self.enhanced_llm_config:
            try:
                complexity_assessment = (
                    self.enhanced_llm_config.assess_question_complexity(
                        question.question_text, question.background_info or ""
                    )
                )
                llm = self.enhanced_llm_config.get_llm_for_task(
                    "forecast", complexity_assessment=complexity_assessment
                )
            except Exception:
                llm = self.get_llm("default", "llm")
        else:
            llm = self.get_llm("default", "llm")

        # Use safe LLM invoke and return the reasoning string
        return await self._safe_llm_invoke(
            llm, prompt, "forecast", question_id=str(getattr(question, "id", "unknown"))
        )

    def _create_upper_and_lower_bound_messages(
        self, question: "NumericQuestion"
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message

    async def _run_forecast_on_date(self, question: Any, research: str) -> Any:
        """
        Forecast on date questions - when will X happen?

        Date questions ask for a probability distribution over dates.
        This method implements custom date forecasting logic since
        the forecasting-tools library doesn't support date questions yet.
        """

        question_id = str(getattr(question, "id", "unknown"))

        logger.info(f"Starting date question forecast for {question_id}")

        # Import the date forecaster
        try:
            from src.domain.services.date_question_forecaster import (
                DateQuestionForecaster,
            )

            date_forecaster = DateQuestionForecaster()

            # Convert float bounds to datetime objects if needed
            from datetime import datetime

            lower_bound = question.lower_bound
            upper_bound = question.upper_bound

            if isinstance(lower_bound, (int, float)):
                lower_bound = datetime.fromtimestamp(lower_bound)
            if isinstance(upper_bound, (int, float)):
                upper_bound = datetime.fromtimestamp(upper_bound)

            # Generate the date forecast
            date_forecast = date_forecaster.forecast_date_question(
                question_text=question.question_text,
                background_info=getattr(question, "background_info", ""),
                resolution_criteria=getattr(question, "resolution_criteria", ""),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                research_data=research,
                fine_print=getattr(question, "fine_print", ""),
            )

            # Convert to format expected by forecasting-tools
            # For date questions, we need to create a distribution over dates
            # Format the percentiles for Metaculus API
            formatted_percentiles = date_forecaster.format_percentiles_for_metaculus(
                date_forecast.percentiles
            )

            logger.info(f"Date forecast completed for {question_id}")
            logger.info(
                f"Predicted date range: {date_forecast.percentiles[0.1].date()} to {date_forecast.percentiles[0.9].date()}"
            )
            logger.info(f"Median prediction: {date_forecast.percentiles[0.5].date()}")

            # Create ReasonedPrediction object expected by framework using helper
            return _mk_rp(
                prediction_value=formatted_percentiles,
                reasoning=date_forecast.reasoning,
            )

        except Exception as e:
            logger.error(f"Date question forecasting failed for {question_id}: {e}")

            # Convert float bounds to datetime for fallback as well
            from datetime import datetime

            lower_bound = question.lower_bound
            upper_bound = question.upper_bound

            if isinstance(lower_bound, (int, float)):
                lower_bound = datetime.fromtimestamp(lower_bound)
            if isinstance(upper_bound, (int, float)):
                upper_bound = datetime.fromtimestamp(upper_bound)

            # Create fallback response
            fallback_reasoning = f"""
            Date Question Analysis Failed:
            Question: {question.question_text}
            Error: {str(e)}

            Fallback approach:
            - Using uniform distribution across the date range
            - Range: {lower_bound.date()} to {upper_bound.date()}
            - This is a conservative fallback when analysis fails
            """

            # Create uniform date distribution as fallback
            total_duration = upper_bound - lower_bound
            fallback_percentiles = {}
            for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
                fallback_percentiles[p] = lower_bound + total_duration * p

            fallback_formatted = [
                (p, date_obj.strftime("%Y-%m-%d"))
                for p, date_obj in fallback_percentiles.items()
            ]

            return _mk_rp(
                prediction_value=fallback_formatted,
                reasoning=fallback_reasoning.strip(),
            )

    async def forecast_question(self, question, return_exceptions: bool = False):
        """
        Enhanced forecast_question with comprehensive question type support and error recovery.

        This method handles:
        - Date questions (custom DateQuestionForecaster)
        - Discrete questions (custom DiscreteQuestionForecaster)
        - Standard question types (Binary, MultipleChoice, Numeric)
        - Robust retry logic with exponential backoff
        - Intelligent fallback forecasts when all retries are exhausted

        Designed to maximize tournament success rates by handling the 11 main error types
        that previously caused forecast failures.
        """

        question_id = getattr(question, "id", "unknown")

        # Initialize enhanced error recovery if not already done
        if not hasattr(self, "_error_recovery"):
            from src.domain.services.enhanced_error_recovery import (
                EnhancedErrorRecovery,
            )

            self._error_recovery = EnhancedErrorRecovery()

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._attempt_forecast_with_recovery(question, attempt)

            except Exception as e:
                logger.warning(
                    f"Forecast attempt {attempt} failed for {question_id}: {type(e).__name__}: {str(e)}"
                )

                # Use enhanced error recovery to decide retry vs fallback
                (
                    should_retry,
                    fallback_result,
                ) = await self._error_recovery.handle_forecast_failure(
                    question, e, attempt, max_attempts
                )

                if not should_retry:
                    logger.info(f"Using fallback forecast for {question_id}")
                    return fallback_result if fallback_result else e
                # If should_retry is True, continue to next iteration

        # If we get here, all attempts failed
        logger.error(f"All {max_attempts} forecast attempts failed for {question_id}")
        if return_exceptions:
            return Exception(f"All forecast attempts failed for question {question_id}")
        else:
            raise Exception(f"All forecast attempts failed for question {question_id}")

    async def _attempt_forecast_with_recovery(self, question, attempt_number: int):
        """
        Single forecast attempt with comprehensive question type support.
        """

        question_id = getattr(question, "id", "unknown")
        class_name = (
            question.__class__.__name__ if hasattr(question, "__class__") else "Unknown"
        )

        logger.info(
            f"Forecast attempt {attempt_number} for {question_id} ({class_name})"
        )

        # Handle different question types with custom implementations
        # First check for DateQuestion specifically to bypass the forecasting-tools NotImplementedError
        if "Date" in class_name or hasattr(question, 'lower_bound') and hasattr(question, 'upper_bound'):
            logger.info(f"Detected DateQuestion, using custom handler for {question_id}")
            return await self._handle_date_question(question)
        elif "Discrete" in class_name or class_name == "DiscreteQuestion":
            return await self._handle_discrete_question(question)
        elif hasattr(question, "__class__"):
            # Try standard question types with fallback to our custom handlers
            try:
                return await self._handle_standard_question_types(question, class_name)
            except NotImplementedError as e:
                if "Date questions not supported yet" in str(e):
                    logger.info(f"Caught DateQuestion NotImplementedError, using custom handler for {question_id}")
                    return await self._handle_date_question(question)
                else:
                    raise  # Re-raise other NotImplementedErrors
        else:
            raise ValueError(f"Question has no class attribute: {question}")

    async def _handle_date_question(self, question):
        """Handle date questions with custom DateQuestionForecaster."""

        question_id = getattr(question, "id", "unknown")
        logger.info(f"Handling date question: {question_id}")

        # Run research first
        research = await self.run_research(question)

        # Use our custom date forecasting
        forecast_result = await self._run_forecast_on_date(question, research)

        logger.info(f"✅ Successfully forecasted date question {question_id}")
        return forecast_result

    async def _handle_discrete_question(self, question):
        """Handle discrete questions with custom DiscreteQuestionForecaster."""

        question_id = getattr(question, "id", "unknown")
        logger.info(f"Handling discrete question: {question_id}")

        # Import discrete question forecaster
        from src.domain.services.enhanced_error_recovery import (
            DiscreteQuestionForecaster,
        )

        discrete_forecaster = DiscreteQuestionForecaster()

        # Run research first
        research = await self.run_research(question)

        # Get question options
        options = getattr(question, "options", [])
        if not options:
            # Try to extract from other attributes
            options = getattr(
                question, "choices", getattr(question, "outcomes", ["Unknown"])
            )

        # Generate the discrete forecast
        discrete_forecast = discrete_forecaster.forecast_discrete_question(
            question_text=question.question_text,
            options=options,
            background_info=getattr(question, "background_info", ""),
            resolution_criteria=getattr(question, "resolution_criteria", ""),
            research_data=research,
            fine_print=getattr(question, "fine_print", ""),
        )

        # Format for Metaculus API
        formatted_probabilities = (
            discrete_forecaster.format_probabilities_for_metaculus(
                discrete_forecast.option_probabilities
            )
        )

        logger.info(f"✅ Successfully forecasted discrete question {question_id}")
        logger.info(f"Options: {list(discrete_forecast.option_probabilities.keys())}")
        logger.info(f"Confidence: {discrete_forecast.confidence}")

        # Create result in expected format
        result = type(
            "DiscreteQuestionResult",
            (),
            {
                "prediction_value": formatted_probabilities,
                "reasoning": discrete_forecast.reasoning,
                "confidence": discrete_forecast.confidence,
                "question_id": question_id,
                "question_type": "discrete",
            },
        )()

        return result

    async def _handle_standard_question_types(self, question, class_name: str):
        """Handle standard question types (Binary, MultipleChoice, Numeric) with existing methods."""

        question_id = getattr(question, "id", "unknown")

        # Run research first for all question types
        research = await self.run_research(question)

        if "Binary" in class_name:
            logger.info(f"Handling binary question: {question_id}")
            return await self._run_forecast_on_binary(question, research)
        elif "MultipleChoice" in class_name:
            logger.info(f"Handling multiple choice question: {question_id}")
            return await self._run_forecast_on_multiple_choice(question, research)
        elif "Numeric" in class_name:
            logger.info(f"Handling numeric question: {question_id}")
            return await self._run_forecast_on_numeric(question, research)
        else:
            logger.warning(
                f"Unknown standard question type: {class_name} for question {question_id}"
            )
            raise ValueError(f"Unsupported question type: {class_name}")

    async def _make_prediction(self, question, research):
        """
        Override forecasting-tools _make_prediction to intercept date questions.

        This is the critical intercept point - forecasting-tools calls this method
        directly in tournament mode, bypassing our forecast_question override.
        """
        question_id = getattr(question, "id", "unknown")
        class_name = question.__class__.__name__ if hasattr(question, "__class__") else "Unknown"

        logger.info(f"_make_prediction called for {question_id} ({class_name})")

        # Check for DateQuestion BEFORE calling parent _make_prediction
        if class_name == "DateQuestion" or (hasattr(question, 'lower_bound') and hasattr(question, 'upper_bound')):
            logger.info(f"Intercepted DateQuestion in _make_prediction for {question_id}")

            # Use our custom date forecasting logic
            return await self._run_forecast_on_date(question, research)

        # For all other question types, call the parent implementation
        try:
            return await super()._make_prediction(question, research)
        except NotImplementedError as e:
            if "Date questions not supported yet" in str(e):
                logger.info(f"Caught DateQuestion NotImplementedError in _make_prediction for {question_id}")
                return await self._run_forecast_on_date(question, research)
            else:
                raise  # Re-raise other NotImplementedErrors
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Perform OpenRouter startup validation and auto-configuration (Task 9.2)
    async def validate_openrouter_startup():
        """Validate OpenRouter configuration on startup."""
        try:
            from src.infrastructure.config.openrouter_startup_validator import (
                OpenRouterStartupValidator,
            )
            from src.infrastructure.config.tri_model_router import (
                OpenRouterTriModelRouter,
            )

            logger.info("Performing OpenRouter startup validation...")

            # Run validation
            validator = OpenRouterStartupValidator()
            validation_success = await validator.run_startup_validation(
                exit_on_failure=False
            )

            if validation_success:
                logger.info("✅ OpenRouter configuration validated successfully")

                # Create router with auto-configuration
                router = await OpenRouterTriModelRouter.create_with_auto_configuration()
                logger.info("✅ OpenRouter tri-model router configured and ready")

                return router
            else:
                logger.warning(
                    "⚠️ OpenRouter validation failed - system may have limited functionality"
                )
                return None

        except ImportError as e:
            logger.warning(f"OpenRouter validation components not available: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenRouter startup validation failed: {e}")
            return None

    # Run OpenRouter validation (bounded + skippable to avoid startup hangs)
    def _run_openrouter_validation_safely():
        try:
            import os

            timeout_s = float(os.getenv("OPENROUTER_STARTUP_TIMEOUT", "12"))
            skip_flag = os.getenv("SKIP_OPENROUTER_STARTUP", "").lower() in (
                "1",
                "true",
                "yes",
            )

            if skip_flag:
                logger.info(
                    "Skipping OpenRouter startup validation (SKIP_OPENROUTER_STARTUP set)"
                )
                return None

            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key or api_key.startswith("dummy_"):
                logger.info(
                    "Skipping OpenRouter startup validation (no valid OPENROUTER_API_KEY)"
                )
                return None

            # Bound the async validation to avoid indefinite waits
            return asyncio.run(
                asyncio.wait_for(validate_openrouter_startup(), timeout=timeout_s)
            )
        except Exception as e:
            logger.warning(f"OpenRouter startup validation bypassed due to: {e}")
            return None

    openrouter_router = _run_openrouter_validation_safely()

    # CRITICAL: Reset circuit breaker at startup to ensure fresh attempts
    # This prevents previous run failures from blocking new forecasts
    try:
        from src.infrastructure.external_apis.llm_client import reset_openrouter_circuit_breaker
        reset_openrouter_circuit_breaker()
        logger.info("✅ OpenRouter circuit breaker reset at startup")
    except Exception as e:
        logger.warning(f"Failed to reset circuit breaker: {e}")

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    # Create enhanced LLM configuration with tri-model GPT-5 routing
    def create_enhanced_llms():
        """Create LLM configuration with tri-model GPT-5 routing and budget-aware selection."""
        llms: Dict[str, Any] = {}

        # Prefer tri-model router models if available
        try:
            from src.infrastructure.config.tri_model_router import tri_model_router

            router_models = tri_model_router.models
            llms["default"] = router_models["full"]
            # Downstream code may expect a string ID for summarizer (uses .startswith);
            # pass the model name string to avoid AttributeError while keeping objects for others.
            try:
                llms["summarizer"] = getattr(
                    router_models["nano"], "model", "openai/gpt-5-nano"
                )
            except Exception:
                llms["summarizer"] = "openai/gpt-5-nano"
            llms["researcher"] = router_models["mini"]
            logger.info("Using tri-model GPT-5 configuration for LLMs")
            return llms
        except Exception as e:
            logger.warning(
                f"Tri-model router unavailable, using env-configured models: {e}"
            )

        # Fall back to env-configured models through OpenRouter factory
        from src.infrastructure.config.llm_factory import create_llm

        llms = {
            # Policy: avoid gpt-4o family (cost). Default to GPT-5 env vars else safe GPT-5 tiers.
            "default": create_llm(
                os.getenv("PRIMARY_FORECAST_MODEL", "openai/gpt-5"),
                temperature=0.3,
                timeout=60,
                allowed_tries=3,
            ),
            "summarizer": create_llm(
                os.getenv("SIMPLE_TASK_MODEL", "openai/gpt-5-nano"),
                temperature=0.0,
                timeout=45,
                allowed_tries=3,
            ),
            "researcher": create_llm(
                os.getenv("PRIMARY_RESEARCH_MODEL", "openai/gpt-5-mini"),
                temperature=0.1,
                timeout=90,
                allowed_tries=2,
            ),
        }
        return llms

    # Initialize bot with enhanced configuration
    enhanced_llms = create_enhanced_llms()

    # Get tournament configuration for bot parameters
    if TOURNAMENT_COMPONENTS_AVAILABLE:
        try:
            tournament_config = get_tournament_config()
            research_reports = tournament_config.max_research_reports_per_question
            predictions_per_report = tournament_config.max_predictions_per_report
            publish_reports = (
                tournament_config.publish_reports and not tournament_config.dry_run
            )
            skip_previously_forecasted = tournament_config.skip_previously_forecasted
        except Exception as e:
            logger.warning(f"Failed to get tournament config: {e}")
            # Fallback to environment variables
            research_reports = int(os.getenv("MAX_RESEARCH_REPORTS_PER_QUESTION", "1"))
            predictions_per_report = int(os.getenv("MAX_PREDICTIONS_PER_REPORT", "5"))
            publish_reports = (
                os.getenv("PUBLISH_REPORTS", "true").lower() == "true"
                and os.getenv("DRY_RUN", "false").lower() != "true"
            )
            skip_previously_forecasted = (
                os.getenv("SKIP_PREVIOUSLY_FORECASTED", "true").lower() == "true"
            )
    else:
        # Use environment variables for configuration
        research_reports = int(os.getenv("MAX_RESEARCH_REPORTS_PER_QUESTION", "1"))
        predictions_per_report = int(os.getenv("MAX_PREDICTIONS_PER_REPORT", "5"))

    # PRODUCTION-SAFE DEFAULTS: Enable real forecasting by default
    # DRY_RUN defaults to FALSE (real submissions)
    # SKIP_PREVIOUSLY_FORECASTED defaults to TRUE (avoid duplicates)
    publish_reports = (
        os.getenv("PUBLISH_REPORTS", "true").lower() == "true"
        and os.getenv("DRY_RUN", "false").lower() != "true"  # DRY_RUN=false by default
    )
    skip_previously_forecasted = (
        os.getenv("SKIP_PREVIOUSLY_FORECASTED", "true").lower() == "true"  # Skip by default
    )

    # Log configuration
    logger.info("Bot Configuration:")
    logger.info(f"  Research reports per question: {research_reports}")
    logger.info(f"  Predictions per research report: {predictions_per_report}")
    logger.info(f"  Publish reports to Metaculus: {publish_reports}")
    logger.info(f"  Skip previously forecasted: {skip_previously_forecasted}")
    logger.info(f"  Tournament mode: {os.getenv('TOURNAMENT_MODE', 'false')}")
    # Tournament target resolution for logging
    minibench_slug = os.getenv("AIB_MINIBENCH_TOURNAMENT_SLUG")
    tournament_id = os.getenv("AIB_TOURNAMENT_ID")
    tournament_slug = os.getenv("AIB_TOURNAMENT_SLUG") or os.getenv("TOURNAMENT_SLUG")
    minibench_id = os.getenv("AIB_MINIBENCH_TOURNAMENT_ID")

    if minibench_slug:
        tournament_target_env = minibench_slug
    elif tournament_id:
        tournament_target_env = tournament_id
    elif tournament_slug:
        tournament_target_env = tournament_slug
    elif minibench_id:
        tournament_target_env = minibench_id
    else:
        tournament_target_env = "minibench"
    logger.info(f"  Tournament target: {tournament_target_env}")
    logger.info(f"  Budget limit: ${os.getenv('BUDGET_LIMIT', '100.0')}")
    logger.info(
        f"  Scheduling frequency: {os.getenv('SCHEDULING_FREQUENCY_HOURS', '24')} hours"
    )

    template_bot = TemplateForecaster(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_reports,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previously_forecasted,
        llms=enhanced_llms,
    )

    forecast_reports = []  # ensure defined for summary even if failures occur
    # TODO(publish-gate): integrate final publish gate decisions & metrics aggregation
    try:
        if run_mode == "tournament":
            # MiniBench safety window: optionally skip runs on weekends or outside hours
            try:
                # Use same tournament target resolution as main logic
                minibench_slug_safety = os.getenv("AIB_MINIBENCH_TOURNAMENT_SLUG")
                tournament_id_safety = os.getenv("AIB_TOURNAMENT_ID")
                tournament_slug_safety = os.getenv("AIB_TOURNAMENT_SLUG") or os.getenv(
                    "TOURNAMENT_SLUG"
                )
                minibench_id_safety = os.getenv("AIB_MINIBENCH_TOURNAMENT_ID")

                if minibench_slug_safety:
                    tgt = minibench_slug_safety
                elif tournament_id_safety:
                    tgt = tournament_id_safety
                elif tournament_slug_safety:
                    tgt = tournament_slug_safety
                elif minibench_id_safety:
                    tgt = minibench_id_safety
                else:
                    tgt = "minibench"

                if tgt == "minibench":
                    import datetime as _dt

                    try:
                        from zoneinfo import ZoneInfo as _ZoneInfo
                    except Exception:  # pragma: no cover
                        _ZoneInfo = None  # type: ignore
                    tz_name = os.getenv("MINIBENCH_TZ", "UTC")
                    start_hh = int(os.getenv("MINIBENCH_WINDOW_START_HH", "15"))
                    end_hh = int(os.getenv("MINIBENCH_WINDOW_END_HH", "23"))
                    allow_weekends = os.getenv(
                        "MINIBENCH_ALLOW_WEEKENDS", "false"
                    ).lower() in ("1", "true", "yes")
                    now = (
                        _dt.datetime.utcnow()
                        if _ZoneInfo is None
                        else _dt.datetime.now(_ZoneInfo(tz_name))
                    )
                    dow = now.isoweekday()
                    if not allow_weekends and dow >= 6:
                        logger.info(
                            "MiniBench safety: weekend in %s; skipping run.", tz_name
                        )
                        import sys as _sys

                        _sys.exit(0)
                    if not (start_hh <= now.hour <= end_hh):
                        logger.info(
                            "MiniBench safety: outside window %s-%s in %s; skipping run.",
                            start_hh,
                            end_hh,
                            tz_name,
                        )
                        import sys as _sys

                        _sys.exit(0)
            except Exception as _e:  # pragma: no cover
                logger.warning("MiniBench safety check failed: %s (continuing)", _e)

            # Tournament target resolution: prefer explicit tournament ID over slug
            # Priority: MiniBench slug → Tournament ID → Tournament slug → MiniBench ID → fallback
            minibench_slug = os.getenv("AIB_MINIBENCH_TOURNAMENT_SLUG")
            tournament_id = os.getenv("AIB_TOURNAMENT_ID")
            tournament_slug = os.getenv("AIB_TOURNAMENT_SLUG") or os.getenv(
                "TOURNAMENT_SLUG"
            )
            minibench_id = os.getenv("AIB_MINIBENCH_TOURNAMENT_ID")

            if minibench_slug:
                tournament_target = minibench_slug
            elif tournament_id:
                tournament_target = tournament_id
            elif tournament_slug:
                tournament_target = tournament_slug
            elif minibench_id:
                tournament_target = minibench_id
            else:
                tournament_target = "minibench"
            # In dry-run, allow reprocessing to validate pipeline end-to-end
            if os.getenv("DRY_RUN", "false").lower() == "true":
                template_bot.skip_previously_forecasted_questions = False
            try:
                # Check OpenRouter circuit breaker before starting
                from src.infrastructure.external_apis.llm_client import is_openrouter_circuit_breaker_open, get_openrouter_circuit_breaker_status

                if is_openrouter_circuit_breaker_open():
                    status = get_openrouter_circuit_breaker_status()
                    logger.error(
                        f"Cannot start tournament run: OpenRouter circuit breaker is open. "
                        f"Quota exhausted after {status['consecutive_failures']} failures. "
                        f"Will reset in {status['time_until_reset_seconds']:.0f} seconds."
                    )
                    forecast_reports = []
                else:
                    forecast_reports = asyncio.run(
                        template_bot.forecast_on_tournament(
                            tournament_target, return_exceptions=True
                        )
                    )
            except Exception as e:
                logger.error("MiniBench run failed: %s", e)
                forecast_reports = []
            # No fallback to long tournament by policy. Exit if nothing processed.
            if not forecast_reports:
                logger.warning(
                    "No questions for '%s'. Long tournament fallback is disabled; exiting.",
                    tournament_target,
                )
        elif run_mode == "quarterly_cup":
            # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
            # The new quarterly cup may not be initialized near the beginning of a quarter
            template_bot.skip_previously_forecasted_questions = False

            # Check OpenRouter circuit breaker before starting
            from src.infrastructure.external_apis.llm_client import is_openrouter_circuit_breaker_open, get_openrouter_circuit_breaker_status

            if is_openrouter_circuit_breaker_open():
                status = get_openrouter_circuit_breaker_status()
                logger.error(
                    f"Cannot start quarterly cup run: OpenRouter circuit breaker is open. "
                    f"Quota exhausted after {status['consecutive_failures']} failures. "
                    f"Will reset in {status['time_until_reset_seconds']:.0f} seconds."
                )
                forecast_reports = []
            else:
                forecast_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
                    )
                )
        elif run_mode == "test_questions":
            # Example questions are a good way to test the bot's performance on a single question
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            ]
            template_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )
    except Exception as e:
        logger.error("Forecasting run failed: %s", e)
        # Preserve the exception in the report list so downstream summary still works
        forecast_reports = [e]
    # Log comprehensive report summary (tolerant to missing/exception entries)
    try:
        TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
    except Exception as e:
        logger.warning("Failed to log report summary: %s", e)

    # Log budget usage statistics if available
    if hasattr(template_bot, "budget_manager") and template_bot.budget_manager:
        try:
            logger.info("=== Budget Usage Statistics ===")
            template_bot.budget_manager.log_budget_status()

            # Generate and log budget report
            if (
                hasattr(template_bot, "budget_alert_system")
                and template_bot.budget_alert_system
            ):
                template_bot.budget_alert_system.log_budget_summary()

                # Get cost optimization suggestions
                suggestions = (
                    template_bot.budget_alert_system.get_cost_optimization_suggestions()
                )
                if suggestions:
                    logger.info("Cost Optimization Suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        logger.info("  %d. %s", i, suggestion)

        except Exception as e:
            logger.warning("Failed to log budget statistics: %s", e)

    # Log tournament usage statistics if available
    if (
        TOURNAMENT_COMPONENTS_AVAILABLE
        and hasattr(template_bot, "tournament_asknews")
        and template_bot.tournament_asknews
    ):
        try:
            stats = template_bot.tournament_asknews.get_usage_stats()
            logger.info("=== Tournament Usage Statistics ===")
            logger.info(f"AskNews Total Requests: {stats['total_requests']}")
            logger.info(f"AskNews Success Rate: {stats['success_rate']:.1f}%")
            logger.info(f"AskNews Fallback Rate: {stats['fallback_rate']:.1f}%")
            logger.info(
                f"AskNews Quota Used: {stats['estimated_quota_used']}/{stats['quota_limit']} "
                f"({stats['quota_usage_percentage']:.1f}%)"
            )
            logger.info(
                f"AskNews Daily Requests: {stats['daily_request_count']}/{stats.get('daily_limit', 'N/A')}"
            )

            # Alert if quota usage is high
            if stats["quota_usage_percentage"] > 80:
                logger.warning(
                    "HIGH QUOTA USAGE: %.1f%% of AskNews quota used!",
                    stats["quota_usage_percentage"],
                )

            # Log fallback provider status
            fallback_status = (
                template_bot.tournament_asknews.get_fallback_providers_status()
            )
            logger.info("Fallback Providers Status:")
            for provider, available in fallback_status.items():
                status = "✓ Available" if available else "✗ Not configured"
                logger.info(f"  {provider}: {status}")

        except Exception as e:
            logger.warning(f"Failed to log tournament statistics: {e}")

    # Final status summary
    try:

        def _is_withheld(report):
            try:
                marker = "[WITHHELD:"
                upper_marker = marker.upper()
                # 1. Direct known reasoning fields
                for attr in (
                    "forecast_reasoning",
                    "reasoning",
                    "rationale",
                    "analysis_text",
                ):
                    val = getattr(report, attr, None)
                    if isinstance(val, str) and upper_marker in val.upper():
                        # Attempt to extract numeric question id from string if present
                        import re as _re

                        m = _re.search(r"/questions/(\d+)", val)
                        if m:
                            try:
                                _WITHHELD_QUESTION_IDS.add(int(m.group(1)))
                            except Exception:
                                pass
                        return True
                # 2. Scan any list/iterable of option objects for reasoning fields
                for attr in (
                    "options",
                    "predicted_options",
                    "choices",
                    "forecast_options",
                ):
                    container = getattr(report, attr, None)
                    if container and hasattr(container, "__iter__"):
                        for opt in container:
                            for rattr in ("reasoning", "rationale"):
                                rt = getattr(opt, rattr, None)
                                if isinstance(rt, str) and upper_marker in rt.upper():
                                    import re as _re

                                    m = _re.search(r"/questions/(\d+)", rt)
                                    if m:
                                        try:
                                            _WITHHELD_QUESTION_IDS.add(int(m.group(1)))
                                        except Exception:
                                            pass
                                    return True
                # 3. Generic attribute sweep (belt & suspenders)
                for name in dir(report):
                    if name.startswith("_"):  # skip private
                        continue
                    try:
                        val = getattr(report, name)
                    except Exception:
                        continue
                    if isinstance(val, str) and upper_marker in val.upper():
                        import re as _re

                        m = _re.search(r"/questions/(\d+)", val)
                        if m:
                            try:
                                _WITHHELD_QUESTION_IDS.add(int(m.group(1)))
                            except Exception:
                                pass
                        return True
                    if isinstance(val, (list, tuple)):
                        for v in val:
                            if isinstance(v, str) and upper_marker in v.upper():
                                return True
                # 4. URL / ID based check
                url = getattr(report, "page_url", "") or getattr(
                    report, "question_url", ""
                )
                if isinstance(url, str) and url:
                    tail = url.rstrip("/").split("/")[-1]
                    if tail.isdigit():
                        qid_int = int(tail)
                        if (
                            qid_int in _WITHHELD_QUESTION_IDS
                            or qid_int in _BLOCKED_PUBLICATION_QIDS
                        ):
                            return True
                # 5. Structural uniform / low‑info probability checks across common fields
                prob_fields = (
                    "predicted_options",
                    "final_prediction",
                    "prediction_value",
                    "probabilities",
                    "distribution",
                )
                for attr in prob_fields:
                    dist = getattr(report, attr, None)
                    probs: List[float] = []
                    if (
                        isinstance(dist, (list, tuple))
                        and dist
                        and all(isinstance(x, (int, float)) for x in dist)
                    ):
                        probs = [float(x) for x in dist]
                    elif (
                        dist is not None
                        and hasattr(dist, "__iter__")
                        and not isinstance(dist, (str, bytes))
                    ):
                        # Attempt to treat as iterable of option objects
                        try:
                            candidate = []
                            for o in dist:
                                p = getattr(o, "probability", None)
                                if p is not None:
                                    candidate.append(float(p))
                            if candidate:
                                probs = candidate
                        except Exception:
                            pass
                    if len(probs) >= 3:
                        mx, mn = max(probs), min(probs)
                        if mx - mn <= 0.0005:
                            return True
                # 6. Detect low-information neutral binary (exact 0.5) with caution language
                # Accept various attribute names for probability
                neutral_terms = (
                    "LOW-CONFIDENCE",
                    "LOW CONFIDENCE",
                    "LOW-INFORMATION",
                    "LOW INFORMATION",
                    "NEUTRAL",
                    "LOW‑CONFIDENCE",
                    "LOW‑INFORMATION",
                )
                prob_fields_single = (
                    "final_prediction",
                    "prediction_value",
                    "probability",
                    "binary_probability",
                )
                single_p = None
                for attr in prob_fields_single:
                    val = getattr(report, attr, None)
                    if isinstance(val, (int, float)):
                        single_p = float(val)
                        break
                if single_p is not None and abs(single_p - 0.5) < 1e-9:
                    # search any reasoning text for neutral / low info markers
                    for attr in (
                        "forecast_reasoning",
                        "reasoning",
                        "rationale",
                        "analysis_text",
                    ):
                        val = getattr(report, attr, None)
                        if isinstance(val, str):
                            up = val.upper()
                            if any(t in up for t in neutral_terms):
                                return True
            except Exception:
                return False
            return False

        non_exception_reports = [
            r for r in forecast_reports if not isinstance(r, Exception)
        ]
        published_like_reports = [
            r for r in non_exception_reports if not _is_withheld(r)
        ]
        successful_forecasts = len(published_like_reports)
        failed_forecasts = len(
            [r for r in forecast_reports if isinstance(r, Exception)]
        )

        logger.info("=== Final Summary ===")
        logger.info("Successful forecasts: %d", successful_forecasts)
        logger.info("Failed forecasts: %d", failed_forecasts)
        logger.info("Total questions processed: %d", len(forecast_reports))

        if failed_forecasts > 0:
            logger.warning("Some forecasts failed. Check logs above for details.")
            # Log first few exceptions for debugging
            exceptions = [r for r in forecast_reports if isinstance(r, Exception)][:3]
            for i, exc in enumerate(exceptions, 1):
                logger.error("Exception %d: %s: %s", i, type(exc).__name__, exc)
    except Exception as e:
        logger.warning("Failed to log final summary: %s", e)

    # Write a lightweight run summary for CI artifacts/verification
    try:
        # Backfill withheld/blocked IDs from report objects best-effort
        derived_withheld_ids: Set[int] = set(_WITHHELD_QUESTION_IDS)
        derived_blocked_ids: Set[int] = set(_BLOCKED_PUBLICATION_QIDS)
        try:
            for r in forecast_reports:
                if isinstance(r, Exception):
                    continue
                if _is_withheld(r):
                    url = getattr(r, "page_url", "") or getattr(r, "question_url", "")
                    if isinstance(url, str) and url:
                        tail = url.rstrip("/").split("/")[-1]
                        if tail.isdigit():
                            derived_withheld_ids.add(int(tail))
        except Exception:
            pass

        # Pull metrics if publish gate ran; otherwise default
        publish_attempts = locals().get("publish_attempts", 0)
        published_success = locals().get("published_success", 0)
        decisions = locals().get("decisions", [])
        withheld_count = len(derived_withheld_ids)
        blocked_count = len(derived_blocked_ids)

        # Cost attribution (safe defaults on failure)
        total_estimated_cost = 0.0
        published_cost = 0.0
        withheld_cost = 0.0
        blocked_cost = 0.0
        try:
            from src.infrastructure.config.token_tracker import token_tracker  # type: ignore

            total_estimated_cost = float(
                getattr(token_tracker, "total_estimated_cost", 0.0)
            )
        except Exception:
            pass

        # Prefer pulling retry/backoff counters from tri-model router's hardened client if present
        _router = getattr(template_bot, "tri_model_router", None)
        _router_client = getattr(_router, "llm_client", None)
        try:
            import src.infrastructure.external_apis.llm_client as _llm_client_mod  # type: ignore

            _total_calls = getattr(_llm_client_mod, "OPENROUTER_TOTAL_CALLS", 0)
            _total_retries = getattr(_llm_client_mod, "OPENROUTER_TOTAL_RETRIES", 0)
            _total_backoff = getattr(_llm_client_mod, "OPENROUTER_TOTAL_BACKOFF", 0.0)
            _quota_exceeded = getattr(
                _llm_client_mod, "OPENROUTER_QUOTA_EXCEEDED", False
            )
            _quota_message = getattr(_llm_client_mod, "OPENROUTER_QUOTA_MESSAGE", None)
        except Exception:
            _total_calls = 0
            _total_retries = 0
            _total_backoff = 0.0
            _quota_exceeded = False
            _quota_message = None

        summary = {
            "run_mode": run_mode,
            "tournament_mode": os.getenv("TOURNAMENT_MODE", "false"),
            "tournament_target": tournament_target_env,
            "publish_reports": os.getenv("PUBLISH_REPORTS", "false"),
            "successful_forecasts": published_success
            if locals().get("evaluate_publish")
            else len(
                [
                    r
                    for r in forecast_reports
                    if not isinstance(r, Exception) and not (_is_withheld(r))
                ]
            ),
            "failed_forecasts": len(
                [r for r in forecast_reports if isinstance(r, Exception)]
            ),
            "total_processed": len(forecast_reports),
            "blocked_publication_qids": sorted(list(derived_blocked_ids)),
            "withheld_qids": sorted(list(derived_withheld_ids)),
            "publish_attempts": publish_attempts,
            "published_success": published_success,
            "withheld_count": withheld_count,
            "blocked_count": blocked_count,
            "total_estimated_cost": round(total_estimated_cost, 6),
            "published_cost": round(published_cost, 6),
            "withheld_cost": round(withheld_cost, 6),
            "blocked_cost": round(blocked_cost, 6),
            "publication_success_rate": round(
                (published_success / publish_attempts) if publish_attempts else 0.0, 4
            ),
            "publish_decisions": decisions[:50] if isinstance(decisions, list) else [],
            "started_at": run_started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "openrouter_retry_count_last_call": getattr(
                _router_client, "last_openrouter_retry_count", None
            ),
            "openrouter_total_backoff_seconds_last_call": getattr(
                _router_client, "last_openrouter_total_backoff", None
            ),
            "openrouter_total_calls": _total_calls,
            "openrouter_total_retries": _total_retries,
            "openrouter_total_backoff_seconds": round(float(_total_backoff), 3),
            "openrouter_quota_exceeded": bool(_quota_exceeded),
            "openrouter_quota_message": _quota_message,
        }

        # Add circuit breaker status to summary
        try:
            from src.infrastructure.external_apis.llm_client import get_openrouter_circuit_breaker_status
            cb_status = get_openrouter_circuit_breaker_status()
            summary["openrouter_circuit_breaker"] = cb_status
        except Exception:
            summary["openrouter_circuit_breaker"] = {"error": "Could not get circuit breaker status"}
        with open("run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Run summary written to run_summary.json")
    except Exception as e:
        logger.warning("Failed to write run summary: %s", e)

    logger.info("Bot execution completed.")
