"""评测子包。"""

from .dataset import Case, load_cases
from .failure import CATEGORY_LABELS, classify
from .runner import Report, format_summary, run_eval, run_self_check, save_report

__all__ = [
    "Case",
    "load_cases",
    "CATEGORY_LABELS",
    "classify",
    "Report",
    "format_summary",
    "run_eval",
    "run_self_check",
    "save_report",
]
