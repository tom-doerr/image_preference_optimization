"""Thin facade re‑exporting ui_sidebar helpers.

Tests may import from ipo.ui.ui; all implementations live in ipo.ui.ui_sidebar.
"""

from .ui_sidebar import (
    sidebar_metric,
    sidebar_metric_rows,
    compute_step_scores,
    render_iter_step_scores,
    render_mu_value_history,
    render_pair_sidebar,
    env_panel,
    status_panel,
    perf_panel,
)

__all__ = [
    "sidebar_metric",
    "sidebar_metric_rows",
    "compute_step_scores",
    "render_iter_step_scores",
    "render_mu_value_history",
    "render_pair_sidebar",
    "env_panel",
    "status_panel",
    "perf_panel",
]
