"""
Shared state and tool registry for the Finance AI Squad.
All agent tools read/write through this module's state.
"""
from typing import Optional
import pandas as pd

# ── Shared State ─────────────────────────────────────────────
store = {}       # keyed dataframes
results = {}                         # analysis results
column_map = {}            # {role: column_name}
findings = {}             # {key: {title, content}}
metadata = {}                        # dataset metadata


def get_active_df() -> Optional[pd.DataFrame]:
    return store.get("cleaned", store.get("main"))


def save_result(key: str, value):
    results[key] = value


def get_result(key: str):
    return results.get(key)


def reset():
    store.clear()
    results.clear()
    column_map.clear()
    findings.clear()
    metadata.clear()


# ── Tool Registry (populated after imports) ──────────────────
TOOL_REGISTRY = {}


def _register():
    from tools.data_ops import discover_files, load_and_profile, clean_data
    from tools.analysis import (
        group_aggregate, compute_ratio, correlation_analysis,
        outlier_detection, top_bottom_n, distribution_summary,
        cross_tabulation, data_quality_audit, save_finding,
        q1_revenue_by_category, q2_avg_delivery_by_region,
        q3_data_quality, q4_return_rate_by_payment,
    )
    from tools.reporting import generate_chart, compile_report

    TOOL_REGISTRY.update({
        "discover_files": discover_files,
        "load_and_profile": load_and_profile,
        "clean_data": clean_data,
        "group_aggregate": group_aggregate,
        "compute_ratio": compute_ratio,
        "correlation_analysis": correlation_analysis,
        "outlier_detection": outlier_detection,
        "top_bottom_n": top_bottom_n,
        "distribution_summary": distribution_summary,
        "cross_tabulation": cross_tabulation,
        "data_quality_audit": data_quality_audit,
        "save_finding": save_finding,
        "generate_chart": generate_chart,
        "compile_report": compile_report,
        "q1_revenue_by_category": q1_revenue_by_category,
        "q2_avg_delivery_by_region": q2_avg_delivery_by_region,
        "q3_data_quality": q3_data_quality,
        "q4_return_rate_by_payment": q4_return_rate_by_payment,
    })


_register()
