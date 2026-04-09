"""
Generic analysis functions usable on any dataset.
The LLM (Planner/Analyst agents) decides which columns to feed into these.
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import tools as state

log = logging.getLogger("Analysis")


def _cols_exist(df, *cols):
    """Return error JSON if any column is missing, else None."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return json.dumps({
            "error": f"Columns not found: {missing}. Available: {list(df.columns)}"
        })
    return None


# ═══════════════════════════════════════════════════════════════
#  GENERIC ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════

def group_aggregate(group_col: str, value_col: str, agg_func: str = "sum") -> str:
    """
    Group by group_col, aggregate value_col.
    agg_func: sum | mean | median | count | min | max | std
    Returns ranked results with share percentages for sum.
    """
    log.info(f"GroupAgg: {group_col} x {value_col} ({agg_func})")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, group_col, value_col)
        if err:
            return err

        values = pd.to_numeric(df[value_col], errors="coerce")
        valid = df.assign(_val=values).dropna(subset=["_val"])

        func_map = {
            "sum": "sum", "mean": "mean", "median": "median",
            "count": "count", "min": "min", "max": "max", "std": "std",
        }
        func = func_map.get(agg_func.lower().strip(), "sum")

        agg = valid.groupby(group_col)["_val"].agg(func).sort_values(ascending=False)
        total = float(values.sum()) if func == "sum" and values.sum() != 0 else None

        details = []
        for key, val in agg.items():
            entry = {"group": str(key), "value": round(float(val), 2)}
            if total:
                entry["share_pct"] = round(float(val) / total * 100, 2)
            details.append(entry)

        result_key = f"{agg_func}_{value_col}_by_{group_col}"
        state.save_result(result_key, {str(k): round(float(v), 2) for k, v in agg.items()})

        return json.dumps({
            "group_col": group_col,
            "value_col": value_col,
            "agg_func": func,
            "row_count": len(details),
            "details": details,
        }, indent=2, default=str)
    except Exception as e:
        log.error(f"GroupAgg failed: {e}")
        return json.dumps({"error": str(e)})


def compute_ratio(numerator_col: str, denominator_col: str, ratio_name: str) -> str:
    """
    Compute a new ratio column = numerator / denominator.
    Useful for financial ratios, margins, rates, etc.
    """
    log.info(f"Ratio: {numerator_col} / {denominator_col} = {ratio_name}")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, numerator_col, denominator_col)
        if err:
            return err

        num = pd.to_numeric(df[numerator_col], errors="coerce")
        den = pd.to_numeric(df[denominator_col], errors="coerce")
        ratio = (num / den.replace(0, np.nan)).round(4)

        state.store["cleaned"][ratio_name] = ratio
        clean = ratio.dropna()

        result = {
            "ratio_name": ratio_name,
            "count": int(clean.count()),
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4),
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
        }
        state.save_result(f"ratio_{ratio_name}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Ratio failed: {e}")
        return json.dumps({"error": str(e)})


def correlation_analysis(col1: str, col2: str) -> str:
    """Compute Pearson and Spearman correlation between two numeric columns."""
    log.info(f"Correlation: {col1} x {col2}")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, col1, col2)
        if err:
            return err

        s1 = pd.to_numeric(df[col1], errors="coerce")
        s2 = pd.to_numeric(df[col2], errors="coerce")
        valid = pd.DataFrame({"a": s1, "b": s2}).dropna()

        if len(valid) < 3:
            return json.dumps({"error": "Too few valid rows for correlation"})

        pearson_r, pearson_p = sp_stats.pearsonr(valid["a"], valid["b"])
        spearman_r, spearman_p = sp_stats.spearmanr(valid["a"], valid["b"])

        strength = "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak"
        direction = "positive" if pearson_r > 0 else "negative"

        result = {
            "col1": col1, "col2": col2,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 6),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 6),
            "interpretation": f"{strength} {direction} correlation",
            "n": len(valid),
        }
        state.save_result(f"corr_{col1}_{col2}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Correlation failed: {e}")
        return json.dumps({"error": str(e)})


def outlier_detection(column: str) -> str:
    """Detect outliers using IQR method. Returns count and bounds."""
    log.info(f"Outliers: {column}")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, column)
        if err:
            return err

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) == 0:
            return json.dumps({"error": f"No numeric values in {column}"})

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]

        result = {
            "column": column,
            "total_values": len(values),
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(values) * 100, 2),
            "bounds": {"lower": round(lower, 2), "upper": round(upper, 2)},
            "iqr": round(iqr, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
        }
        state.save_result(f"outliers_{column}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Outlier detection failed: {e}")
        return json.dumps({"error": str(e)})


def top_bottom_n(rank_col: str, n: str = "10", ascending: str = "false") -> str:
    """
    Return top or bottom N rows ranked by rank_col.
    ascending='true' for bottom N, 'false' for top N.
    """
    log.info(f"Top/Bottom: {rank_col} n={n} asc={ascending}")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, rank_col)
        if err:
            return err

        n_int = int(n)
        asc = ascending.lower().strip() in ("true", "1", "yes")
        values = pd.to_numeric(df[rank_col], errors="coerce")
        df_sorted = df.assign(_rank_val=values).dropna(subset=["_rank_val"])
        df_sorted = df_sorted.sort_values("_rank_val", ascending=asc).head(n_int)

        records = json.loads(df_sorted.drop(columns=["_rank_val"]).to_json(orient="records", default_handler=str))
        label = "bottom" if asc else "top"

        result = {
            "column": rank_col,
            "direction": label,
            "n": n_int,
            "records": records,
        }
        state.save_result(f"{label}_{n_int}_{rank_col}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Top/Bottom failed: {e}")
        return json.dumps({"error": str(e)})


def distribution_summary(column: str) -> str:
    """Full statistical distribution for a column: mean, median, skew, kurtosis, percentiles."""
    log.info(f"Distribution: {column}")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, column)
        if err:
            return err

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) == 0:
            return json.dumps({"error": f"No numeric values in {column}"})

        result = {
            "column": column,
            "count": len(values),
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "skewness": round(float(values.skew()), 4),
            "kurtosis": round(float(values.kurtosis()), 4),
            "p5": round(float(values.quantile(0.05)), 4),
            "p25": round(float(values.quantile(0.25)), 4),
            "p75": round(float(values.quantile(0.75)), 4),
            "p95": round(float(values.quantile(0.95)), 4),
        }
        state.save_result(f"dist_{column}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Distribution failed: {e}")
        return json.dumps({"error": str(e)})


def cross_tabulation(row_col: str, col_col: str, value_col: str = "", agg_func: str = "count") -> str:
    """
    Pivot / cross-tabulation analysis.
    If value_col is empty, computes frequency counts.
    """
    log.info(f"CrossTab: {row_col} x {col_col} ({agg_func})")
    try:
        df = state.get_active_df()
        cols_to_check = [row_col, col_col]
        if value_col:
            cols_to_check.append(value_col)
        err = _cols_exist(df, *cols_to_check)
        if err:
            return err

        if value_col:
            pivot = pd.pivot_table(
                df, index=row_col, columns=col_col,
                values=value_col, aggfunc=agg_func, fill_value=0,
            )
        else:
            pivot = pd.crosstab(df[row_col], df[col_col])

        records = json.loads(pivot.reset_index().to_json(orient="records", default_handler=str))
        result = {
            "row_col": row_col,
            "col_col": col_col,
            "shape": list(pivot.shape),
            "records": records[:50],
        }
        state.save_result(f"xtab_{row_col}_{col_col}", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"CrossTab failed: {e}")
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════

def data_quality_audit(columns: str = "") -> str:
    """
    Comprehensive data quality audit on the ORIGINAL (uncleaned) data.
    Pass comma-separated column names or empty for all columns.
    Checks: duplicates, nulls, outliers, type consistency.
    """
    log.info("Data quality audit")
    try:
        df = state.store.get("original", state.store.get("main"))
        if df is None:
            return json.dumps({"error": "No data loaded"})

        if columns.strip():
            audit_cols = [c.strip() for c in columns.split(",")]
        else:
            audit_cols = list(df.columns)

        total_cells = df.shape[0] * df.shape[1]
        total_nulls = int(df.isnull().sum().sum())
        total_dupes = int(df.duplicated().sum())

        col_issues = {}
        for col in audit_cols:
            if col not in df.columns:
                continue
            issues = {"nulls": int(df[col].isna().sum())}
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid_numeric = numeric.dropna()

            if len(valid_numeric) > 0.5 * len(df[col].dropna()):
                q1 = valid_numeric.quantile(0.25)
                q3 = valid_numeric.quantile(0.75)
                iqr = q3 - q1
                outlier_mask = (valid_numeric < q1 - 1.5 * iqr) | (valid_numeric > q3 + 1.5 * iqr)
                issues["outliers"] = int(outlier_mask.sum())
                issues["negatives"] = int((valid_numeric < 0).sum())
            else:
                issues["unique_values"] = int(df[col].nunique())

            col_issues[col] = issues

        total_issues = total_nulls + total_dupes
        for ci in col_issues.values():
            total_issues += ci.get("outliers", 0) + ci.get("negatives", 0)

        quality_score = max(0, round(100 * (1 - total_issues / max(total_cells, 1)), 1))

        result = {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "total_cells": total_cells,
            "total_null_cells": total_nulls,
            "total_duplicate_rows": total_dupes,
            "column_issues": col_issues,
            "quality_score": quality_score,
        }
        state.save_result("data_quality", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Audit failed: {e}")
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#  FINDING MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def save_finding(key: str, title: str, content: str) -> str:
    """
    Save an analysis finding for the final report.
    key: short identifier like 'Q1', 'Q2', 'revenue_insight'
    title: human-readable title
    content: the finding text
    """
    state.findings[key] = {
        "title": title,
        "content": content,
        "saved_at": datetime.now().isoformat(),
    }
    log.info(f"Finding saved: {key} - {title}")
    return json.dumps({"status": "saved", "key": key, "title": title})


# ═══════════════════════════════════════════════════════════════
#  COMPETITION-SPECIFIC Q1-Q4 (strict scorer format)
# ═══════════════════════════════════════════════════════════════

def q1_revenue_by_category(category_col: str, revenue_col: str) -> str:
    """Q1: Total revenue per category, ranked highest to lowest."""
    log.info(f"Q1: revenue [{category_col}] x [{revenue_col}]")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, category_col, revenue_col)
        if err:
            return err
        revenue = pd.to_numeric(df[revenue_col], errors="coerce")
        valid = df.assign(_rev=revenue).dropna(subset=["_rev"])
        agg = valid.groupby(category_col)["_rev"].sum().sort_values(ascending=False)
        result = {str(k): round(float(v), 2) for k, v in agg.items()}
        state.save_result("q1", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Q1 failed: {e}")
        return json.dumps({"error": str(e)})


def q2_avg_delivery_by_region(region_col: str, delivery_col: str) -> str:
    """Q2: Average delivery days per region, ranked slowest to fastest."""
    log.info(f"Q2: delivery [{region_col}] x [{delivery_col}]")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, region_col, delivery_col)
        if err:
            return err
        days = pd.to_numeric(df[delivery_col], errors="coerce")
        valid = df.assign(_days=days).dropna(subset=["_days"])
        agg = valid.groupby(region_col)["_days"].mean().sort_values(ascending=False)
        result = {str(k): round(float(v), 2) for k, v in agg.items()}
        state.save_result("q2", result)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Q2 failed: {e}")
        return json.dumps({"error": str(e)})


def q3_data_quality(order_id_col: str, quantity_col: str,
                    price_col: str, discount_col: str) -> str:
    """
    Q3: Exact 5 data-quality counts on the ORIGINAL (uncleaned) data.
      1. Duplicate order IDs
      2. Quantity outliers (IQR 1.5x)
      3. Price format errors (null + non-numeric + negative)
      4. Invalid discounts (null + non-numeric + outside [0,100])
      5. Total null cells (entire dataset)
    """
    log.info("Q3: data quality audit")
    try:
        df = state.store.get("original", state.store.get("main"))
        if df is None:
            return json.dumps({"error": "No data loaded"})

        # 1. Duplicate order IDs
        dup_ids = int(df[order_id_col].duplicated().sum())

        # 2. Quantity outliers (IQR)
        qty = pd.to_numeric(df[quantity_col], errors="coerce")
        q1_val = qty.quantile(0.25)
        q3_val = qty.quantile(0.75)
        iqr = q3_val - q1_val
        qty_outliers = int(((qty < q1_val - 1.5 * iqr) | (qty > q3_val + 1.5 * iqr)).sum())

        # 3. Price format errors (null + non-numeric + negative)
        price_num = pd.to_numeric(df[price_col], errors="coerce")
        price_errors = int(price_num.isna().sum()) + int((price_num < 0).sum())

        # 4. Invalid discounts (null + non-numeric + outside [0, 100])
        disc = pd.to_numeric(df[discount_col], errors="coerce")
        disc_invalid = int(disc.isna().sum()) + int(((disc < 0) | (disc > 100)).sum())

        # 5. Total null cells
        total_nulls = int(df.isnull().sum().sum())

        result = {
            "duplicate_order_ids": dup_ids,
            "quantity_outliers": qty_outliers,
            "price_format_errors": price_errors,
            "invalid_discounts": disc_invalid,
            "total_null_cells": total_nulls,
        }
        state.save_result("q3", result)
        log.info(f"  Q3 counts: {result}")
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Q3 failed: {e}")
        return json.dumps({"error": str(e)})


def q4_return_rate_by_payment(payment_col: str, return_col: str) -> str:
    """Q4: Return-rate percentage per payment method, ranked highest to lowest."""
    log.info(f"Q4: returns [{payment_col}] x [{return_col}]")
    try:
        df = state.get_active_df()
        err = _cols_exist(df, payment_col, return_col)
        if err:
            return err

        # flexible return-flag parsing
        ret = pd.to_numeric(df[return_col], errors="coerce")
        if ret.isna().sum() > len(ret) * 0.5:
            ret = df[return_col].map({
                True: 1, False: 0, "Yes": 1, "No": 0,
                "yes": 1, "no": 0, "TRUE": 1, "FALSE": 0,
                "1": 1, "0": 0, 1: 1, 0: 0,
            })
        working = df.assign(_ret=ret).dropna(subset=["_ret"])
        rates = (working.groupby(payment_col)["_ret"].mean() * 100).sort_values(ascending=False)
        result = {str(k): round(float(v), 2) for k, v in rates.items()}

        overall = round(float(working["_ret"].mean() * 100), 2)
        state.save_result("q4", result)
        state.save_result("overall_return_rate", overall)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        log.error(f"Q4 failed: {e}")
        return json.dumps({"error": str(e)})
