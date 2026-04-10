"""
Reporting: chart generation and submission compilation.
"""
import json
import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tools as state

log = logging.getLogger("Reporting")

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (11, 6),
    "font.size": 11,
    "axes.titlesize": 14,
})

OUTPUT_DIR = "output"


def generate_chart(chart_type: str, x_col: str, y_col: str, title: str = "") -> str:
    """
    Generate a chart from the active dataset.
    chart_type: bar | barh | line | scatter | pie | heatmap
    x_col: column for x-axis (or group column)
    y_col: column for y-axis (or value column)
    title: chart title
    """
    log.info(f"Chart: {chart_type} {x_col} x {y_col}")
    try:
        df = state.get_active_df()
        if df is None:
            return json.dumps({"error": "No data loaded"})

        missing = [c for c in (x_col, y_col) if c and c not in df.columns]
        if missing:
            return json.dumps({"error": f"Columns not found: {missing}. Available: {list(df.columns)}"})

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_title = (title or f"{y_col}_by_{x_col}").replace(" ", "_")[:50]
        path = f"{OUTPUT_DIR}/{safe_title}.png"

        fig, ax = plt.subplots()
        ct = chart_type.lower().strip()

        y_numeric = pd.to_numeric(df[y_col], errors="coerce")

        if ct in ("bar", "barh"):
            agg = df.assign(_v=y_numeric).dropna(subset=["_v"]).groupby(x_col)["_v"].sum().sort_values(ascending=False)
            if len(agg) > 20:
                agg = agg.head(20)
            colors = sns.color_palette("viridis", len(agg))
            if ct == "barh":
                cats = list(agg.index)[::-1]
                vals = [float(agg[c]) for c in cats]
                ax.barh(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
                ax.set_xlabel(y_col)
            else:
                cats = list(agg.index)
                vals = [float(agg[c]) for c in cats]
                ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
                ax.set_ylabel(y_col)
                plt.xticks(rotation=30, ha="right")

        elif ct == "line":
            grouped = df.assign(_v=y_numeric).dropna(subset=["_v"]).groupby(x_col)["_v"].mean()
            ax.plot(grouped.index, grouped.values, marker="o", linewidth=2)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=30, ha="right")

        elif ct == "scatter":
            x_numeric = pd.to_numeric(df[x_col], errors="coerce")
            valid = pd.DataFrame({"x": x_numeric, "y": y_numeric}).dropna()
            ax.scatter(valid["x"], valid["y"], alpha=0.5, s=20)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif ct == "pie":
            agg = df.assign(_v=y_numeric).dropna(subset=["_v"]).groupby(x_col)["_v"].sum()
            if len(agg) > 10:
                top = agg.nlargest(9)
                top["Other"] = agg.sum() - top.sum()
                agg = top
            ax.pie(agg.values, labels=agg.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")

        else:
            return json.dumps({"error": f"Unknown chart_type: {chart_type}. Use: bar, barh, line, scatter, pie"})

        if title:
            ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        log.info(f"  Chart saved: {path}")
        return json.dumps({"status": "saved", "path": path, "chart_type": ct})
    except Exception as e:
        log.error(f"Chart failed: {e}")
        return json.dumps({"error": str(e)})


def compile_report(output_format: str = "text",
                   output_path: str = "") -> str:
    """
    Compile results into STRICT single-line Q1-Q5 submission format
    matching the official output_format.txt template:

        Q1: [ranked labeled values]
        Q2: [ranked labeled values]
        Q3: [five required counts]
        Q4: [ranked labeled values]
        Q5: [exactly 3 sentences]

    Each Q section is a single line.  Labeled values within a section
    are comma-separated.  No markdown, no currency symbols.
    """
    log.info("Compiling strict Q1-Q5 report")
    try:
        q1 = state.get_result("q1")
        q2 = state.get_result("q2")
        q3 = state.get_result("q3")
        q4 = state.get_result("q4")

        lines = []

        # Q1: comma-separated ranked labels
        if q1 and isinstance(q1, dict):
            q1_parts = [f"{k}: {v}" for k, v in q1.items()]
            lines.append(f"Q1: {', '.join(q1_parts)}")
        else:
            lines.append("Q1: No data")

        # Q2: comma-separated ranked labels with "days" suffix
        if q2 and isinstance(q2, dict):
            q2_parts = [f"{k}: {v} days" for k, v in q2.items()]
            lines.append(f"Q2: {', '.join(q2_parts)}")
        else:
            lines.append("Q2: No data")

        # Q3: five required counts
        if q3 and isinstance(q3, dict):
            q3_parts = [
                f"Duplicate order IDs: {q3.get('duplicate_order_ids', 0)}",
                f"Quantity outliers: {q3.get('quantity_outliers', 0)}",
                f"Price format errors: {q3.get('price_format_errors', 0)}",
                f"Invalid discounts: {q3.get('invalid_discounts', 0)}",
                f"Total null cells: {q3.get('total_null_cells', 0)}",
            ]
            lines.append(f"Q3: {', '.join(q3_parts)}")
        else:
            lines.append("Q3: No data")

        # Q4: comma-separated ranked labels with "%" suffix
        if q4 and isinstance(q4, dict):
            q4_parts = [f"{k}: {v}%" for k, v in q4.items()]
            lines.append(f"Q4: {', '.join(q4_parts)}")
        else:
            lines.append("Q4: No data")

        # Q5: exactly 3 sentences
        q5 = state.get_result("q5") or _generate_q5()
        lines.append(f"Q5: {q5}")

        report = "\n".join(lines)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = output_path or f"{OUTPUT_DIR}/submission.txt"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

        log.info(f"  Report saved: {out_path} ({len(report)} chars)")
        return json.dumps({
            "status": "saved",
            "path": out_path,
            "length": len(report),
            "preview": report[:500],
        }, indent=2)
    except Exception as e:
        log.error(f"Report failed: {e}")
        return json.dumps({"error": str(e)})


def _generate_q5() -> str:
    """Auto-generate Q5 executive summary from Q1-Q4 results."""
    q1 = state.get_result("q1")
    q2 = state.get_result("q2")
    q4 = state.get_result("q4")
    overall_ret = state.get_result("overall_return_rate")

    parts = []
    if q1:
        top_cat = list(q1.keys())[0]
        top_val = list(q1.values())[0]
        parts.append(
            f"{top_cat} generated the strongest revenue at {top_val}, "
            f"so it is the top-performing category."
        )
    if q2:
        slow_reg = list(q2.keys())[0]
        slow_days = list(q2.values())[0]
        parts.append(
            f"{slow_reg} had the slowest delivery performance at {slow_days} days, "
            f"which is the weakest regional signal."
        )
    if overall_ret is not None:
        parts.append(
            f"The overall return rate was {overall_ret}%, so the business is performing "
            f"reasonably well but still has room to reduce returns."
        )
    elif q4:
        avg_rate = round(sum(q4.values()) / len(q4), 2)
        parts.append(
            f"The average return rate was {avg_rate}% across payment methods."
        )

    summary = " ".join(parts) if parts else "Analysis complete."
    state.save_result("q5", summary)
    return summary
