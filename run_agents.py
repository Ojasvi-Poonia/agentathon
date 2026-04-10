#!/usr/bin/env python3
"""
Agentathon 2026 -- Enterprise Finance AI Squad
Dual-mode pipeline:
  STEP 1: Deterministic Q1-Q5 analysis (always runs, <2s, no API key needed)
  STEP 2: Google ADK 6-agent pipeline (runs if GEMINI_API_KEY is set)

Agents: Data Engineer -> Planner -> Analyst -> Auditor -> Synthesizer -> Validator

Usage:
    python run_agents.py --data ./data                 # both pipelines
    python run_agents.py --data ./data --fallback      # deterministic only
    python run_agents.py --data ./data --problem problem.txt
    python run_agents.py --data ./data --map "category=ProductType,revenue=Sales"
"""
import argparse
import asyncio
import logging
import os
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents import SequentialAgent

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("Pipeline")


def load_problem_statement(problem_arg: str) -> str:
    """Load problem statement from file path or inline text."""
    if not problem_arg:
        return ""
    if os.path.isfile(problem_arg):
        with open(problem_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return problem_arg


async def _run_adk_pipeline(
    squad: "SequentialAgent", data_dir: str, problem: str
) -> None:
    """Run the ADK SequentialAgent via InMemoryRunner and stream events."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=squad, app_name="finance_squad")
    session = await runner.session_service.create_session(
        app_name="finance_squad",
        user_id="agentathon_user",
    )

    kickoff_text = (
        f"Analyze the dataset in '{data_dir}'. "
        f"Run the full Q1-Q5 pipeline end-to-end. "
        f"Problem statement: {problem or 'standard Q1-Q4 financial analysis'}."
    )
    content = types.Content(
        role="user",
        parts=[types.Part(text=kickoff_text)],
    )

    print()
    async for event in runner.run_async(
        user_id="agentathon_user",
        session_id=session.id,
        new_message=content,
    ):
        author = getattr(event, "author", "agent") or "agent"
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    preview = part.text.strip().replace("\n", " ")[:180]
                    if preview:
                        print(f"    [{author}] {preview}")
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    print(f"    [{author}] -> {fc.name}()")


def run_deterministic_pipeline(data_dir: str, output_path: str) -> bool:
    """
    Primary deterministic pipeline.  Runs the EXACT Q1-Q4 analyses
    that the scorer expects, in strict format.  Fast (<5 s for accuracy + speed pts).
    """
    import glob as gl
    from tools.data_ops import load_and_profile, clean_data
    from tools.analysis import (
        q1_revenue_by_category, q2_avg_delivery_by_region,
        q3_data_quality, q4_return_rate_by_payment,
    )
    from tools.reporting import compile_report, generate_chart
    import tools as state

    print("\n  [Pipeline] Running deterministic Q1-Q5 analysis...")

    # ── Discover & Load ──────────────────────────────────────
    all_files = []
    for ext in ("csv", "json", "xlsx", "xls", "parquet"):
        all_files += gl.glob(os.path.join(data_dir, f"**/*.{ext}"), recursive=True)

    if not all_files:
        print("  No data files found.")
        return False
    print(f"  Found {len(all_files)} file(s)")

    for f in all_files:
        load_and_profile(f)
    for f in all_files:
        clean_data(f)

    col_map = state.column_map
    df = state.get_active_df()
    if df is None:
        print("  No data loaded.")
        return False
    print(f"  Rows: {df.shape[0]}  Cols: {df.shape[1]}")
    print(f"  Detected: {col_map}")

    # ── Compute revenue if we have quantity + price + discount ──
    # Formula: revenue = quantity * unit_price * (1 - discount_percent/100)
    #
    # We derive the three numeric series from the ORIGINAL raw data (not the
    # cleaned df) so that:
    #   1. quantity/unit_price/discount are all imputed CONSISTENTLY with
    #      their own median — not mixed mode-fill (object) and median-fill
    #      (numeric).
    #   2. Format errors (non-numeric unit_price strings) get imputed too,
    #      instead of being silently dropped from the sum.
    # This matches the most common "clean then compute" interpretation of
    # the problem statement and uses all rows in the dataset.
    import pandas as pd
    qty_col = col_map.get("quantity")
    prc_col = col_map.get("price")
    dsc_col = col_map.get("discount")
    if qty_col and prc_col and dsc_col:
        df_orig = state.store.get("original", state.store["cleaned"])
        qty_num = pd.to_numeric(df_orig[qty_col], errors="coerce")
        prc_num = pd.to_numeric(df_orig[prc_col], errors="coerce")
        dsc_num = pd.to_numeric(df_orig[dsc_col], errors="coerce")
        qty_num = qty_num.fillna(qty_num.median())
        prc_num = prc_num.fillna(prc_num.median())
        dsc_num = dsc_num.fillna(dsc_num.median())
        revenue = qty_num * prc_num * (1 - dsc_num / 100)

        df_clean = state.store["cleaned"].copy()
        df_clean["_revenue"] = revenue.values
        state.store["cleaned"] = df_clean
        col_map["revenue"] = "_revenue"
        print(f"  Computed revenue = {qty_col} * {prc_col} * (1 - {dsc_col}/100)")

    # ── Q1: Revenue by Category ──────────────────────────────
    cat_col = col_map.get("category")
    rev_col = col_map.get("revenue")
    if cat_col and rev_col:
        q1_revenue_by_category(cat_col, rev_col)
        print(f"  Q1 done  ({cat_col} x {rev_col})")
    else:
        print(f"  Q1 SKIP  category={cat_col} revenue={rev_col}")

    # ── Q2: Delivery by Region ───────────────────────────────
    reg_col = col_map.get("region")
    del_col = col_map.get("delivery_days")
    if reg_col and del_col:
        q2_avg_delivery_by_region(reg_col, del_col)
        print(f"  Q2 done  ({reg_col} x {del_col})")
    else:
        print(f"  Q2 SKIP  region={reg_col} delivery={del_col}")

    # ── Q3: Data Quality ─────────────────────────────────────
    oid = col_map.get("entity_id")
    qty = col_map.get("quantity")
    prc = col_map.get("price")
    dsc = col_map.get("discount")
    if all([oid, qty, prc, dsc]):
        q3_data_quality(oid, qty, prc, dsc)
        print(f"  Q3 done  ({oid}, {qty}, {prc}, {dsc})")
    else:
        print(f"  Q3 SKIP  id={oid} qty={qty} price={prc} disc={dsc}")

    # ── Q4: Return Rate by Payment ───────────────────────────
    pay_col = col_map.get("payment")
    ret_col = col_map.get("returned")
    if pay_col and ret_col:
        q4_return_rate_by_payment(pay_col, ret_col)
        print(f"  Q4 done  ({pay_col} x {ret_col})")
    else:
        print(f"  Q4 SKIP  payment={pay_col} returned={ret_col}")

    # ── Charts ───────────────────────────────────────────────
    if cat_col and rev_col:
        generate_chart("barh", cat_col, rev_col, "Q1 Revenue by Category")
    if reg_col and del_col:
        generate_chart("bar", reg_col, del_col, "Q2 Delivery by Region")
    if pay_col and ret_col:
        generate_chart("bar", pay_col, ret_col, "Q4 Return Rate by Payment")

    # ── Compile strict Q1-Q5 submission ──────────────────────
    compile_report(output_path=output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Finance AI Squad -- Agentathon 2026"
    )
    parser.add_argument("--data", default="./data", help="Dataset directory")
    parser.add_argument(
        "--team", default="team-name",
        help="Team name -- used as output file <team>.txt"
    )
    parser.add_argument(
        "--output", default="",
        help="Explicit output path (overrides --team)"
    )
    parser.add_argument(
        "--problem", default="",
        help="Problem statement: file path or inline text"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Skip Gemini agents and run deterministic fallback only"
    )
    parser.add_argument(
        "--map", default="",
        help="Manual column overrides: role=col,role=col  e.g. "
             "'category=product_category,revenue=_revenue'"
    )
    args = parser.parse_args()

    # Resolve output path
    if not args.output:
        safe_team = args.team.strip().replace(" ", "-").replace("/", "-")
        args.output = f"output/{safe_team}.txt"

    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Apply manual column overrides if provided
    if args.map:
        import tools as _st
        for pair in args.map.split(","):
            if "=" in pair:
                role, col = pair.strip().split("=", 1)
                _st.column_map[role.strip()] = col.strip()
        print(f"  Manual overrides: {_st.column_map}")

    print("\n" + "=" * 62)
    print("  ENTERPRISE FINANCE AI SQUAD -- Agentathon 2026")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    problem = load_problem_statement(args.problem)
    if problem:
        print(f"\n  Problem: {problem[:200]}...")
    else:
        print("\n  No problem statement provided. Agents will infer from data.")

    start = time.time()

    # ── STEP 1: Always run deterministic pipeline first (fast, accurate) ──
    print("\n  STEP 1: Deterministic Q1-Q5 pipeline")
    success = run_deterministic_pipeline(args.data, args.output)

    # ── STEP 2: Run Google ADK agent pipeline (if not --fallback) ──
    if not args.fallback:
        try:
            from dotenv import load_dotenv
            load_dotenv()

            api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv(
                "GOOGLE_API_KEY", ""
            )
            if api_key and api_key != "your_key_here":
                # ADK reads GOOGLE_API_KEY from env
                os.environ["GOOGLE_API_KEY"] = api_key

                print("\n  STEP 2: Google ADK agent pipeline")
                print("  6-agent squad: Data Engineer, Planner, Analyst, "
                      "Auditor, Synthesizer, Validator")

                import asyncio
                from orchestrator import build_squad
                from tools.reporting import compile_report

                squad = build_squad(args.data, problem)
                asyncio.run(_run_adk_pipeline(squad, args.data, problem))

                # Re-compile using any improved findings from the agents
                compile_report(output_path=args.output)
            else:
                print("\n  STEP 2: Skipped (no GEMINI_API_KEY / GOOGLE_API_KEY)")
        except Exception as e:
            log.error(f"ADK agent pipeline failed: {e}")
            traceback.print_exc()
            print(f"\n  Agent demo failed: {e}")
            print("  Deterministic submission remains valid.")

    elapsed = time.time() - start

    print(f"\n{'=' * 62}")
    print(f"  EXECUTION COMPLETE -- {elapsed:.1f}s")
    print(f"  Submission: {args.output}")
    print(f"  Charts: output/")
    print(f"  Log: logs/pipeline.log")
    print(f"{'=' * 62}\n")

    if os.path.exists(args.output):
        print("-- SUBMISSION --")
        with open(args.output, encoding="utf-8") as f:
            content = f.read()
            print(content)
            print(f"\n({len(content)} characters)")
    else:
        print("  WARNING: No submission file generated.")


if __name__ == "__main__":
    main()
