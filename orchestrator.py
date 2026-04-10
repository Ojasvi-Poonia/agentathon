"""
6-Agent Orchestrator built on Google ADK (Agent Development Kit).

Pipeline: Data Engineer -> Planner -> Analyst -> Auditor -> Synthesizer -> Validator

Each agent is an ADK `LlmAgent` with:
  - A dedicated system instruction
  - A set of Python function tools (auto-introspected by ADK)
  - An `output_key` that stores its text output into session state
    so the next agent can reference it via {key} in its instruction.

The 6 agents are wrapped in an ADK `SequentialAgent` that runs them in order.
Execution happens through `InMemoryRunner.run_async()`.
"""
from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents import LlmAgent, SequentialAgent

from config import GEMINI_MODEL
from tools.analysis import (
    compute_ratio,
    correlation_analysis,
    cross_tabulation,
    data_quality_audit,
    distribution_summary,
    group_aggregate,
    outlier_detection,
    q1_revenue_by_category,
    q2_avg_delivery_by_region,
    q3_data_quality,
    q4_return_rate_by_payment,
    save_finding,
    top_bottom_n,
)
from tools.data_ops import clean_data, discover_files, load_and_profile
from tools.reporting import compile_report, generate_chart

log = logging.getLogger("Orchestrator")


# ═══════════════════════════════════════════════════════════════
#  AGENT FACTORIES
# ═══════════════════════════════════════════════════════════════


def _build_data_engineer(data_dir: str) -> LlmAgent:
    return LlmAgent(
        name="data_engineer",
        model=GEMINI_MODEL,
        description="Discovers, loads, profiles, and cleans data files.",
        instruction=(
            "You are a data engineering agent. Your job is to prepare the "
            "dataset for downstream analysis.\n\n"
            "Steps you MUST follow in order:\n"
            f"1. Call discover_files with directory='{data_dir}' to list files.\n"
            "2. For each CSV/JSON/XLSX found, call load_and_profile(file_path).\n"
            "3. For each file, call clean_data(file_path).\n\n"
            "After all tool calls, output a concise report containing:\n"
            "  - List of files loaded\n"
            "  - Exact column names from the data\n"
            "  - Detected roles from the profile output (entity_id, category, "
            "region, payment, quantity, price, discount, delivery_days, "
            "returned, revenue)\n"
            "  - Row count and null count\n\n"
            "Downstream agents depend on this report — be thorough and precise."
        ),
        tools=[discover_files, load_and_profile, clean_data],
        output_key="data_profile",
    )


def _build_planner(problem_statement: str) -> LlmAgent:
    problem_block = (
        f"\n\nPROBLEM STATEMENT FROM ORGANIZERS:\n{problem_statement}\n"
        if problem_statement
        else "\n\nNo explicit problem statement — run standard Q1-Q5 analysis.\n"
    )
    return LlmAgent(
        name="planner",
        model=GEMINI_MODEL,
        description=(
            "Creates an analysis plan from the data profile and "
            "problem statement."
        ),
        instruction=(
            "You are a strategic planner. You do NOT call tools — you reason "
            "and produce a plan.\n\n"
            "Read the data profile from the Data Engineer:\n"
            "{data_profile}\n"
            + problem_block
            + "\n"
            "Produce a numbered analysis plan covering at least:\n"
            "  Q1 — Revenue by category: which category column, which "
            "revenue column\n"
            "  Q2 — Average delivery by region: which region column, which "
            "delivery_days column\n"
            "  Q3 — Data quality audit: order_id, quantity, price, discount "
            "columns\n"
            "  Q4 — Return rate by payment method: payment column, returned "
            "column\n"
            "  Q5 — Executive summary synthesizing Q1-Q4\n\n"
            "For each Q, specify the EXACT column names to use (from the "
            "data profile). If a required column is missing, note it and "
            "suggest the closest alternative."
        ),
        output_key="analysis_plan",
    )


def _build_analyst() -> LlmAgent:
    return LlmAgent(
        name="analyst",
        model=GEMINI_MODEL,
        description="Executes the analysis plan using analysis tools.",
        instruction=(
            "You are a financial analyst. Execute the analysis plan produced "
            "by the planner.\n\n"
            "The plan:\n{analysis_plan}\n\n"
            "For Q1, Q2, and Q4, call the competition-optimized tools:\n"
            "  - q1_revenue_by_category(category_col, revenue_col)\n"
            "  - q2_avg_delivery_by_region(region_col, delivery_col)\n"
            "  - q4_return_rate_by_payment(payment_col, return_col)\n\n"
            "For any additional analyses the planner requested, use the "
            "generic tools (group_aggregate, correlation_analysis, "
            "outlier_detection, etc.).\n\n"
            "Use the EXACT column names from the data profile. "
            "Do NOT skip any question. If a tool call fails, try with a "
            "different column and continue. After each analysis, call "
            "save_finding(key, title, content) with the result."
        ),
        tools=[
            q1_revenue_by_category,
            q2_avg_delivery_by_region,
            q4_return_rate_by_payment,
            group_aggregate,
            compute_ratio,
            correlation_analysis,
            outlier_detection,
            top_bottom_n,
            distribution_summary,
            cross_tabulation,
            save_finding,
        ],
        output_key="analysis_results",
    )


def _build_auditor() -> LlmAgent:
    return LlmAgent(
        name="auditor",
        model=GEMINI_MODEL,
        description="Runs the Q3 data quality audit.",
        instruction=(
            "You are a data quality auditor. Based on the data profile:\n"
            "{data_profile}\n\n"
            "Call q3_data_quality with the exact column names for:\n"
            "  order_id_col, quantity_col, price_col, discount_col\n\n"
            "This computes the 5 exact counts required by Q3:\n"
            "  1. Duplicate order IDs\n"
            "  2. Quantity outliers (IQR method)\n"
            "  3. Price format errors (null + negative)\n"
            "  4. Invalid discounts (null + outside 0-100)\n"
            "  5. Total null cells\n\n"
            "After the audit, call save_finding(key='Q3', title='Data "
            "Quality Audit', content=<results>)."
        ),
        tools=[q3_data_quality, data_quality_audit, save_finding],
        output_key="audit_results",
    )


def _build_synthesizer() -> LlmAgent:
    return LlmAgent(
        name="synthesizer",
        model=GEMINI_MODEL,
        description="Generates charts and compiles the final submission.",
        instruction=(
            "You are a report synthesizer. Your inputs:\n"
            "  Analyst results: {analysis_results}\n"
            "  Audit results: {audit_results}\n\n"
            "Steps:\n"
            "1. Call generate_chart for at least 3 key findings: "
            "Q1 (barh), Q2 (bar), Q4 (bar).\n"
            "2. Write an executive summary that ties Q1-Q4 insights together "
            "and call save_finding(key='Q5', title='Executive Summary', "
            "content=<summary>).\n"
            "3. Call compile_report() to build the final Q1-Q5 submission "
            "file in strict format.\n\n"
            "The submission must be saved to output/submission.txt."
        ),
        tools=[generate_chart, compile_report, save_finding],
        output_key="report_summary",
    )


def _build_validator() -> LlmAgent:
    return LlmAgent(
        name="validator",
        model=GEMINI_MODEL,
        description="Reviews the final submission for completeness.",
        instruction=(
            "You are a validation agent. Review the report summary:\n"
            "{report_summary}\n\n"
            "Verify:\n"
            "  - Q1, Q2, Q3, Q4 all have answers\n"
            "  - Q3 has all 5 required counts\n"
            "  - Q5 executive summary is present\n"
            "  - Submission is formatted correctly\n\n"
            "Output exactly one of:\n"
            "  PASS: <short confirmation>\n"
            "  FAIL: <specific gaps>"
        ),
        output_key="validation_status",
    )


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════


def build_squad(
    data_dir: str, problem_statement: Optional[str] = None
) -> SequentialAgent:
    """Build the 6-agent sequential pipeline using Google ADK.

    Args:
        data_dir: Directory containing the dataset files.
        problem_statement: Optional free-text problem statement.

    Returns:
        A `SequentialAgent` ready to be handed to an ADK `Runner`.
    """
    problem = (problem_statement or "").strip()

    squad = SequentialAgent(
        name="finance_squad",
        description=(
            "Autonomous 6-agent pipeline that ingests, analyzes, audits, "
            "visualizes, and validates any dataset -- built for Agentathon 2026."
        ),
        sub_agents=[
            _build_data_engineer(data_dir),
            _build_planner(problem),
            _build_analyst(),
            _build_auditor(),
            _build_synthesizer(),
            _build_validator(),
        ],
    )
    log.info("Built 6-agent ADK squad")
    return squad
