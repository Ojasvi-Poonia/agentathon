"""
6-Agent Orchestrator using Gemini 2.0 Flash native function calling.

Agents:
  1. Data Engineer   — discover, load, profile, clean
  2. Planner         — analyze data profile + problem statement, create analysis plan
  3. Analyst         — execute dynamic analyses using generic tools
  4. Auditor         — data quality audit
  5. Synthesizer     — generate charts, compile report
  6. Validator       — review output completeness, flag gaps

Each agent runs as a separate Gemini chat with dedicated tools.
The orchestrator passes context sequentially and handles retries.
"""
import json
import logging
import time

import google.generativeai as genai

from config import GEMINI_MODEL, MAX_AGENT_TURNS, TEMPERATURE
from tools import TOOL_REGISTRY

log = logging.getLogger("Orchestrator")


# ═══════════════════════════════════════════════════════════════
#  HELPER: build Gemini FunctionDeclaration from simple dicts
# ═══════════════════════════════════════════════════════════════

def _param(description: str) -> genai.protos.Schema:
    return genai.protos.Schema(type=genai.protos.Type.STRING, description=description)


def _fn(name, desc, params, required=None):
    props = {k: _param(v) for k, v in params.items()}
    schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties=props,
        required=required or list(params.keys()),
    )
    return genai.protos.FunctionDeclaration(name=name, description=desc, parameters=schema)


# ═══════════════════════════════════════════════════════════════
#  TOOL DECLARATIONS PER AGENT
# ═══════════════════════════════════════════════════════════════

DATA_TOOLS = [genai.protos.Tool(function_declarations=[
    _fn("discover_files",
        "Scan a directory for CSV, JSON, XLSX, Parquet data files",
        {"directory": "Path to scan"}),
    _fn("load_and_profile",
        "Load a data file, generate deep statistical profile with column types/nulls/distributions, and auto-detect column roles",
        {"file_path": "Path to the data file"}),
    _fn("clean_data",
        "Clean dataset: parse dates, fill numeric nulls with median, fill categorical with mode. Returns transformation log",
        {"file_path": "Path to the dataset"}),
])]

ANALYSIS_TOOLS = [genai.protos.Tool(function_declarations=[
    _fn("group_aggregate",
        "Group by one column and aggregate another. Returns ranked results with share percentages",
        {"group_col": "Column to group by",
         "value_col": "Column to aggregate",
         "agg_func": "Aggregation: sum, mean, median, count, min, max, std"},
        required=["group_col", "value_col"]),
    _fn("compute_ratio",
        "Compute a new ratio = numerator_col / denominator_col. For financial ratios, margins, rates",
        {"numerator_col": "Numerator column",
         "denominator_col": "Denominator column",
         "ratio_name": "Name for the new ratio column"}),
    _fn("correlation_analysis",
        "Compute Pearson and Spearman correlation between two numeric columns",
        {"col1": "First column", "col2": "Second column"}),
    _fn("outlier_detection",
        "Detect outliers using IQR method. Returns count, bounds, and percentage",
        {"column": "Column to check for outliers"}),
    _fn("top_bottom_n",
        "Return top or bottom N rows ranked by a column",
        {"rank_col": "Column to rank by",
         "n": "Number of rows (default 10)",
         "ascending": "true for bottom N, false for top N"},
        required=["rank_col"]),
    _fn("distribution_summary",
        "Full statistical distribution: mean, median, skew, kurtosis, percentiles",
        {"column": "Column to analyze"}),
    _fn("cross_tabulation",
        "Pivot/cross-tabulation between two categorical columns, optionally aggregating a value column",
        {"row_col": "Row grouping column",
         "col_col": "Column grouping column",
         "value_col": "Optional value column to aggregate",
         "agg_func": "Aggregation function (default: count)"},
        required=["row_col", "col_col"]),
    _fn("save_finding",
        "Save an analysis finding for the final report. Call this after each analysis to record the answer",
        {"key": "Short identifier like Q1, Q2, revenue_insight",
         "title": "Human-readable title for this finding",
         "content": "The finding text — the actual answer"}),
])]

AUDIT_TOOLS = [genai.protos.Tool(function_declarations=[
    _fn("data_quality_audit",
        "Comprehensive quality audit: nulls, duplicates, outliers, type consistency. Pass comma-separated columns or empty for all",
        {"columns": "Comma-separated column names to audit, or empty for all"},
        required=[]),
    _fn("save_finding",
        "Save an audit finding for the final report",
        {"key": "Short identifier like Q3, data_quality",
         "title": "Title for this finding",
         "content": "The audit finding text"}),
])]

REPORT_TOOLS = [genai.protos.Tool(function_declarations=[
    _fn("generate_chart",
        "Generate a chart from the data. Saves PNG to output/",
        {"chart_type": "Chart type: bar, barh, line, scatter, pie",
         "x_col": "Column for x-axis or grouping",
         "y_col": "Column for y-axis or values",
         "title": "Chart title"}),
    _fn("compile_report",
        "Compile all saved findings into the final submission file",
        {"output_format": "Format: text"},
        required=[]),
    _fn("save_finding",
        "Save a synthesis finding for the final report",
        {"key": "Short identifier",
         "title": "Title",
         "content": "The finding text"}),
])]


# ═══════════════════════════════════════════════════════════════
#  GEMINI AGENT
# ═══════════════════════════════════════════════════════════════

class GeminiAgent:
    """Single AI agent backed by Gemini with function calling and error recovery."""

    def __init__(self, name, role, tools=None, model=GEMINI_MODEL):
        self.name = name
        config = {"temperature": TEMPERATURE}
        if tools:
            self.model = genai.GenerativeModel(
                model_name=model,
                system_instruction=role,
                tools=tools,
                generation_config=genai.GenerationConfig(**config),
            )
        else:
            self.model = genai.GenerativeModel(
                model_name=model,
                system_instruction=role,
                generation_config=genai.GenerationConfig(**config),
            )
        self.chat = self.model.start_chat()
        self.tools_available = tools is not None

    def execute(self, task: str, max_turns: int = MAX_AGENT_TURNS) -> str:
        """Run a task with automatic tool-calling loop and error recovery."""
        print(f"\n  [{self.name}] Starting...")
        log.info(f"[{self.name}] Task: {task[:200]}")

        try:
            response = self.chat.send_message(task)
        except Exception as e:
            err_msg = f"[{self.name}] Failed to send initial message: {e}"
            log.error(err_msg)
            return err_msg

        for turn in range(max_turns):
            fn_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                result = "\n".join(text_parts)
                print(f"  [{self.name}] Done (turn {turn + 1})")
                log.info(f"[{self.name}] Completed in {turn + 1} turns")
                return result

            fn_responses = []
            for fc in fn_calls:
                fname = fc.name
                args = dict(fc.args)
                print(f"    > {fname}({json.dumps(args)[:100]})")

                if fname in TOOL_REGISTRY:
                    try:
                        result = TOOL_REGISTRY[fname](**args)
                    except Exception as e:
                        result = json.dumps({
                            "error": str(e),
                            "hint": f"Tool '{fname}' failed. Try different parameters or another approach.",
                        })
                        log.warning(f"  Tool {fname} error: {e}")
                else:
                    result = json.dumps({
                        "error": f"Unknown tool: {fname}",
                        "available_tools": list(TOOL_REGISTRY.keys()),
                    })

                preview = result[:150].replace("\n", " ")
                print(f"      = {preview}")
                log.info(f"  {fname} -> {preview}")

                fn_responses.append(
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=fname,
                        response={"result": result},
                    ))
                )

            try:
                response = self.chat.send_message(fn_responses)
            except Exception as e:
                err_msg = f"[{self.name}] Error sending tool responses: {e}"
                log.error(err_msg)
                return err_msg

        return f"[{self.name}] Reached max turns ({max_turns}). Partial results may be saved."


# ═══════════════════════════════════════════════════════════════
#  SQUAD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class SquadOrchestrator:
    """Sequential multi-agent orchestrator with context passing."""

    def __init__(self):
        self.pipeline = []
        self.context = ""
        self.execution_log = []

    def add(self, agent: GeminiAgent, task: str):
        self.pipeline.append((agent, task))
        return self

    def run(self) -> str:
        print("\n" + "=" * 60)
        print("  SQUAD ORCHESTRATOR -- Autonomous Execution")
        print("=" * 60)

        for i, (agent, task_template) in enumerate(self.pipeline):
            print(f"\n{'~' * 55}")
            print(f"  STAGE {i+1}/{len(self.pipeline)}: {agent.name}")
            print(f"{'~' * 55}")

            full_task = task_template
            if self.context:
                full_task = (
                    f"Context from previous agents:\n{self.context[:3000]}\n\n"
                    f"Your task:\n{task_template}"
                )

            t0 = time.time()
            try:
                result = agent.execute(full_task)
                elapsed = time.time() - t0
                self.context = result
                self.execution_log.append({
                    "agent": agent.name,
                    "status": "success",
                    "elapsed_s": round(elapsed, 1),
                    "output_preview": result[:300],
                })
                print(f"  [{agent.name}] Completed in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                err = f"Agent error: {e}"
                print(f"  [{agent.name}] FAILED: {err}")
                log.error(err)
                self.context = err
                self.execution_log.append({
                    "agent": agent.name,
                    "status": "error",
                    "elapsed_s": round(elapsed, 1),
                    "error": str(e),
                })

        print(f"\n{'=' * 60}")
        print("  ALL AGENTS COMPLETE")
        print(f"{'=' * 60}")
        return self.context


# ═══════════════════════════════════════════════════════════════
#  BUILD THE 6-AGENT SQUAD
# ═══════════════════════════════════════════════════════════════

def build_squad(data_dir: str, problem_statement: str = "") -> SquadOrchestrator:
    """Build the full 6-agent finance analysis squad."""

    orch = SquadOrchestrator()

    # ── Agent 1: Data Engineer ────────────────────────────────
    orch.add(
        GeminiAgent(
            name="Data Engineer",
            role=(
                "You are a data engineering agent. Your job is to discover data files, "
                "load them, generate deep statistical profiles, and clean the data. "
                "Steps: 1) discover_files to find all data files, 2) load_and_profile each file, "
                "3) clean_data for each file. "
                "Report the EXACT column names, their data types, detected roles, "
                "and key statistics. This information is critical for downstream agents."
            ),
            tools=DATA_TOOLS,
        ),
        f"Scan '{data_dir}' for all data files. Load and profile each file. Clean the data. "
        f"Report: exact column names, data types, detected roles, row count, null count, "
        f"and sample values. Be thorough -- downstream agents depend on your column report."
    )

    # ── Agent 2: Planner ──────────────────────────────────────
    planner_context = ""
    if problem_statement:
        planner_context = f"\n\nPROBLEM STATEMENT:\n{problem_statement}\n"

    orch.add(
        GeminiAgent(
            name="Planner",
            role=(
                "You are a strategic planning agent. Given a data profile and a problem statement, "
                "you create a detailed analysis plan. You decide WHICH analyses to run and "
                "WHICH columns to use. Your plan must be specific and actionable.\n\n"
                "Available analysis tools for the Analyst agent:\n"
                "- group_aggregate(group_col, value_col, agg_func) -- group and aggregate\n"
                "- compute_ratio(numerator_col, denominator_col, ratio_name) -- compute ratios\n"
                "- correlation_analysis(col1, col2) -- correlation between columns\n"
                "- outlier_detection(column) -- find outliers\n"
                "- top_bottom_n(rank_col, n, ascending) -- top/bottom N\n"
                "- distribution_summary(column) -- statistical distribution\n"
                "- cross_tabulation(row_col, col_col, value_col, agg_func) -- pivot analysis\n\n"
                "Output a numbered plan with exact tool calls and column names. "
                "Each step should answer a specific question. Label them Q1, Q2, Q3, etc.\n"
                "Include a Q for data quality audit and a final Q for executive summary."
            ),
            tools=None,
        ),
        f"Based on the data profile from the Data Engineer, create a detailed analysis plan. "
        f"Decide what questions to answer and which tool calls to make with exact column names."
        f"{planner_context}"
    )

    # ── Agent 3: Analyst ──────────────────────────────────────
    orch.add(
        GeminiAgent(
            name="Analyst",
            role=(
                "You are a financial analyst agent. Execute the analysis plan from the Planner. "
                "For EACH question in the plan:\n"
                "1) Call the appropriate analysis tool with the exact column names\n"
                "2) Interpret the results\n"
                "3) Call save_finding to record the answer\n\n"
                "IMPORTANT: After each analysis tool call, immediately call save_finding "
                "with the key (Q1, Q2, etc.), a title, and a clear answer. "
                "The content should be a direct answer to the question, including key numbers.\n\n"
                "If a tool call fails, try alternative columns or approaches. "
                "Do not stop on errors -- skip and move to the next question."
            ),
            tools=ANALYSIS_TOOLS,
        ),
        "Execute ALL the analysis steps from the Planner's plan. "
        "For each step: call the analysis tool, interpret results, then call save_finding. "
        "Use the EXACT column names from the data profile. Complete every question in the plan."
    )

    # ── Agent 4: Auditor ──────────────────────────────────────
    orch.add(
        GeminiAgent(
            name="Auditor",
            role=(
                "You are a data quality auditor. Run a comprehensive audit on the dataset. "
                "Check for: duplicate rows, null values, outliers, data type issues. "
                "Call data_quality_audit, then save_finding with the audit results. "
                "Be specific about counts and quality score."
            ),
            tools=AUDIT_TOOLS,
        ),
        "Run a full data quality audit. Report: duplicate count, null count per column, "
        "outlier counts, and overall quality score. Save findings for the report."
    )

    # ── Agent 5: Synthesizer ──────────────────────────────────
    orch.add(
        GeminiAgent(
            name="Synthesizer",
            role=(
                "You are a report synthesis agent. Your tasks:\n"
                "1) Generate appropriate charts for the key findings "
                "(use generate_chart with bar/barh/line/scatter/pie)\n"
                "2) Write an executive summary that ties all findings together\n"
                "3) Call save_finding with key='executive_summary' for the summary\n"
                "4) Call compile_report to generate the final submission file\n\n"
                "Generate at least 3 charts. The executive summary should highlight "
                "the most impactful findings and provide actionable recommendations."
            ),
            tools=REPORT_TOOLS,
        ),
        "Generate charts for key findings. Write an executive summary. "
        "Call compile_report to produce the final submission."
    )

    # ── Agent 6: Validator ────────────────────────────────────
    orch.add(
        GeminiAgent(
            name="Validator",
            role=(
                "You are a validation agent. Review the compiled submission for completeness. "
                "Check that all questions from the plan have answers. "
                "Check that the executive summary exists. "
                "If anything is missing, note it clearly. "
                "Output a final validation status: PASS or FAIL with details."
            ),
            tools=None,
        ),
        "Review the submission from the Synthesizer. Verify all questions are answered, "
        "charts are generated, and the executive summary is present. "
        "Output PASS or FAIL with specific details about any gaps."
    )

    return orch
