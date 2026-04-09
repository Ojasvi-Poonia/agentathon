# Enterprise Finance AI Squad

**Agentathon 2026** — Autonomous multi-agent data analysis system.

6 AI agents that independently ingest any dataset, detect its schema, run analysis, audit quality, generate charts, and produce a formatted report — with zero human intervention.

## Architecture

```
 Dataset (CSV/JSON/XLSX)
         |
         v
 +-------------------+
 |  Data Engineer     |  discover -> load -> profile -> clean
 +--------+----------+
          |  column map + data profile
          v
 +-------------------+
 |  Planner           |  reads profile, creates dynamic analysis plan
 +--------+----------+
          |  analysis plan
          v
 +-------------------+
 |  Analyst           |  Q1: revenue by category
 |  (8 generic tools) |  Q2: delivery by region
 |                    |  Q4: return rate by payment
 +--------+----------+
          |
          v
 +-------------------+
 |  Auditor           |  Q3: data quality (5 counts)
 +--------+----------+
          |
          v
 +-------------------+
 |  Synthesizer       |  Q5: executive summary + charts
 +--------+----------+
          |
          v
 +-------------------+
 |  Validator         |  checks completeness: PASS / FAIL
 +--------+----------+
          |
          v
   output/submission.txt + charts/
```

## Tech Stack

- **LLM**: Gemini 2.0 Flash (native function calling)
- **Framework**: Custom Python orchestrator (compatible with Google ADK)
- **Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Column Detection**: Fuzzy matching with 3-signal confidence scoring (name + type + statistics)

## Project Structure

```
config.py              # 21 role signatures for fuzzy column detection
orchestrator.py        # 6-agent system + Gemini function calling
run_agents.py          # Entry point (deterministic + agent pipeline)
tools/
  __init__.py          # Shared state + 18-tool registry
  data_ops.py          # Load, profile, clean, detect columns
  analysis.py          # Generic analysis + competition Q1-Q4
  reporting.py         # Charts + strict Q1-Q5 formatter
```

## Quick Start

```bash
pip install -r requirements.txt

# Set Gemini API key
echo "GEMINI_API_KEY=your_key" > .env

# Run (deterministic — fast, no API needed)
python3 run_agents.py --data ./data --fallback

# Run (full agent pipeline)
python3 run_agents.py --data ./data

# Manual column override if needed
python3 run_agents.py --data ./data --fallback --map "category=ProductType,revenue=Sales"
```

## Key Features

- **Zero-config column detection** — works on any dataset via fuzzy name/type/stats matching
- **Hands-off execution** — no human intervention after code freeze
- **Deterministic fallback** — runs without LLM if API is unavailable
- **Sub-2-second execution** — deterministic pipeline completes in ~1.7s
- **Self-recovering agents** — errors are caught and alternative approaches tried
- **Strict output format** — matches scorer's Q1-Q5 parser exactly

## Team

Built for the Agentathon 2026, Department of AI & DS, MSRIT.
