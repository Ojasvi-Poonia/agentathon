# Low Cortisol — Finance AI Squad

**Team:** Low Cortisol
**Event:** Agentathon 2026, Department of AI & Data Science, Ramaiah Institute of Technology
**Stack:** Google ADK (Agent Development Kit) + Gemini 2.0 Flash + Python

An autonomous 6-agent AI squad that ingests a raw customer-orders dataset, detects its schema via fuzzy matching, runs Q1-Q5 analysis following the exact problem-statement formulas, audits data quality, generates charts, and produces a formatted submission file — in under 3 seconds, with zero human intervention during the hands-off execution phase.

---

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
 |  (11 tools)        |  Q2: delivery by region
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
 |  Validator         |  PASS / FAIL completeness check
 +--------+----------+
          |
          v
   output/low-cortisol.txt  +  charts/
```

Each agent is a Google ADK `LlmAgent` with dedicated tools (auto-introspected from Python function signatures). The six agents are wrapped in a `SequentialAgent` and executed via `InMemoryRunner.run_async()`. Context flows between agents through ADK session-state `output_key` templating.

## Dual Execution Path

The system has **two pipelines** that both use the same tool functions:

1. **Deterministic pipeline** — pure pandas, no API calls, runs in ~2 seconds. This is what produces the scored submission. It's deterministic, reproducible, and works offline.
2. **Google ADK agent pipeline** — runs after the deterministic pipeline as a live demonstration. Shows the Gemini-powered autonomous reasoning for judges. Requires a Gemini API key.

If Gemini is unavailable, the deterministic submission is already saved to disk before the ADK pipeline even starts.

## Project Structure

```
.
├── config.py               # 21 role signatures for fuzzy column detection
├── orchestrator.py         # 6-agent ADK SequentialAgent pipeline
├── run_agents.py           # Entry point (deterministic + ADK)
├── tools/
│   ├── __init__.py         # Shared state + 18-tool registry
│   ├── data_ops.py         # Load, profile, clean, detect columns
│   ├── analysis.py         # Generic analysis + Q1-Q4 competition tools
│   └── reporting.py        # Charts + strict Q1-Q5 formatter
├── data/
│   └── train_data.csv      # Dev dataset from the contest organizers
├── output/                 # Generated submission + charts (gitignored)
├── logs/                   # Runtime logs (gitignored)
├── problem_statement.md    # Contest problem statement
├── output_format.txt       # Contest output format rules
├── README for contest.md   # Contest README from organizers
├── requirements.txt
├── .env.example
└── .gitignore
```

## Key Features

- **Zero-config column detection** — fuzzy name + type + statistics scoring maps any schema to 21 known roles with no manual configuration.
- **Consistent "clean then compute"** — numeric nulls AND format errors are uniformly median-imputed before computing revenue. All 3,500 rows contribute to Q1.
- **Strict Q1-Q5 format** — matches `output_format.txt` character-for-character: single-line per question, UTF-8, no markdown, exactly 3 sentences for Q5.
- **Sub-3-second execution** — the deterministic pipeline completes the full Q1-Q5 analysis in ~2.5 seconds, well under the 5-minute speed bonus threshold.
- **Self-recovering agents** — each tool call returns an error hint if it fails; agents retry with alternate parameters.
- **Manual override escape hatch** — `--map` flag for emergency column re-mapping if fuzzy detection ever fails.

## Prerequisites

- **Python 3.12** (recommended) or **3.11**.
  Google ADK declares `Requires-Python: >=3.9`, but 3.12 is the most stable and well-supported version. Python 3.9 works too but emits deprecation warnings from some dependencies.
- A Gemini API key (optional — only needed for the live ADK agent demo; the deterministic submission works without one).

## Installation

```bash
git clone <your-fork-url>
cd finance_squad

# Install dependencies
python3.12 -m pip install -r requirements.txt

# Copy the env template and add your Gemini key (optional)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY if you want to run the ADK demo
```

## Usage

### Primary submission (deterministic, fast, no API needed)

```bash
python3.12 run_agents.py --data ./data --team low-cortisol --fallback
```

This produces `output/low-cortisol.txt` — the file you submit — in about 2.5 seconds.

### Full run with Google ADK agent demo

```bash
python3.12 run_agents.py --data ./data --team low-cortisol
```

This runs the deterministic pipeline first (saves the submission), then runs the 6-agent ADK pipeline as a live demonstration.

### With a problem statement

```bash
python3.12 run_agents.py --data ./data --team low-cortisol --problem problem_statement.md
```

### Emergency column override (if auto-detection ever misses)

```bash
python3.12 run_agents.py --data ./data --team low-cortisol --fallback \
    --map "category=product_category,region=customer_region,revenue=_revenue"
```

### Pointing to test_data.csv at the 3-hour mark

```bash
# Drop test_data.csv into ./data/ (alongside or replacing train_data.csv)
python3.12 run_agents.py --data ./data --team low-cortisol --fallback
# Submit output/low-cortisol.txt
```

## How the Q1-Q5 Analysis Works

| Q | Computation | Source |
|---|-------------|--------|
| Q1 | `sum(qty * unit_price * (1 - discount/100))` grouped by `product_category`, ranked highest to lowest. Numeric columns imputed with median (including unit_price format errors) for consistency. | `tools/analysis.py: q1_revenue_by_category` + revenue precomputation in `run_agents.py` |
| Q2 | `mean(delivery_days)` grouped by `customer_region`, ranked slowest to fastest. | `tools/analysis.py: q2_avg_delivery_by_region` |
| Q3 | Five counts on the **original raw data** (not cleaned): duplicate order IDs, `qty > 1000`, price format errors (null + non-numeric + negative), invalid discounts (null + outside [0,100]), total null cells. | `tools/analysis.py: q3_data_quality` |
| Q4 | `mean(is_returned)` × 100 grouped by `payment_method`, ranked highest to lowest. Return status strings `"Returned" / "Not Returned" / "Pending"` are mapped to binary values. | `tools/analysis.py: q4_return_rate_by_payment` |
| Q5 | Auto-generated 3-sentence executive summary synthesizing Q1, Q2, and overall return rate. | `tools/reporting.py: _generate_q5` |

## Output Format

The submission file strictly matches `output_format.txt`:

```
Q1: Books: 224419419.16, Beauty: 177110970.21, ...
Q2: East: 12.37 days, South: 12.37 days, ...
Q3: Duplicate order IDs: 68, Quantity outliers: 12, Price format errors: 299, Invalid discounts: 297, Total null cells: 2795
Q4: UPI: 10.48%, Credit Card: 10.04%, ...
Q5: [sentence 1]. [sentence 2]. [sentence 3].
```

- 5 lines, each starting with the exact `Q{n}:` label
- UTF-8 encoded
- Under 200KB
- No markdown, no code blocks, no tables
- Comma-separated ranked labeled values for Q1, Q2, Q4
- All five required counts for Q3
- Exactly three sentences for Q5

## Team

**Low Cortisol** — Agentathon 2026
