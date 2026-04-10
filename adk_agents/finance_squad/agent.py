"""ADK Web entry point for the Finance AI Squad.

This module is auto-discovered by ``adk web`` when pointed at the
``adk_agents/`` directory. It exposes a ``root_agent`` variable -- the
full 6-agent :class:`SequentialAgent` -- which the web UI will visualise
and let judges interact with.

Model selection precedence (highest to lowest):
    1. ``GEMINI_MODEL`` environment variable (set this before launching
       ``adk web`` to override without editing code).
    2. ``config.GEMINI_MODEL`` from :mod:`config`.

When Google returns 503 UNAVAILABLE on the current model, export a
different model and restart the server, for example::

    export GEMINI_MODEL=gemini-2.0-flash-lite
    adk web ./adk_agents --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import config/, tools/, orchestrator/
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the working directory is the project root so relative paths work
os.chdir(_PROJECT_ROOT)

from orchestrator import build_squad  # noqa: E402

# Default dataset location relative to the project root
_DATA_DIR: str = str(_PROJECT_ROOT / "data")

# Model override from environment (optional). build_squad falls back to
# config.GEMINI_MODEL when this is empty.
_MODEL_OVERRIDE: str = os.environ.get("GEMINI_MODEL", "").strip() or ""

# Build the 6-agent squad. The Planner/Analyst instructions reference the
# data directory baked in at build time. Judges can ask the agent to
# analyse this dataset via the chat interface.
root_agent = build_squad(
    data_dir=_DATA_DIR,
    problem_statement=(
        "RetailIQ customer orders analysis. Run the standard Q1-Q5 pipeline: "
        "Q1 total revenue by product_category, Q2 average delivery by "
        "customer_region, Q3 data quality counts, Q4 return rate by "
        "payment_method, Q5 three-sentence executive summary."
    ),
    model=_MODEL_OVERRIDE or None,
)
