"""
Configuration and column signature definitions for the Finance AI Squad.
Expanded to handle generic datasets + enterprise finance data.
"""

GEMINI_MODEL = "gemini-2.0-flash"
MAX_AGENT_TURNS = 15
TEMPERATURE = 0.05

# Weights for fuzzy column matching
W_NAME = 0.50
W_TYPE = 0.25
W_STATS = 0.25

# Minimum confidence to accept a column mapping
MIN_CONFIDENCE = 0.20

ROLE_SIGNATURES = {
    # ── Identifiers ──────────────────────────────────────────
    "entity_id": {
        "name_patterns": [
            "order_id", "id", "transaction_id", "company_id", "ticker",
            "symbol", "stock_id", "entity_id", "account_id", "customer_id",
            "record_id", "invoice_id", "trade_id", "order_no", "orderid",
            "serial_no", "loan_id", "policy_id", "claim_id", "isin",
            "ref_code", "reference", "order_number", "txn_no", "txn_id",
        ],
        "type_hint": "object",
        "uniqueness_threshold": 0.7,
    },
    "entity_name": {
        "name_patterns": [
            "company", "company_name", "name", "firm", "organization",
            "customer_name", "product_name", "stock_name", "entity",
            "borrower", "issuer", "counterparty", "vendor", "supplier",
        ],
        "type_hint": "object",
        "cardinality_range": (2, 10000),
    },

    # ── Categorical / Grouping ───────────────────────────────
    "category": {
        "name_patterns": [
            "category", "product_category", "sector", "industry",
            "segment", "department", "type", "class", "group",
            "asset_class", "fund_type", "instrument_type", "sub_sector",
            "business_line", "vertical", "classification",
            "product_line", "product_type", "item_type", "cat", "dept",
        ],
        "type_hint": "object",
        "cardinality_range": (2, 100),
    },
    "region": {
        "name_patterns": [
            "region", "country", "state", "geography", "market",
            "territory", "zone", "location", "city", "exchange",
            "delivery_region", "area", "geo", "branch", "office",
        ],
        "type_hint": "object",
        "cardinality_range": (2, 60),
    },
    "payment": {
        "name_patterns": [
            "payment_method", "payment", "payment_type", "pay_mode",
            "payment_mode", "instrument", "currency", "pay_method",
            "settlement_mode", "settlement", "txn_mode", "mode_of_payment",
        ],
        "type_hint": "object",
        "cardinality_range": (2, 20),
    },
    "rating": {
        "name_patterns": [
            "rating", "credit_rating", "grade", "risk_rating",
            "quality_rating", "esg_rating", "bond_rating", "tier",
        ],
        "type_hint": "object",
        "cardinality_range": (2, 25),
    },

    # ── Revenue / Sales ──────────────────────────────────────
    "revenue": {
        "name_patterns": [
            "revenue", "total_revenue", "sales", "net_sales",
            "gross_revenue", "total_amount", "amount", "total",
            "order_value", "total_price", "gross_sales", "turnover",
            "total_income", "top_line",
        ],
        "type_hint": "number",
    },
    "profit": {
        "name_patterns": [
            "profit", "net_income", "net_profit", "earnings",
            "operating_income", "operating_profit", "ebit",
            "pbt", "pat", "gross_profit", "bottom_line",
            "profit_after_tax", "profit_before_tax",
        ],
        "type_hint": "number",
    },
    "ebitda": {
        "name_patterns": [
            "ebitda", "operating_cashflow", "operating_cash_flow",
            "ebitda_margin",
        ],
        "type_hint": "number",
    },

    # ── Balance Sheet ────────────────────────────────────────
    "assets": {
        "name_patterns": [
            "total_assets", "assets", "net_assets", "total_asset",
            "book_value", "nav", "aum", "gross_assets",
        ],
        "type_hint": "number",
    },
    "liabilities": {
        "name_patterns": [
            "total_liabilities", "liabilities", "total_debt",
            "debt", "borrowings", "total_liability", "long_term_debt",
            "short_term_debt", "current_liabilities",
        ],
        "type_hint": "number",
    },
    "equity": {
        "name_patterns": [
            "equity", "shareholders_equity", "net_worth",
            "total_equity", "shareholder_equity", "book_equity",
            "stockholders_equity",
        ],
        "type_hint": "number",
    },

    # ── Market Data ──────────────────────────────────────────
    "market_cap": {
        "name_patterns": [
            "market_cap", "market_capitalization", "mcap",
            "market_value", "enterprise_value", "ev",
        ],
        "type_hint": "number",
    },
    "price": {
        "name_patterns": [
            "price", "stock_price", "share_price", "close",
            "closing_price", "last_price", "unit_price",
            "cost", "mrp", "selling_price", "open", "high", "low",
            "adj_close", "current_price", "rate_per_unit", "rate",
            "item_price", "base_price",
        ],
        "type_hint": "number",
    },
    "quantity": {
        "name_patterns": [
            "quantity", "qty", "units", "volume", "shares",
            "num_items", "trade_volume", "count", "item_count",
            "shares_outstanding", "pieces", "items", "no_of_items",
        ],
        "type_hint": "number",
        "value_range": (0, 1e15),
    },

    # ── Ratios / Margins ────────────────────────────────────
    "margin": {
        "name_patterns": [
            "margin", "gross_margin", "operating_margin",
            "net_margin", "profit_margin", "ebitda_margin",
            "contribution_margin",
        ],
        "type_hint": "number",
        "value_range": (-200, 200),
    },
    "ratio": {
        "name_patterns": [
            "pe_ratio", "p_e", "debt_to_equity", "current_ratio",
            "roe", "roa", "roce", "eps", "dividend_yield",
            "beta", "sharpe_ratio", "price_to_book", "pb_ratio",
            "quick_ratio", "interest_coverage", "asset_turnover",
        ],
        "type_hint": "number",
    },
    "discount": {
        "name_patterns": [
            "discount", "discount_percent", "disc", "rebate",
            "markdown", "discount_pct", "discount_rate",
        ],
        "type_hint": "number",
        "value_range": (0, 100),
    },

    # ── Operational ──────────────────────────────────────────
    "delivery_days": {
        "name_patterns": [
            "delivery_days", "delivery_time", "days_to_deliver",
            "shipping_days", "lead_time", "transit_days", "tat",
            "settlement_days", "processing_time", "delivery_day",
            "turnaround_time",
        ],
        "type_hint": "number",
        "value_range": (0, 365),
    },
    "date": {
        "name_patterns": [
            "date", "order_date", "transaction_date", "trade_date",
            "fiscal_year", "fiscal_quarter", "year", "quarter",
            "period", "reporting_date", "settlement_date",
            "created_at", "timestamp", "report_date",
        ],
        "type_hint": "date",
    },

    # ── Binary Flags ─────────────────────────────────────────
    "returned": {
        "name_patterns": [
            "returned", "is_returned", "return_flag", "return_status",
            "cancelled", "is_cancelled", "refunded", "defaulted",
            "is_default", "churned", "is_fraud", "flagged",
        ],
        "type_hint": "binary",
    },
}
