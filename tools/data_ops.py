"""
Data operations: discovery, loading, profiling, cleaning, column detection.
Handles any dataset schema -- e-commerce, finance, or unknown.
"""
import json
import os
import glob
import hashlib
import logging

import numpy as np
import pandas as pd

import tools as state
from config import ROLE_SIGNATURES, W_NAME, W_TYPE, W_STATS, MIN_CONFIDENCE

log = logging.getLogger("DataOps")


# ═══════════════════════════════════════════════════════════════
#  FUZZY COLUMN MAPPER
# ═══════════════════════════════════════════════════════════════

def _name_score(col_name, patterns) -> float:
    col = col_name.lower().strip().replace(" ", "_")
    if col in patterns:
        return 1.0
    for p in patterns:
        if p in col or col in p:
            return 0.8
    col_tokens = set(col.replace("_", " ").split())
    best = 0.0
    for p in patterns:
        p_tokens = set(p.replace("_", " ").split())
        overlap = col_tokens & p_tokens
        if overlap:
            best = max(best, 0.5 * len(overlap) / max(len(col_tokens), len(p_tokens)))
    return best


def _type_score(series: pd.Series, hint: str) -> float:
    if hint == "number":
        numeric = pd.to_numeric(series, errors="coerce")
        return float(numeric.notna().sum()) / max(len(series), 1)
    if hint == "object":
        return 1.0 if series.dtype == "object" else 0.3
    if hint == "binary":
        uniq = series.dropna().unique()
        if len(uniq) <= 6:
            binary_vals = {0, 1, True, False, "Yes", "No", "yes", "no", "0", "1"}
            return sum(1 for v in uniq if v in binary_vals) / max(len(uniq), 1)
        return 0.1
    if hint == "date":
        sample = series.dropna().head(30).astype(str)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return float(parsed.notna().sum()) / max(len(sample), 1)
        except Exception:
            return 0.0
    return 0.5


def _stats_score(series: pd.Series, sig: dict) -> float:
    if "cardinality_range" in sig:
        n = series.nunique()
        lo, hi = sig["cardinality_range"]
        if lo <= n <= hi:
            return 1.0
        return 0.3 if n < lo else max(0.1, 1.0 - (n - hi) / 1000)
    if "uniqueness_threshold" in sig:
        return min(1.0, (series.nunique() / max(len(series), 1)) / sig["uniqueness_threshold"])
    if "value_range" in sig:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            return 0.0
        lo, hi = sig["value_range"]
        return float(((numeric >= lo) & (numeric <= hi)).mean())
    return 0.5


def detect_columns(df: pd.DataFrame) -> dict:
    """Confidence-scored fuzzy column detection. Returns {role: column_name}."""
    scores = {}

    for role, sig in ROLE_SIGNATURES.items():
        role_scores = []
        for col in df.columns:
            ns = _name_score(col, sig["name_patterns"])
            ts = _type_score(df[col], sig["type_hint"])
            ss = _stats_score(df[col], sig)
            weighted = ns * W_NAME + ts * W_TYPE + ss * W_STATS
            role_scores.append((col, round(weighted, 3)))
        role_scores.sort(key=lambda x: x[1], reverse=True)
        scores[role] = role_scores

    mapping = {}
    used = set()
    role_order = sorted(scores, key=lambda r: scores[r][0][1] if scores[r] else 0, reverse=True)

    for role in role_order:
        for col, score in scores[role]:
            if col not in used and score > MIN_CONFIDENCE:
                mapping[role] = col
                used.add(col)
                log.info(f"  {role:20s} -> {col:25s} (conf={score:.2f})")
                break

    if "revenue" not in mapping and "price" in mapping:
        mapping["revenue"] = mapping["price"]
        log.info(f"  {'revenue':20s} -> {mapping['price']:25s} (fallback: using price)")

    state.column_map.update(mapping)
    return mapping


# ═══════════════════════════════════════════════════════════════
#  STATISTICAL PROFILER
# ═══════════════════════════════════════════════════════════════

def _profile_column(series: pd.Series) -> dict:
    p = {
        "dtype": str(series.dtype),
        "count": len(series),
        "nulls": int(series.isna().sum()),
        "null_pct": round(series.isna().mean() * 100, 2),
        "unique": int(series.nunique()),
    }
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            p.update({
                "mean": round(float(clean.mean()), 4),
                "median": round(float(clean.median()), 4),
                "std": round(float(clean.std()), 4),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "q1": float(clean.quantile(0.25)),
                "q3": float(clean.quantile(0.75)),
            })
    else:
        top = series.value_counts().head(5)
        p["top_values"] = {str(k): int(v) for k, v in top.items()}
    return p


# ═══════════════════════════════════════════════════════════════
#  TOOL FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def discover_files(directory: str) -> str:
    """Scan directory for CSV, JSON, XLSX, Parquet data files."""
    log.info(f"Scanning {directory}")
    files = []
    for ext in ("csv", "json", "xlsx", "xls", "tsv", "parquet"):
        for f in glob.glob(os.path.join(directory, f"**/*.{ext}"), recursive=True):
            files.append({
                "path": f,
                "extension": ext,
                "size_kb": round(os.path.getsize(f) / 1024, 1),
            })
    log.info(f"Found {len(files)} files")
    return json.dumps({"files": files, "count": len(files)}, indent=2)


def load_and_profile(file_path: str) -> str:
    """
    Load a dataset with encoding detection, generate deep statistical profile,
    and auto-detect column roles via fuzzy matching.
    """
    log.info(f"Loading {file_path}")
    try:
        ext = os.path.splitext(file_path)[1].lower()
        df = None

        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)

        if df is None or df.empty:
            return json.dumps({"error": f"Failed to load {file_path}"})

        state.store["main"] = df
        state.store["original"] = df.copy()
        state.store[file_path] = df

        fingerprint = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]

        col_profiles = {}
        for col in df.columns:
            col_profiles[col] = _profile_column(df[col])

        col_map = detect_columns(df)

        state.metadata["fingerprint"] = fingerprint
        state.metadata["shape"] = list(df.shape)

        summary = {
            "file": file_path,
            "fingerprint": fingerprint,
            "rows": df.shape[0],
            "columns": list(df.columns),
            "dtypes": {c: str(d) for c, d in df.dtypes.items()},
            "total_nulls": int(df.isnull().sum().sum()),
            "total_duplicates": int(df.duplicated().sum()),
            "column_profiles": col_profiles,
            "detected_roles": col_map,
            "sample_rows": json.loads(df.head(3).to_json(orient="records", default_handler=str)),
        }
        log.info(f"  Shape: {df.shape}, Nulls: {summary['total_nulls']}")
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        log.error(f"Load failed: {e}")
        return json.dumps({"error": str(e)})


def clean_data(file_path: str) -> str:
    """
    Intelligent data cleaning: parse dates, fill numeric nulls with median,
    fill categorical nulls with mode. Does NOT drop rows.
    """
    log.info("Cleaning data")
    try:
        df = state.store.get(file_path, state.store.get("main"))
        if df is None:
            return json.dumps({"error": "No data loaded"})
        df = df.copy()
        changes = []
        orig_nulls = int(df.isnull().sum().sum())

        for c in df.select_dtypes(include=["object"]).columns:
            sample = df[c].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, format="mixed")
                if parsed.notna().sum() > len(sample) * 0.8:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    changes.append(f"Parsed {c} as datetime")
            except (ValueError, TypeError):
                pass

        for c in df.select_dtypes(include=[np.number]).columns:
            n = int(df[c].isna().sum())
            if n > 0:
                med = df[c].median()
                df[c] = df[c].fillna(med)
                changes.append(f"Filled {n} nulls in {c} with median={med:.2f}")

        for c in df.select_dtypes(include=["object"]).columns:
            n = int(df[c].isna().sum())
            if n > 0:
                m = df[c].mode()
                if len(m) > 0:
                    df[c] = df[c].fillna(m.iloc[0])
                    changes.append(f"Filled {n} nulls in {c} with mode='{m.iloc[0]}'")

        state.store["cleaned"] = df
        final_nulls = int(df.isnull().sum().sum())
        log.info(f"  Nulls: {orig_nulls} -> {final_nulls}")

        return json.dumps({
            "rows": df.shape[0],
            "nulls_before": orig_nulls,
            "nulls_after": final_nulls,
            "transformations": changes,
        }, indent=2, default=str)
    except Exception as e:
        log.error(f"Clean failed: {e}")
        return json.dumps({"error": str(e)})
