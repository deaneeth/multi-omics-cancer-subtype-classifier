"""
patient_converter.py
====================
In-memory conversion of per-modality omics files (from a real hospital or
research pipeline) into a single upload-ready CSV for the MLOmics demo.

Called by the Streamlit "Data Converter" tab.  All operations are in-memory
(no temporary files written to disk).

Public API
----------
convert_patient_data(cancer_type, sample_id, modality_files, cfg, ...) -> ConversionResult
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import IO

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_LABELS = {
    "mrna":  "mRNA Expression",
    "mirna": "miRNA Expression",
    "methy": "DNA Methylation",
    "cnv":   "Copy Number Variation",
}

MODALITY_NOTES = {
    "mrna": (
        "Expected: log2(TPM + 1), z-scored per gene.\n"
        "Feature names: gene symbols (e.g. ESR1, TP53, BRCA1)."
    ),
    "mirna": (
        "Expected: log2(RPM + 1), z-scored per miRNA.\n"
        "Feature names: any common format — hsa-miR-21, hsa.mir.21, "
        "hsa-miR-21-5p — all normalised automatically."
    ),
    "methy": (
        "Expected: M-values [log2(B/(1-B))], z-scored per probe.\n"
        "Feature names: gene symbols linked to CpG probes (e.g. BRCA1, ESR1)."
    ),
    "cnv": (
        "Expected: log2 copy-number ratios, averaged per gene, z-scored.\n"
        "Feature names: gene symbols (e.g. ERBB2, MYC, CDKN2A)."
    ),
}

# Data format options for the UI
DATA_FORMATS = {
    "zscore": "Already z-score normalized (mean~0, std~1)",
    "log2":   "Log2-transformed (e.g. log2(TPM+1), log2(RPM+1))",
    "raw":    "Raw counts / values (integers or floats, not log-transformed)",
    "beta":   "Beta values 0-1 (methylation only)",
}

# Which formats are valid per modality
VALID_FORMATS = {
    "mrna":  ["zscore", "log2", "raw"],
    "mirna": ["zscore", "log2", "raw"],
    "methy": ["zscore", "log2", "beta"],
    "cnv":   ["zscore", "log2"],
}


# ---------------------------------------------------------------------------
# Normalization transforms
# ---------------------------------------------------------------------------

def _transform_values(values: np.ndarray, data_format: str, mod: str) -> np.ndarray:
    """Apply the appropriate transform chain to bring values into z-score space.

    Transform chains:
        zscore: no-op (already normalized)
        log2:   z-score using per-feature mean/std estimation (single-sample approx)
        raw:    log2(x + 1) → z-score
        beta:   M-value transform [log2(b/(1-b))] → z-score (methylation only)

    For single-sample normalization, we approximate z-scoring by centering on the
    median and scaling by the IQR-estimated std. This is less accurate than cohort-
    based z-scoring but is the only option for single-patient uploads.
    """
    arr = values.copy().astype(np.float64)

    if data_format == "zscore":
        return arr

    if data_format == "raw":
        # Clip negatives (shouldn't exist in counts, but safety)
        arr = np.maximum(arr, 0.0)
        arr = np.log2(arr + 1.0)

    if data_format == "beta":
        # Beta values (0-1) -> M-values: M = log2(beta / (1 - beta))
        # Clip to avoid log(0) or log(inf)
        arr = np.clip(arr, 0.001, 0.999)
        arr = np.log2(arr / (1.0 - arr))

    # At this point data is in log2 space (for raw/beta) or already log2 (for log2 format)
    # Apply simple z-score: subtract median, divide by robust std
    # This is a single-sample approximation — not as accurate as cohort-based normalization
    median_val = np.nanmedian(arr)
    iqr = np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)
    # Estimate std from IQR: std ≈ IQR / 1.349 (for normal distribution)
    robust_std = iqr / 1.349 if iqr > 0 else 1.0
    arr = (arr - median_val) / robust_std

    return arr


# ---------------------------------------------------------------------------
# miRNA name normalisation
# ---------------------------------------------------------------------------

def _normalise_mirna(name: str) -> str:
    """Convert any common miRNA name variant to training-data dot format.

    Examples
    --------
    hsa-miR-21-5p  → hsa.mir.21
    hsa-let-7a-1   → hsa.let.7a.1
    HSA-MIR-21     → hsa.mir.21
    """
    s = str(name).strip().lower().replace("-", ".").replace("_", ".")
    for suffix in (".5p", ".3p"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


# ---------------------------------------------------------------------------
# Per-modality file loader
# ---------------------------------------------------------------------------

def _load_series(file_obj: IO[bytes], sample_id: str, mod: str) -> pd.Series:
    """Read a CSV file-object and return a Series {feature_name: value}.

    Supports two orientations:
      - features-as-rows : index = feature names, columns = sample IDs
      - features-as-cols : index = sample IDs,   columns = feature names
    Detection uses sample_id presence before falling back to shape heuristic.
    """
    df = pd.read_csv(file_obj, index_col=0)

    # Coerce numeric columns — handles any trailing spaces or string artefacts
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Detect orientation ──────────────────────────────────────────
    if sample_id in df.columns:
        series = df[sample_id]
    elif sample_id in df.index:
        series = df.loc[sample_id]
    elif df.shape[0] >= df.shape[1]:
        # More rows than cols → features are likely rows
        series = df.iloc[:, 0]
    else:
        series = df.iloc[0]

    series = series.dropna()

    # ── miRNA name normalisation ────────────────────────────────────
    if mod == "mirna":
        series.index = [_normalise_mirna(n) for n in series.index]

    return series.rename(sample_id)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModCoverage:
    total:   int
    matched: int
    missing: int

    @property
    def pct(self) -> float:
        return 100.0 * self.matched / self.total if self.total else 0.0

    @property
    def status(self) -> str:
        if self.matched == 0:
            return "not_provided"
        if self.pct >= 80:
            return "good"
        if self.pct >= 40:
            return "partial"
        return "poor"


@dataclass
class ConversionResult:
    df:          pd.DataFrame              # upload-ready DataFrame
    coverage:    dict[str, ModCoverage]    # per-modality coverage stats
    sample_id:   str
    cancer_type: str
    warnings:    list[str] = field(default_factory=list)

    def to_csv_bytes(self) -> bytes:
        return self.df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_patient_data(
    cancer_type:    str,
    sample_id:      str,
    modality_files: dict[str, IO[bytes] | None],
    cfg:            dict,
    data_format:    str = "zscore",
) -> ConversionResult:
    """Convert per-modality file objects to an upload-ready DataFrame.

    Parameters
    ----------
    cancer_type : str
        "GS-BRCA" or "GS-COAD".
    sample_id : str
        Patient identifier written into the output CSV.
    modality_files : dict
        Keys: "mrna", "mirna", "methy", "cnv".
        Values: file-like objects (from st.file_uploader) or None.
    cfg : dict
        App config loaded from ``app/model_artifacts/config_{c}.json``.
    data_format : str
        One of "zscore", "log2", "raw", "beta". Controls normalization.
        "zscore" = already normalized (no transform).
        "log2"   = log2-transformed, needs z-scoring.
        "raw"    = raw counts, needs log2(x+1) then z-scoring.
        "beta"   = methylation beta values (0-1), needs M-value then z-scoring.

    Returns
    -------
    ConversionResult
    """
    feature_names  = cfg["feature_names"]
    modality_order = cfg["modality_order"]
    modality_dims  = cfg["modality_dims"]

    output_vector = np.zeros(len(feature_names), dtype=np.float64)
    coverage: dict[str, ModCoverage] = {}
    warnings: list[str] = []

    for mod in modality_order:
        dim   = modality_dims[mod]
        start = sum(modality_dims[m] for m in modality_order[: modality_order.index(mod)])

        if modality_files.get(mod) is None:
            coverage[mod] = ModCoverage(total=dim, matched=0, missing=dim)
            continue

        try:
            series = _load_series(modality_files[mod], sample_id, mod)
        except Exception as exc:
            warnings.append(f"{MODALITY_LABELS[mod]}: failed to read file — {exc}")
            coverage[mod] = ModCoverage(total=dim, matched=0, missing=dim)
            continue

        matched = 0
        for idx in range(start, start + dim):
            bare = feature_names[idx].split("_", 1)[1]
            if bare in series.index:
                val = series[bare]
                output_vector[idx] = float(val) if pd.notna(val) else 0.0
                matched += 1

        # Apply normalization transform if data is not already z-scored
        if data_format != "zscore" and matched > 0:
            # Determine effective format for this modality
            effective_format = data_format
            # "beta" only applies to methylation; for other modalities treat as "log2"
            if effective_format == "beta" and mod != "methy":
                effective_format = "log2"
            # "raw" doesn't apply to CNV (CNV is always log2 ratios)
            if effective_format == "raw" and mod == "cnv":
                effective_format = "log2"

            # Get the matched values slice
            mod_slice = output_vector[start:start + dim]
            # Only transform non-zero values (zeros are missing/unmatched features)
            mask = np.zeros(dim, dtype=bool)
            for i in range(dim):
                bare = feature_names[start + i].split("_", 1)[1]
                if bare in series.index:
                    mask[i] = True

            if mask.any():
                matched_vals = mod_slice[mask]
                transformed = _transform_values(matched_vals, effective_format, mod)
                mod_slice[mask] = transformed
                output_vector[start:start + dim] = mod_slice

        coverage[mod] = ModCoverage(total=dim, matched=matched, missing=dim - matched)

        if coverage[mod].pct < 40:
            warnings.append(
                f"{MODALITY_LABELS[mod]}: only {coverage[mod].pct:.0f}% of expected "
                f"features matched. Check that feature names are gene symbols."
            )

    # Add format-specific warning
    if data_format != "zscore":
        warnings.append(
            f"Data was normalized from '{DATA_FORMATS[data_format]}' format using "
            f"single-sample approximation. Results may be less accurate than cohort-"
            f"based normalization. Batch effects are NOT corrected."
        )

    row = {"sample_id": sample_id}
    row.update(dict(zip(feature_names, output_vector)))
    df_out = pd.DataFrame([row])

    return ConversionResult(
        df=df_out,
        coverage=coverage,
        sample_id=sample_id,
        cancer_type=cancer_type,
        warnings=warnings,
    )
