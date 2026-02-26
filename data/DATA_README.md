# MLOmics Dataset — Download & Setup Instructions

> **⚠️ Disclaimer:** This data is used for academic research only. Not for clinical use.
> This is an academic research prototype. It should not be used for any clinical decision-making.

---

## 1. Data Source

| Property | Detail |
|---|---|
| **Dataset** | MLOmics: Cancer Multi-Omics Benchmark |
| **Primary Source** | [Figshare](https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127) |
| **Alternative Source** | [HuggingFace](https://huggingface.co/datasets/AIBIC/MLOmics) |
| **License** | CC-BY-4.0 |
| **Original TCGA Data** | Source data originates from The Cancer Genome Atlas (TCGA) |

---

## 2. Citation

If you use this dataset, please cite:

> Chen, Z. et al. "MLOmics: Cancer Multi-Omics Database for Machine Learning Research."
> Figshare. Dataset. https://doi.org/10.6084/m9.figshare.28729127

---

## 3. What to Download

This project uses **Top** (ANOVA-selected) features from the **Classification** datasets
for two cancer types: **GS-BRCA** and **GS-COAD**.

### GS-BRCA (Breast Cancer) — Top Features

| File | Description | Size |
|---|---|---|
| `BRCA_mRNA_top.csv` | mRNA expression (5,000 features × 671 samples) | ~61 MB |
| `BRCA_miRNA_top.csv` | miRNA expression (366 features × 671 samples) | ~2 MB |
| `BRCA_Methy_top.csv` | DNA Methylation (5,000 features × 671 samples) | ~62 MB |
| `BRCA_CNV_top.csv` | Copy Number Variation (5,000 features × 671 samples) | ~61 MB |
| `54814634_BRCA_label_num.csv` | Subtype labels (671 samples, 5 subtypes) | ~1 KB |

### GS-COAD (Colon Cancer) — Top Features

| File | Description | Size |
|---|---|---|
| `COAD_mRNA_top.csv` | mRNA expression (5,000 features × 260 samples) | ~24 MB |
| `COAD_miRNA_top.csv` | miRNA expression (200 features × 260 samples) | ~1 MB |
| `COAD_Methy_top.csv` | DNA Methylation (5,000 features × 260 samples) | ~24 MB |
| `COAD_CNV_top.csv` | Copy Number Variation (5,000 features × 260 samples) | ~24 MB |
| `54814562_COAD_label_num.csv` | Subtype labels (260 samples, 4 subtypes) | ~1 KB |

**Total download size:** ~260 MB (10 files)

---

## 4. Where to Place Files

Place downloaded files in these directories (relative to project root):

```
data/raw/GS-BRCA/Top/
├── BRCA_mRNA_top.csv
├── BRCA_miRNA_top.csv
├── BRCA_Methy_top.csv
├── BRCA_CNV_top.csv
└── 54814634_BRCA_label_num.csv

data/raw/GS-COAD/Top/
├── COAD_mRNA_top.csv
├── COAD_miRNA_top.csv
├── COAD_Methy_top.csv
├── COAD_CNV_top.csv
└── 54814562_COAD_label_num.csv
```

---

## 5. Dataset Statistics (Verified from Actual Files)

### Sample Counts

| Cancer | Samples | mRNA Features | miRNA Features | Methy Features | CNV Features | Subtypes |
|--------|---------|---------------|----------------|----------------|--------------|----------|
| GS-BRCA | 671 | 5,000 | 366 | 5,000 | 5,000 | 5 |
| GS-COAD | 260 | 5,000 | 200 | 5,000 | 5,000 | 4 |

### Subtype Label Distributions

**GS-BRCA (5 subtypes):**

| Label | Count | Percentage |
|-------|-------|------------|
| 0 | 353 | 52.6% |
| 1 | 42 | 6.3% |
| 2 | 132 | 19.7% |
| 3 | 31 | 4.6% |
| 4 | 113 | 16.8% |

**GS-COAD (4 subtypes):**

| Label | Count | Percentage |
|-------|-------|------------|
| 0 | 174 | 66.9% |
| 1 | 48 | 18.5% |
| 2 | 34 | 13.1% |
| 3 | 4 | 1.5% |

> **Note:** Both datasets exhibit class imbalance. GS-COAD label 3 has only 4 samples — this may impact model performance for that subtype.

---

## 6. Important Data Format Notes

### CSV Layout (CRITICAL)

MLOmics CSV files are **transposed** relative to standard ML format:

- **Rows** = features (genes / CpG sites / CNV regions)
- **Columns** = samples (patients)
- **First column** = feature identifiers
- **Header row** = sample IDs

**You MUST transpose the data** (to samples × features) before using it for model input.

### Label File Format

- Single column named `Label`
- Integer values starting from `0`
- Row order corresponds to sample order in modality files

### Feature Scale: Top

Top features are pre-selected via global ANOVA statistical testing (p < 0.05), selecting
the most significant features. This is a **known limitation** — ANOVA was applied globally
rather than within each cross-validation fold. Document this in any reports.

---

## 7. Checksums

After downloading, verify file integrity using checksums. Generate with:

```powershell
# Windows (PowerShell)
Get-FileHash data/raw/GS-BRCA/Top/* -Algorithm SHA256 | Format-Table Hash, Path > data/checksums_brca.txt
Get-FileHash data/raw/GS-COAD/Top/* -Algorithm SHA256 | Format-Table Hash, Path > data/checksums_coad.txt
```

Reference checksum files (if generated): `data/checksums_brca.txt`, `data/checksums_coad.txt`

---

## 8. Data Redistribution Policy

- **Do NOT** redistribute the raw CSV files
- Instead, share this README with download instructions and checksums
- Original data is from TCGA and released under CC-BY-4.0 via MLOmics
- Always cite the MLOmics dataset (see Section 2)

---

## 9. Related Resources

| Resource | URL |
|---|---|
| MLOmics GitHub (benchmark repo) | https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark |
| MLOmics Paper | See Figshare page for associated publication |
| STRING Mapping File | `Cancer-Multi-Omics-Benchmark/Complementary Resources/Mapping File/` |
| TCGA Data Portal | https://portal.gdc.cancer.gov/ |

---

*Last updated: February 2026*
*Project: MLOmics — Latent-Fusion Multi-Omics Classifier for Cancer Subtype Prediction*
