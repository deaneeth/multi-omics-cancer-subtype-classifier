# Data Card — Lab Patient Conversion Files

*Simulated hospital genomics lab export for testing the MLOmics Data Converter tab.*

---

## What Are These Files?

These 4 CSV files simulate what a hospital genomics lab would send after sequencing a single patient's tumour. Each file contains one type of biological measurement (one "modality") for one patient. They are designed to be uploaded into the **Data Converter** tab of the MLOmics Streamlit app, which combines them into a single prediction-ready CSV.

---

## The Patient

| Field | Value |
|---|---|
| Name | Amara Nwosu |
| Sample ID | PT-BRCA-LAB-2026-038F |
| Age/Sex | 38 years old, Female |
| Diagnosis | Invasive ductal carcinoma, left breast |
| Clinical info | ER-, PR-, HER2- (Triple-Negative), Stage IIB, Ki-67 74%, BRCA1 mutation |
| Cancer type | GS-BRCA (Breast Cancer) |
| Ground truth subtype | Basal-like (do not reveal before prediction) |

---

## The 4 Files

| File | Modality | Features | Size | Format |
|---|---|---|---|---|
| `amara_nwosu_mrna.csv` | mRNA gene expression | 5,000 genes | 134 KB | gene_symbol index, 1 patient column |
| `amara_nwosu_mirna.csv` | miRNA expression | 200 miRNAs | 6.6 KB | miRBase dash-format IDs (e.g. hsa-miR-21) |
| `amara_nwosu_methylation.csv` | DNA methylation | 5,000 probes | 133 KB | gene_symbol index, 1 patient column |
| `amara_nwosu_cnv.csv` | Copy number variation | 5,000 genes | 134 KB | gene_symbol index, 1 patient column |

### File Format (all 4 files)

```
gene_symbol,PT-BRCA-LAB-2026-038F
C1orf210,1.6304
GPBP1L1,-2.7024
DEF8,0.776
...
```

- First column = feature name (gene symbol or miRNA ID)
- Second column = the patient's measured value (z-score normalised)
- Features as rows, patient as column (standard lab export orientation)

### miRNA Naming Convention

The miRNA file uses standard miRBase dash format:
- `hsa-let-7a-1` (lethal family)
- `hsa-miR-21` (microRNA family, capital R)
- `hsa-miR-130b` (with strand suffix when applicable)

The Data Converter automatically normalises these to the training format (dots, lowercase): `hsa.let.7a.1`, `hsa.mir.21`, `hsa.mir.130b`.

---

## How These Were Generated

**Script:** `scripts/create_lab_patient_files.py`

**Method:**
1. Loaded training data for the Basal-like class (class 0) from the best CV fold
2. Computed per-feature mean and standard deviation for Basal-like patients
3. Sampled one patient: `value = Normal(class_mean, class_std)` per feature, clipped to [-4.5, 4.5]
4. Split the 15,366-feature vector back into 4 modality-specific files
5. Stripped modality prefixes (e.g. `mrna_ESR1` becomes just `ESR1`)
6. Converted miRNA names from training format (dots) to lab format (dashes)
7. Excluded 166 `nan_*` miRNA features (imputation artifacts, not real probes)

**Important:** Values are z-score normalised (mean ~0, std ~1). This matches the training pipeline's output AFTER normalisation. Real hospital data would need normalisation before upload.

---

## Expected Converter Results

| Modality | Coverage | Explanation |
|---|---|---|
| mRNA | 100% (5000/5000) | All gene symbols match exactly |
| miRNA | 55% (200/366) | 200 real probes match; 166 `nan_*` features filled with 0 |
| Methylation | 100% (5000/5000) | All gene symbols match exactly |
| CNV | 100% (5000/5000) | All gene symbols match exactly |

**Output:** 1 row x 15,366 features + sample_id column = 15,367 columns total.

---

## Expected Prediction Results

All 3 models should predict **Basal-like** with high confidence (>95%) because:
- The data was sampled from the Basal-like class distribution
- Feature values are already in the expected normalised space
- Full 4-modality coverage provides maximum signal

**Verified results:**
- XGBoost: Basal-like, 99.8% confidence
- Intermediate Fusion: Basal-like, 95.5% confidence
- Pathway-Aware Fusion: Basal-like, 98.9% confidence

---

## Limitations

1. **Not real patient data** — generated from statistical distributions, not an actual tumour biopsy
2. **No inter-feature correlations** — each feature sampled independently (real biology has complex co-expression patterns)
3. **No batch effects** — real hospital data from a different sequencing platform would have systematic biases
4. **Pre-normalised** — real lab data requires log2 transformation + z-score normalisation before upload
5. **Self-consistency test only** — high confidence is expected because the data comes from the same distribution the model was trained on

---

## How to Use

1. Open the MLOmics Streamlit app (`streamlit run app/streamlit_app.py`)
2. Go to the **Data Converter** tab
3. Select Cancer Type: **GS-BRCA**
4. Enter Sample ID: **PT-BRCA-LAB-2026-038F** (or any ID you want)
5. Upload all 4 files in the correct modality slots
6. Click **Convert & Prepare File**
7. Review the coverage report (should match table above)
8. Download the prediction-ready CSV
9. Switch to the **Prediction** tab
10. Upload the downloaded CSV and run predictions with all 3 models

---

## Generation Seed & Reproducibility

- Random seed: 13
- Generated from: Best fold (fold 3) Basal-like training patients (n=283)
- Script is deterministic — re-running produces identical files

---

*Created: 2026-05-11 | Script: `scripts/create_lab_patient_files.py`*
