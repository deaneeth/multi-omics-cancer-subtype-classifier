# Data Card — Demo Patient Datasets

*Synthetic single-patient prediction-ready CSVs for demonstrating the MLOmics Prediction tab.*

---

## What Are These Files?

These 2 CSV files are **ready-to-upload** prediction files. Unlike the lab conversion files (in `convertion/`), these do NOT need the Data Converter tab. You upload them directly into the **Prediction** tab.

Each file represents one fictional patient with a complete 4-modality multi-omics profile — all features already named, prefixed, and normalised in the exact format the models expect.

---

## The Two Patients

### 1. Sarah Mitchell — Breast Cancer (Luminal A)

| Field | Value |
|---|---|
| File | `demo_patient_sarah_mitchell_BRCA.csv` |
| Sample ID | PT-2026-BRCA-0047 |
| Age/Sex | 58 years old, Female |
| Diagnosis | Invasive ductal carcinoma, right breast |
| Clinical info | ER+, PR+, HER2-, Ki-67 12% (low), BRCA1/2 wild-type |
| Cancer type | GS-BRCA |
| Ground truth | Luminal A |
| File shape | 1 row x 15,367 columns (sample_id + 15,366 features) |

**Why Luminal A?** This is the most common "good prognosis" breast cancer subtype — hormone-receptor positive, slow-growing, typically treated with tamoxifen alone. Represents 19.7% of the BRCA dataset.

### 2. Robert Okonkwo — Colon Cancer (CMS2 Canonical)

| Field | Value |
|---|---|
| File | `demo_patient_robert_okonkwo_COAD.csv` |
| Sample ID | PT-2026-COAD-0062 |
| Age/Sex | 64 years old, Male |
| Diagnosis | Colon adenocarcinoma, ascending colon |
| Clinical info | Microsatellite stable (MSS), WNT/MYC pathway activation |
| Cancer type | GS-COAD |
| Ground truth | CMS2 (Canonical) |
| File shape | 1 row x 15,201 columns (sample_id + 15,200 features) |

**Why CMS2?** This is the "canonical" WNT-driven colon cancer subtype — the most common in clinical practice, typically microsatellite-stable, responds to standard chemotherapy (FOLFOX). Represents 18.5% of the COAD dataset.

---

## File Format

Both files share the same structure:

```csv
sample_id,mrna_C1orf210,mrna_GPBP1L1,...,mirna_hsa.let.7a.1,...,methy_WDR46,...,cnv_IVNS1ABP,...
PT-2026-BRCA-0047,0.4521,-1.2034,...,-0.631,...,2.534,...,0.649,...
```

- **Row 1:** Header with `sample_id` followed by all feature names with modality prefixes
- **Row 2:** The patient's values (z-score normalised, typically between -4.5 and +4.5)
- **Feature prefixes:** `mrna_`, `mirna_`, `methy_`, `cnv_` identify which modality each feature belongs to
- **Feature order:** mRNA (5000) → miRNA (366 BRCA / 200 COAD) → Methylation (5000) → CNV (5000)

---

## How These Were Generated

**Script:** `scripts/create_lab_patient_files.py` (Sarah and Robert were generated in an earlier session using the same methodology)

**Method:**
1. Loaded the training data for the target subtype class from the best CV fold
2. Computed per-feature mean and standard deviation for that class
3. Sampled one patient: `value = Normal(class_mean, class_std)` per feature
4. Clipped values to [-4.5, 4.5] to avoid physiologically implausible extremes
5. Assembled into a single-row DataFrame with all 4 modalities concatenated
6. Added `sample_id` as the first column
7. Saved as CSV ready for direct upload

**Seeds:** Different random seeds used for each patient to ensure distinct profiles.

---

## Expected Prediction Results

### Sarah Mitchell (BRCA, Luminal A)

| Model | Expected Prediction | Expected Confidence |
|---|---|---|
| XGBoost | Luminal A | >90% |
| Intermediate Fusion | Luminal A | >85% |
| Pathway-Aware Fusion | Luminal A | >85% |

### Robert Okonkwo (COAD, CMS2)

| Model | Expected Prediction | Expected Confidence |
|---|---|---|
| XGBoost | CMS2 (Canonical) | >80% |
| Intermediate Fusion | CMS2 (Canonical) | >75% |
| Pathway-Aware Fusion | CMS2 (Canonical) | >80% |

**Note:** COAD confidence is generally lower than BRCA because the COAD dataset is smaller (260 vs 671 patients) and has higher class imbalance.

---

## Difference From Lab Conversion Files

| Aspect | Demo Patients (this folder) | Lab Files (`convertion/`) |
|---|---|---|
| Upload destination | Prediction tab directly | Data Converter tab first |
| Number of files | 1 per patient | 4 per patient (one per modality) |
| Feature names | Prefixed (`mrna_ESR1`) | Raw (`ESR1`) |
| miRNA format | Training format (`hsa.mir.21`) | Lab format (`hsa-miR-21`) |
| Ready to predict? | Yes, immediately | No, needs conversion first |
| Purpose | Quick demo of prediction | Demo of full hospital-to-prediction workflow |

---

## Limitations

1. **Synthetic, not real** — generated from class statistics, not actual tumour biopsies
2. **No inter-feature correlations** — features sampled independently (real tumours have complex co-expression networks)
3. **Self-consistency test** — high confidence expected because data comes from the training distribution
4. **Not clinically validated** — these are academic demonstration files only

---

## How to Use

1. Open the MLOmics Streamlit app
2. Go to the **Prediction** tab
3. Select the correct Cancer Type (GS-BRCA for Sarah, GS-COAD for Robert)
4. Upload the CSV file
5. Try all 3 models and compare predictions

---

*Created: 2026-05-11 | Location: `app/test_datasets/demo/`*
