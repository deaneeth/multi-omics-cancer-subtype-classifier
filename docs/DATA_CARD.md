# MLOmics Dataset Card

> Format inspired by [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al., 2021).

---

## Dataset Identity

| Field | Value |
|---|---|
| Name | MLOmics Cancer Multi-Omics Database for Machine Learning |
| Version | 1.0 (as of download date) |
| Source | [Figshare — doi:10.6084/m9.figshare.28729127](https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127) |
| Licence | CC BY 4.0 (per Figshare record) |
| Upstream source | TCGA (The Cancer Genome Atlas) via GEO / GDC |

---

## Cohorts

| Cohort | Samples | Subtypes | Subtype labels |
|---|---|---|---|
| GS-BRCA (Breast) | ~671 | 5 | Basal-like (0), HER2-enriched (1), Luminal A (2), Luminal B (3), Normal-like (4) |
| GS-COAD (Colon) | ~260 | 4 | CMS1 (0), CMS2 (1), CMS3 (2), CMS4 (3) |

---

## Modalities

| Key | Description | Selected features |
|---|---|---|
| `mrna` | mRNA gene expression (RNA-seq, log-normalised) | Top 5,000 by variance |
| `mirna` | miRNA expression | 366 (BRCA), 200 (COAD) |
| `methy` | DNA methylation (beta values) | Top 5,000 by variance |
| `cnv` | Copy Number Variation (segment means) | Top 5,000 by variance |

Total features per sample: **15,366** (BRCA), **15,200** (COAD).

---

## File Format

Each CSV has **features as rows, samples as columns** (transposed from the ML convention).
All downstream code calls `df.T` after loading. See `src/data_loader.py:load_modality()`.

Label files (`*_label_num.csv`) contain integer class indices only, **no sample-ID column**.
Alignment is positional — row i of the label file corresponds to column i of the feature CSVs.
This is verified by `scripts/verify_label_alignment.py` and guarded by checksums in
`data/label_file_checksums.json`.

---

## Known Limitations and Biases

| Issue | Detail |
|---|---|
| Class imbalance | BRCA is heavily Luminal A (Basal-like and Normal-like are minority classes). Models use inverse-frequency class weights to partially compensate. |
| TCGA batch effects | TCGA data contains known plate/batch effects. No explicit batch correction is applied; z-score normalisation per modality partially mitigates this. |
| Positional label alignment | Label CSVs lack sample IDs. Alignment relies on consistent column ordering between mRNA and label files as produced by the MLOmics benchmark pipeline. |
| KEGG pathway coverage | Only 35.18% (BRCA) / 37.48% (COAD) of mRNA features map to a KEGG pathway; 62–65% of mRNA features are treated as "unmapped" by the PathwayAwareFusion encoder. |
| No independent test set | All reported metrics come from 5-fold CV. Performance on independent cohorts (GTEx, METABRIC) has not been evaluated. |
| Feature selection leakage risk | Top-N feature selection was performed on the full dataset by the upstream MLOmics benchmark, not fold-specifically. This constitutes a mild form of data leakage (acknowledged limitation). |

---

## Preprocessing Applied by This Project

1. Load raw CSVs, transpose to (samples × features).
2. Align sample IDs across modalities using `get_common_samples()`.
3. Median imputation for missing values — fitted on train fold only (`PerModalityImputer`).
4. z-score normalisation per modality — fitted on train fold only (`PerModalityScaler`).
5. Fold splits loaded from `data/cv_folds.json` (patient-level; fixed across all models).

---

## Citation

If you use this dataset, please cite the original MLOmics Figshare record and TCGA:

```
MLOmics: Cancer Multi-Omics Database for Machine Learning.
Figshare. doi:10.6084/m9.figshare.28729127
```

```
The Cancer Genome Atlas Research Network. (2013).
Comprehensive molecular portraits of human breast tumours. Nature, 490, 61–70.
```
