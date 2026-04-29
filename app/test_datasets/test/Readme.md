app/test_datasets/test/ — real TCGA patients (from create_test_datasets.py)

  ┌────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
  │              File              │                                    What it is                                    │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ test_brca_batch.csv            │ 20 real TCGA val-set patients, 4 per subtype, sample_id column first             │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ test_brca_single_<Subtype>.csv │ Single highest-confidence TCGA patient per subtype                               │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ test_brca_metadata.csv         │ True labels, XGBoost prediction, confidence, clinical description — don't upload │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ (same for COAD)                │                                                                                  │
  └────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘

REAL TCGA PATIENTS (use these for demo/presentation)
  test_brca_batch.csv          20 rows — real TCGA patients, 4 per BRCA subtype
  test_brca_single_*.csv       1 row each — highest-confidence patient per subtype (5 files)
  test_brca_metadata.csv       reference: true label vs prediction — don't upload

  test_coad_batch.csv          12 rows — real TCGA patients
  test_coad_single_*.csv       1 row each (3 files — CMS4 has only 4 training examples total)
  test_coad_metadata.csv       reference only