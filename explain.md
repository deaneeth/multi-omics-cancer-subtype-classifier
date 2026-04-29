● You're asking exactly the right questions. Let me answer them directly.

---

You're 100% correct on the diagnosis requirement.

The patient already knows they have breast cancer or colon cancer before this tool is used. This is a subtype classifier, not a cancer detector.
The input is processed genomic data from a tumor biopsy — not something a patient provides themselves. The realistic workflow is:

Oncologist orders tumor biopsy
→ Lab does RNA sequencing + methylation + CNV profiling
→ Bioinformatician processes raw data into the CSV format
→ Tool classifies the molecular subtype
→ Oncologist decides treatment

A patient who "only knows they have cancer somewhere" cannot use this. They need: (1) confirmed breast or colon cancer diagnosis, and (2) full
multi-omics profiling from their tumor — which costs thousands of dollars at a genomics lab and takes weeks. │

---

What is the actual purpose?

Molecular subtype determines treatment, not just the cancer type. Two breast cancer patients can have wildly different correct treatments:

┌───────────┬────────────────────────────┬───────────────────────────────────────────────┬────────────────────────────────┐
│ Patient │ Subtype the model predicts │ Wrong treatment │ Right treatment │
├───────────┼────────────────────────────┼───────────────────────────────────────────────┼────────────────────────────────┤
│ Patient A │ Luminal A │ Chemotherapy (overkill, unnecessary toxicity) │ Tamoxifen/hormone therapy only │
├───────────┼────────────────────────────┼───────────────────────────────────────────────┼────────────────────────────────┤
│ Patient B │ HER2-enriched │ Hormone therapy (won't work) │ Trastuzumab (Herceptin®) │
├───────────┼────────────────────────────┼───────────────────────────────────────────────┼────────────────────────────────┤
│ Patient C │ Basal-like │ Hormone therapy (won't work, ER-) │ Carboplatin + PARP inhibitors │
└───────────┴────────────────────────────┴───────────────────────────────────────────────┴────────────────────────────────┘

Standard clinical subtyping uses immunohistochemistry (IHC — testing ER/PR/HER2 protein markers), which misclassifies roughly 10–15% of patients. Multi-omics
profiling captures the full molecular picture and is more accurate. This tool automates that classification.

---

Best file for demonstrating to a non-technical person:

test_brca_single_HER2-enriched.csv — one real de-identified TCGA breast cancer patient, correctly classified at 99.7% confidence. The story you tell:

▎ "This is the tumor genomic profile of a breast cancer patient — 15,366 measurements across four molecular layers of their tumor. The model says: HER2-enriched,
▎ 99.7% confident. That tells the oncologist this patient needs trastuzumab (Herceptin), a targeted antibody therapy. Without the subtype, they might receive hormone
▎ therapy instead, which would do nothing because this tumor is not driven by estrogen."

Then switch to test_brca_batch.csv (20 patients) to show it working at scale — the model processes all 20 in seconds, each with a different prediction and confidence
score.

---

What this project actually is (for your BSc presentation)

It is a clinical decision support research tool. Not a consumer app. The audience is oncologists and computational biologists. The research question it answers is:
"Can machine learning on multi-omics data match or outperform standard IHC-based subtyping, and which molecular signals drive each subtype?" — and your results
(F1=0.808 for BRCA, pathway enrichment matching known biology) say yes.
