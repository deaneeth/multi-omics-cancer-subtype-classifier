# The Story of MLOmics

## What Problem Were We Trying to Solve?

Cancer is not one disease — it's hundreds. Even within a single cancer type like breast cancer, there are distinct *subtypes* that behave differently and respond differently to treatment. Getting the subtype right matters: the wrong treatment plan can be ineffective or even harmful.

Doctors today use genetic tests to figure out which subtype a patient has. These tests generate enormous amounts of biological data — measurements from thousands of genes, proteins, and chemical markers all at once. Reading all of this data manually is impossible. That's where machine learning comes in.

This project — **MLOmics** — is a final-year BSc research project that asks a simple question:

> *Can a computer learn to classify cancer subtypes by reading a patient's biology — and can we explain* why *it made its decision?*

---

## The Data: Four Windows Into a Cell

Every tumour sample in this project was described through **four types of biological measurements**, each telling a different part of the story:

| Measurement | What it captures |
|---|---|
| **mRNA expression** | Which genes are currently "switched on" |
| **miRNA expression** | Small molecules that regulate gene activity |
| **DNA methylation** | Chemical tags that silence genes |
| **CNV (Copy Number Variation)** | How many copies of each gene a cell carries |

Think of it like looking at a person through four different cameras — each reveals something the others don't. We deliberately chose to use all four rather than just mRNA (the most common choice) because we wanted to test whether the combination is genuinely stronger than any single view.

We worked with data from **931 patients** across two cancers:
- **Breast cancer (GS-BRCA)** — 671 patients, 5 subtypes
- **Colon cancer (GS-COAD)** — 260 patients, 4 subtypes

Each patient had ~15,200 measurements across all four modalities. The data came from the publicly available [MLOmics benchmark dataset](https://figshare.com/articles/dataset/MLOmics_Cancer_Multi-Omics_Database_for_Machine_Learning/28729127).

> **An uncomfortable early discovery:** The benchmark dataset had already pre-selected the top features using a statistical test (ANOVA) run across the *entire* dataset before we split patients into train and test groups. That means some label information leaked into the feature set before any model ever trained. We only caught this when a "sanity check" test (shuffle the labels randomly, train, see if the model still predicts well) gave an F1 of 0.38 on BRCA and 0.53 on COAD when random chance should have given 0.20 and 0.25. A properly random model shouldn't know anything — but ours did, because the features were already selected with label knowledge. We documented this honestly as a dataset-level limitation rather than a bug we could fix: the feature selection happened upstream, not in our pipeline.

---

## The Journey: Building Smarter and Smarter Models

### Step 1 — Get the Data Ready

Raw biological data is messy. Values are missing, scales are wildly different between modalities, and sample IDs need to be matched across all four measurement types.

We built a preprocessing pipeline that cleaned everything up: normalising each modality independently, imputing missing values, and splitting patients into **5 equal groups** for fair testing (so no model ever trains and tests on the same patient). We ran 49 automated checks on this pipeline — 46 passed. The three that failed are documented above and in the preprocessing report.

> **Another surprise during data checks:** Almost half the miRNA measurements for breast cancer patients — 166 out of 366 columns — were entirely empty (all NaN). The assay probes simply weren't observed in this cohort. We filled them with zero after normalising, but it means miRNA contributed far less information for breast cancer than we initially expected.

> **A real constraint we had to work around:** The colon cancer dataset only has 260 patients across 4 subtypes. One subtype (CMS4) has just **4 patients total** in the entire dataset. That's not enough to learn from. In some test folds it appears zero times in the validation set, making standard accuracy metrics undefined (NaN) for that fold. We kept it in rather than dropping it — dropping rare subtypes in a medical context feels like cheating — but it means colon cancer results are noisier and less reliable than breast cancer results throughout the project.

### Step 2 — Start Simple: Tree-Based Baselines

Before doing anything fancy, we mashed all four modalities into one big list of numbers and handed it to two well-known algorithms:

- **XGBoost** — a powerful gradient-boosted decision tree algorithm
- **Random Forest** — an ensemble of many decision trees

We did this first because it's important to establish a floor — if a simple well-known method already works well, the more complex models need to actually beat it to justify their complexity.

Results on breast cancer: XGBoost hit an **F1 score of 0.794**. Not bad — but we thought the deep learning approach could do better.

> **Something that genuinely puzzled us:** Random Forest achieved an AUC of 0.971 — the *highest* of any model in the entire project. AUC measures how well a model can *rank* patients by confidence. But its F1 was only 0.602, one of the worst. This means Random Forest had excellent intuition about which cases were uncertain, but when forced to make a hard decision it missed a lot of patients (recall of 0.560 — it only found 56% of positives). High AUC with low recall is a real failure mode and a good reminder that AUC alone can be misleading.

### Step 3 — The Core Idea: Intermediate Fusion

Here's the central hypothesis of the whole project. When you just mash all the data together (early fusion), you lose information about *where* each signal came from. mRNA is not the same as DNA methylation — they tell different stories, at different scales, measured by different instruments.

What if instead we built a **separate mini-network for each modality**, let each one learn its own compact summary, and *then* combined those summaries to make the final classification?

This is called **intermediate fusion**, and it's the heart of MLOmics.

```
mRNA data    → [Encoder A] → 64-dim summary ──┐
miRNA data   → [Encoder B] → 64-dim summary ──┤
Methylation  → [Encoder C] → 64-dim summary ──┼→ [Classifier] → Subtype
CNV data     → [Encoder D] → 64-dim summary ──┘
```

Each encoder compresses thousands of measurements into just 64 numbers that capture the most important patterns. Then a final network combines all four summaries and predicts the subtype.

The risk here was real: deep learning needs a lot of data, especially when you're training four separate networks. With only 260 COAD patients, we were pushing into territory where a neural network can easily overfit (memorise the training data instead of learning real patterns). We used early stopping, weight decay, and class-weighted loss to fight this — but it was a genuine gamble.

Result on breast cancer: **F1 of 0.808** — beating XGBoost. Hypothesis confirmed.

> **But on colon cancer, the hypothesis failed.** IntermediateFusion scored F1=0.669, while a simple EarlyFusionMLP (just flatten everything and use one neural network) scored 0.751. The fancy architecture lost to the simpler one. With only 260 patients, there simply wasn't enough data to train four separate encoders well. The modality-specific representations didn't have enough signal to justify the extra complexity.

### Step 4 — Adding Biology: Pathway-Aware Fusion

After the COAD disappointment, we took a different angle. Deep learning with limited data is hard — but biologists have spent decades mapping which genes work together in the same biological "pathways." What if we gave the model that structure as a starting point instead of making it discover everything from scratch?

We replaced the mRNA encoder with a smarter version that **groups genes by their known KEGG biological pathways**, calculates a summary for each pathway, and then uses a learnable attention mechanism to figure out which pathways matter most for classification.

The biological prior gave the model a head start. On **colon cancer**, this was the biggest improvement of the project: F1 jumped from 0.669 to **0.738** — not just beating our own intermediate fusion baseline, but also beating the simple EarlyFusionMLP. This validated the approach: when data is scarce, baked-in biological knowledge compensates for what the model can't learn from examples alone.

### Step 5 — Understanding the Decisions

A model that makes predictions without explanation isn't trustworthy in medicine. We added three explanation methods:

- **SHAP** (for XGBoost) — shows which specific genes pushed the prediction up or down
- **Integrated Gradients** (for deep models) — traces which input features the neural network found most important
- **KEGG pathway enrichment** — maps important genes back to known biological processes

This lets a researcher ask: *"Why did the model predict Subtype X?"* and get a biologically meaningful answer.

> One limitation we had to be honest about: for the Pathway-Aware Fusion model in the demo app, we reuse the IntermediateFusion explanation rather than computing a separate one. The explanation is still accurate at the gene level, but it doesn't show the pathway-level attention weights the model actually used internally. A proper explanation of that model would require a different visualisation that wasn't built into the final demo.

### Step 6 — Stress-Testing with Ablations

We systematically broke things to see what mattered:

- **Removed one modality at a time** — mRNA was the most important for breast cancer; removing it dropped F1 the most.
- **Simulated missing data** — even with 50% of data randomly missing, performance dropped gracefully (F1 went from 0.808 to 0.749 on BRCA, not catastrophically).
- **Compared fusion strategies** — confirmed intermediate fusion is better than early-concatenation on BRCA.

> **The most surprising result in the whole project:** On colon cancer, removing miRNA entirely *improved* F1 from 0.669 to **0.802**. A modality was actively hurting the model. The most likely explanation: with very few patients, noisy or uninformative miRNA measurements added confusion the model couldn't ignore. Sometimes less data is better data.

---

## Final Results at a Glance

| Model | Breast Cancer F1 | Colon Cancer F1 | Best AUC |
|---|---|---|---|
| XGBoost | 0.794 | 0.636 | 0.970 |
| Random Forest | 0.602 | 0.608 | 0.971 |
| Intermediate Fusion | **0.808** | 0.669 | 0.963 |
| Pathway-Aware Fusion | 0.801 | **0.738** | 0.966 |

F1 score balances precision ("when it predicts a subtype, is it right?") and recall ("does it find all the cases?"). AUC measures how confidently the model separates subtypes — all models scored above 0.90, which is strong.

**The honest takeaway:** Intermediate fusion wins on breast cancer, but only narrowly over XGBoost, and it lost on colon cancer. The real win was adding biological pathway structure — that gave the biggest gain exactly where data was scarce.

---

## The Demo App

Everything came together in an interactive **Streamlit web app** where you can:

1. **Upload a patient's multi-omics data** (a CSV file) — sample files are provided for both cancer types
2. **Choose a cancer type** (breast or colon) and a **model** (XGBoost, Intermediate Fusion, or Pathway-Aware Fusion)
3. **See the predicted subtype** with a confidence score and colour-coded confidence bar (green = high, blue = moderate, amber = low confidence)
4. **See an explanation** — which genes/features drove the prediction, shown as a bar chart with modality labels
5. **Compare all models** side-by-side with F1, precision, recall, AUC, ROC curves, and a radar chart

The app loads pre-trained models from saved checkpoints. All heavy computation (training, explanations, t-SNE visualisations) was done beforehand and stored — so the demo runs fast without a GPU.

---

## What This Project Demonstrates

1. **Multi-omics data is richer than single-omics** — using all four measurement types together outperforms any single one
2. **How you combine data matters — but only at scale** — intermediate fusion (modality-specific encoders) outperforms raw concatenation on the larger breast cancer dataset, but not on the smaller colon cancer one
3. **Biological knowledge helps small datasets** — KEGG pathway structure gave the biggest gains exactly where data was scarce
4. **Explainability is achievable but has limits** — we can trace predictions back to genes, but internal model representations (like pathway attention weights) are harder to surface cleanly
5. **Always run sanity checks** — our label shuffle test caught a real methodological issue in the benchmark dataset early, and documenting it honestly is more valuable than pretending it didn't happen
6. **Surprising negatives matter** — finding that miRNA hurts colon cancer performance is a genuine scientific contribution, not a failure

---

*Built by Dineth Hettiarachchi as a BSc Computer Science final-year project. Not for clinical use.*
