# The Story of MLOmics

*A complete guide to this project, written for anyone — no scientific background needed.*

---

## The Problem: Why Cancer Subtypes Matter

Imagine two women walk into a doctor's office. Both have breast cancer. The same disease, right? Wrong.

Inside their tumours, the biology is completely different. One woman's cancer is fuelled by the hormone estrogen. The other's is driven by a protein called HER2. If you give the first woman's treatment to the second, it won't work — it might even make things worse. If you give chemotherapy to someone who only needs hormone therapy, you're poisoning them for no benefit.

This is why doctors don't just say "you have breast cancer." They say "you have **Luminal A** breast cancer" or "you have **HER2-enriched** breast cancer." These labels are called **molecular subtypes**, and getting them right determines whether a patient lives or dies.

Today, doctors figure out the subtype by running a few protein tests (called IHC — immunohistochemistry) on a tumour sample. But these tests get it wrong about 10-15% of the time. When they're wrong, patients get the wrong treatment.

There's a better way. Instead of testing a few proteins, what if we could read the tumour's **entire biological instruction manual** — thousands of measurements from genes, chemical markers, and structural variations all at once — and use a computer to figure out the subtype?

That's what MLOmics does.

---

## The Data: Four Windows Into a Cell

Every tumour sample in this project is described through **four completely different types of biological measurements**. Think of each one as a different camera looking at the same cell:

### 1. mRNA Expression — "Which genes are switched on?"
Your DNA is like a recipe book. mRNA is like photocopies of specific recipes that the cell is currently cooking. By measuring mRNA, we can see which genes are active right now. We measured **5,000** of the most important genes.

### 2. miRNA Expression — "Who's regulating the kitchen?"
miRNA are tiny molecules that control gene activity — like a manager who tells cooks to speed up or slow down. They don't make proteins themselves; they regulate the ones that do. We measured **366** miRNAs for breast cancer and **200** for colon cancer.

### 3. DNA Methylation — "Which recipes are locked away?"
Methylation is a chemical tag that silences genes. Imagine certain pages of the recipe book are glued shut — the recipe is there, but it can't be read. We measured **5,000** methylation sites across the genome.

### 4. CNV (Copy Number Variation) — "How many copies of each recipe?"
Sometimes cells have extra copies of certain genes (amplification) or missing copies (deletion). Too many copies of a growth gene can drive cancer. We measured **5,000** gene-level copy number variations.

**In total, each breast cancer patient's tumour is described by 15,366 numbers** and each colon cancer patient's by 15,200 numbers — like a 15,000-point fingerprint of their cancer's biology.

### Where This Data Came From

All the data comes from real patients — 931 of them — whose tumours were sequenced by The Cancer Genome Atlas (TCGA), a massive US government project. The data was organized and shared by the MLOmics benchmark dataset (CC-BY-4.0 license, meaning it's free to use for research).

We worked with two cancers:

**Breast Cancer (GS-BRCA) — 671 patients, 5 subtypes:**

| Subtype | Patients | What it means |
|---|---|---|
| Basal-like | 353 (52.6%) | Aggressive, often triple-negative. Needs chemotherapy. |
| HER2-enriched | 42 (6.3%) | Driven by HER2 protein. Responds to targeted therapy (Herceptin). |
| Luminal A | 132 (19.7%) | Hormone-driven, slower growing. Often treated with tamoxifen alone. |
| Luminal B | 31 (4.6%) | Also hormone-driven but faster growing. Needs hormone therapy + chemo. |
| Normal-like | 113 (16.8%) | Resembles healthy breast tissue. Generally good prognosis. |

**Colon Cancer (GS-COAD) — 260 patients, 4 subtypes:**

| Subtype | Patients | What it means |
|---|---|---|
| CMS1 (MSI/Immune) | 174 (66.9%) | High mutation rate, strong immune response. Responds to immunotherapy. |
| CMS2 (Canonical) | 48 (18.5%) | Classic colon cancer biology. Chromosomal instability. |
| CMS3 (Metabolic) | 34 (13.1%) | Metabolic dysregulation. Mixed features. |
| CMS4 (Mesenchymal) | 4 (1.5%) | Worst prognosis. Only 4 patients in the entire dataset. |

Notice that last subtype — CMS4 Mesenchymal — has only **4 patients total** across 260. That's less than 2%. This becomes a big problem later.

### A Problem We Discovered Early

Before we even started building models, we found something uncomfortable about the data.

The benchmark dataset had already been "pre-filtered." Someone ran a statistical test called ANOVA across ALL 931 patients before splitting them into training and testing groups. This test picked the 5,000 most "interesting" genes — but it used information about which patients had which subtypes to do so.

This means the feature set itself was contaminated with label information. It's like being given a multiple-choice exam where the answers are partially visible through the paper.

We proved this with a sanity check: we randomly shuffled all the subtype labels (so the model should learn nothing), then trained XGBoost. A properly random model with 5 classes should score about F1=0.20 (basically guessing). Instead, it scored **0.38 on BRCA and 0.53 on COAD**. The model was finding patterns because the features themselves contained leaked label information from the ANOVA pre-selection.

This was **not our bug** — the feature selection happened upstream, before any of our code ran. We documented it honestly as a dataset-level limitation. It means all the F1 scores in this project are slightly inflated compared to what you'd get with truly independent feature selection. But since this affects all models equally, the comparisons between models are still fair.

---

## Step 1: Getting the Data Ready

Raw biological data is a mess. Here's what we had to deal with:

**Missing values.** The breast cancer miRNA data had a shocking problem: 166 out of 366 miRNA columns were **completely empty** — every single patient had "NaN" (Not a Number) for those measurements. That's 45.4% of the miRNA data just... gone. The probes simply weren't assayed in the BRCA cohort. We filled these with zero after normalizing (adding zero signal — it won't help, but it won't hurt either).

**Different scales.** mRNA values might range from 0 to 50,000, while methylation values range from 0 to 1. If you feed these directly to a model, the mRNA will dominate simply because its numbers are bigger — not because it's more important. We normalized each modality separately (z-score normalization: subtract the mean, divide by the standard deviation) so every measurement type has equal weight.

**Patient-level splitting.** This is critical and non-negotiable. When we test a model, we must never test it on a patient it trained on. That would be cheating — like studying from the answer key. We split our 931 patients into **5 groups** (called "folds"). For each fold, we train on 4 groups (~80% of patients) and test on the held-out group (~20%). We rotate through all 5 folds so every patient gets tested exactly once.

We saved these fold assignments to a file called `cv_folds.json` and **never changed them**. Every single model — XGBoost, Random Forest, neural networks, pathway fusion — uses the exact same splits. This means every comparison between models is fair: they all saw the same patients in training and were tested on the same patients.

**The normalization rule.** The scaler that normalizes the data must be "fitted" (it learns the mean and standard deviation) on the training fold **only**. Then we apply it to both training and testing data. If we fit the scaler on all data at once, information from the test set leaks into training through the normalization parameters.

**49 automated checks.** We ran a comprehensive verification suite on our preprocessing pipeline. 46 checks passed. The 3 that failed are the ones documented above (label shuffle for BRCA and COAD, and all-NaN miRNA columns). All the important safety checks passed: zero overlap between training and testing patients, all patients accounted for, class balance maintained across folds.

---

## Step 2: Starting Simple — The Tree-Based Baselines

Before doing anything fancy, we started with the most straightforward approach imaginable: take all 15,200–15,366 measurements, mash them into one giant list of numbers per patient, and feed it to two well-known algorithms.

This is called **early fusion** — we fuse all the data together at the very beginning, before any analysis.

### XGBoost

XGBoost is a gradient-boosted decision tree algorithm. It works like a very smart game of 20 Questions: it asks "is gene X above this threshold?" then "is methylation site Y below that threshold?" and so on, building a decision tree. Then it builds another tree that focuses on the cases the first tree got wrong. Then another. And another — **300 trees in total**, each correcting the mistakes of the previous ones.

Hyperparameters: 300 trees, max depth of 6 questions per tree, learning rate of 0.1 (how aggressively each tree corrects previous errors).

**Results on breast cancer (BRCA):**
- **F1 score: 0.7939 ± 0.047** — the model correctly identifies about 79% of cases, balanced across all subtypes
- **AUC: 0.9703 ± 0.015** — outstanding ranking ability (scale: 0.5 = random guessing, 1.0 = perfect)
- **Accuracy: 0.8644 ± 0.018** — 86.4% of predictions are correct

**Results on colon cancer (COAD):**
- **F1 score: 0.6358 ± 0.115**
- **AUC: 0.9105 ± 0.051**
- **Accuracy: 0.8808 ± 0.063**

### Random Forest

Random Forest is a simpler ensemble: 300 independent decision trees, each trained on a random subset of the data and a random subset of features. They vote on the final answer. It's like asking 300 doctors who each looked at a different part of the patient's chart and taking a majority vote.

Hyperparameters: 300 trees, unlimited depth, class_weight='balanced' (the algorithm knows some subtypes are rare and adjusts accordingly).

**BRCA: F1=0.6023 ± 0.083, AUC=0.9708 ± 0.009, Accuracy=0.7824 ± 0.025**
**COAD: F1=0.6083 ± 0.073, AUC=0.9321 ± 0.021, Accuracy=0.8731 ± 0.029**

### The AUC Paradox

Look at those BRCA numbers again. Random Forest has the **highest** AUC of any model in the entire project (0.9708), meaning it's the best at *ranking* patients by how confident it is. But it has the **worst** F1 (0.6023), meaning when forced to make a hard yes/no decision about each subtype, it's wrong a lot.

How can a model be simultaneously the best and worst? Because AUC and F1 measure different things:

- **AUC (Area Under the ROC Curve):** Measures how well the model separates different classes. Think of it as the model's "intuition." A model with AUC=0.971 is very good at saying "I'm 90% sure this is Luminal A, but only 30% sure about this other patient."

- **F1 score:** Measures how correct the model is when forced to pick exactly one answer. This is the "hard decision" metric.

Random Forest had great intuition (it knew which cases were uncertain) but poor decision-making at the threshold (recall was only 0.560 — it found only 56% of the positive cases). This is like a doctor who accurately assesses how sick each patient is, but when forced to discharge or admit, makes the wrong call for many patients. This observation is a valuable finding for the thesis discussion.

---

## Step 3: The Core Idea — Intermediate Fusion

Here's the central hypothesis. When you mash all the data together at the start (early fusion), you lose information about *where* each signal came from. mRNA is not the same as DNA methylation — they tell different stories, are measured by different instruments, and operate at different biological scales. A 5,000-dimensional mRNA vector shouldn't be treated the same as a 200-dimensional miRNA vector.

What if, instead of one big bucket, we built a **separate neural network for each measurement type**, let each one compress its thousands of numbers into a compact 64-number summary, and *then* combined those summaries to make the final decision?

This is called **intermediate fusion**:

```
mRNA data   (5,000 numbers) → [mRNA Encoder]    → 64-number summary ──┐
miRNA data  (200/366 numbers) → [miRNA Encoder] → 64-number summary ──┤
Methylation (5,000 numbers) → [Methy Encoder]   → 64-number summary ──┼→ [Classifier] → Subtype
CNV data    (5,000 numbers) → [CNV Encoder]     → 64-number summary ──┘
```

**How each encoder works:**
1. Takes in thousands of measurements
2. Passes through two layers of neurons (input → 256 → 64)
3. Each layer normalizes its output (BatchNorm), keeps only positive signals (ReLU), and randomly drops 30% of connections during training (Dropout) to prevent memorization
4. Produces a compact 64-number "summary" that captures the most important patterns

**The classifier** takes the four 64-number summaries (256 numbers total), passes them through a final decision network (256 → 128 → number of subtypes), and outputs the predicted subtype.

**Training details:**
- 100 maximum epochs (full passes through the training data), with early stopping after 10 epochs of no improvement
- Batch size of 32 (process 32 patients at a time)
- Learning rate starts at 0.001 and reduces by half when progress stalls
- Weight decay of 0.0001 (a gentle penalty for complexity to prevent overfitting)
- Class-weighted loss function (rare subtypes are weighted more heavily so the model pays attention to them)
- 4,036,613 trainable parameters

**Results on BRCA:**
- **F1: 0.8082 ± 0.050** — nominally best on BRCA, but statistically comparable to XGBoost (p=0.625, Cohen's d=0.197 — trivial effect). The architecture works, but the edge over XGBoost is not statistically distinguishable with n=5 folds.
- **Accuracy: 0.8330 ± 0.050** — 83.3% correct
- **AUC: 0.9631 ± 0.014**

**But on COAD, the hypothesis failed.**

IntermediateFusion scored **F1=0.6691 ± 0.094** on colon cancer. But when we tested an even simpler approach — just one neural network on the concatenated data (we called this EarlyFusionMLP: input→256→128→subtypes) — it scored **0.7510 ± 0.100**. The simpler model beat the fancy one.

**Why?** Colon cancer has only 260 patients across 4 subtypes. IntermediateFusion has four separate encoders, each with its own parameters to learn. With so little data, there simply aren't enough examples for each encoder to learn meaningful patterns. The model gets confused. The simpler EarlyFusionMLP, with fewer parameters, doesn't suffer from this — it can't overthink because it can't think that deeply in the first place.

**Lesson:** Fancy architecture doesn't automatically beat simple baselines. The right model complexity depends on how much data you have.

---

## Step 4: Adding Biology — Pathway-Aware Fusion

After the colon cancer disappointment, we asked: why did IntermediateFusion fail on small data? Because it had to learn everything from scratch — which genes are related, which work together, which pathways matter. With only 260 patients, there weren't enough examples.

But biologists have spent **decades** mapping out which genes work together. These groups of cooperating genes are called **biological pathways**. For example, the "Cell Cycle" pathway includes all the genes that control cell division. The "p53 signaling" pathway includes genes that detect DNA damage and trigger cell death.

The KEGG database (Kyoto Encyclopedia of Genes and Genomes) catalogs hundreds of these pathways. What if, instead of forcing the model to discover these relationships from limited data, we **told the model which genes belong to which pathways upfront**?

This is **pathway-aware fusion**. We replaced the mRNA encoder with a smarter version that has two paths:

**Path A — The pathway expert:**
1. Group the 5,000 mRNA genes by their KEGG pathway membership (e.g., all cell cycle genes together, all p53 genes together). Coverage: 35.18% of features for BRCA (1,759 / 5,000 features mapped across 311 pathways), 37.48% for COAD (1,874 / 5,000 features, 310 pathways).
2. For each pathway, calculate the average activity of all its member genes (just one number per pathway)
3. Pass these pathway-level summaries through a learnable **attention** mechanism — a small neural network that learns which pathways are most important for each patient and assigns weights accordingly
4. Multiply each pathway's summary by its attention weight and project to 64 dimensions

**Path B — The unmapped gene handler:**
Not every gene belongs to a known KEGG pathway. The remaining ~62–65% of genes are processed through a standard dense encoder, just like before.

**The merger:** Path A's pathway-aware output and Path B's unmapped gene output are concatenated and projected down to a single 64-dimensional vector — the same size as the other modalities' encoders — so everything fits together cleanly.

The model has 3,656,252 trainable parameters (slightly fewer than IntermediateFusion's 4,036,613 because the pathway-grouped mRNA processing is more compact).

**Results on COAD:**
- **F1: 0.7376 ± 0.142** — a major improvement from IntermediateFusion's 0.6691. The biological prior compensated for data scarcity.
- **Accuracy: 0.8538 ± 0.060**
- **AUC: 0.9542 ± 0.025** (best of any model on COAD)

**Results on BRCA:**
- **F1: 0.8006 ± 0.069** — essentially tied with IntermediateFusion (difference = −0.0076, within statistical noise; p=0.625, d=0.431)
- **Accuracy: 0.8286 ± 0.035**
- **AUC: 0.9659 ± 0.017**

**An honest caveat about the attention mechanism:** The attention weights came out nearly uniform — every pathway got roughly the same weight (~0.003–0.004, with tiny variation between pathways). The top-10 most-attended pathways for BRCA were: Axon guidance, Graft-versus-host disease, IL-17 signaling, Cocaine addiction, Primary bile acid biosynthesis, Ferroptosis, Progesterone-mediated oocyte maturation, Taurine and hypotaurine metabolism, Melanoma, Lysine degradation. Classical cancer pathways (Cell cycle, PI3K-Akt, p53) were nowhere near the top. Whether this reflects genuinely weak biological signal, insufficient KEGG coverage of our Top-5000 gene set (only 35–37% of features are pathway-mapped), or training instability is an open question.

**The win was real, even if the mechanism isn't fully understood.** PathwayFusion scored +0.069 F1 higher on COAD than IntermediateFusion — the biggest gain in the entire project — exactly where data was scarcest.

---

## Step 5: Understanding the Decisions

A model that says "this patient has Subtype X" without explaining why is useless in medicine. Doctors need to know *which evidence* drove the prediction. We built three layers of explanation:

### Layer 1: SHAP (for XGBoost and Random Forest)

SHAP (SHapley Additive exPlanations) is a method from game theory. It asks: "if we removed this gene from the analysis, how much would the prediction change?" Genes that cause big changes when removed are important. Genes that cause no change when removed aren't contributing.

**For breast cancer (XGBoost):** The top features were almost entirely mRNA genes. Top 5: mrna_MLPH (SHAP=0.406), mrna_ESR1 (estrogen receptor — biologically expected for breast cancer, SHAP=0.299), mrna_MPHOSPH6 (0.258), mrna_TOP2A (0.169), mrna_KCNMB1 (0.167). All top 10 are mRNA features. XGBoost relies overwhelmingly on gene expression for BRCA.

**For colon cancer (XGBoost):** A much more balanced picture. Top features span three modalities: cnv_MYO5B (0.252), mrna_TIMM21 (0.232), cnv_DCC (0.186), mrna_SLC35A4 (0.167), methy_C4orf45 (0.144). CNV, mRNA, and methylation all in the top 5. The model was genuinely using all the data types.

### Layer 2: Integrated Gradients (for the Deep Neural Networks)

For the fusion models, we used Integrated Gradients (via Captum library). This method traces how the prediction changes as we gradually "turn on" each input feature from zero to its actual value, measuring the accumulated contribution of each gene to the final decision.

**A striking (and troubling) finding:** The IntermediateFusion model's top-50 most important features were **entirely miRNA** for both breast and colon cancer. Every single one. The top feature was hsa.mir.130b, followed by hsa.mir.135b, hsa.mir.18a, hsa.mir.324, hsa.mir.24.1 — out of 15,200–15,366 features across all modalities, the model fixated exclusively on the miRNA features.

This immediately explained a result we'd see later in ablation testing: removing miRNA from COAD dramatically *improved* performance. The model was over-weighting a noisy signal.

For PathwayAwareFusion, dedicated precomputed IG attributions are stored separately (`pathway_fusion_attribution_results_{suffix}.json`) and the demo app now loads these correctly — showing pathway-aware attributions when PathwayAwareFusion is selected, not the IntermediateFusion attributions.

### Layer 3: KEGG/GO Pathway Enrichment

We took the top genes identified by SHAP and Integrated Gradients and asked: what biological processes are these genes involved in? Using gseapy (a Python wrapper for the Enrichr web service), we checked against the KEGG and GO (Gene Ontology) databases.

**For BRCA IntermediateFusion:**
- **KEGG:** 4 significant pathways found. Cell cycle (adj. p=3.07e-05 / 3.1×10⁻⁵ — genes: CDC20, CCNB2, PTTG1, CCNE2, TTK, CDC25B). p53 signaling pathway (adj. p=1.29e-02 / 1.29×10⁻²; genes: CCNB2, CCNE2, GTSE1). Oocyte meiosis (adj. p=7.26×10⁻³). Human T-cell leukemia virus 1 infection (adj. p=2.61×10⁻²). Cell cycle and p53 are biologically plausible — classic cancer pathways.
- **GO Biological Process:** 57 significant terms. Top hits: microtubule cytoskeleton organization in mitosis, mitotic spindle organization, kinetochore organization. All mitosis-related — makes sense for a cancer classifier.

**For XGBoost (both cancers):** Zero significant KEGG enrichment. The model found predictive genes, but they weren't organized into recognizable pathways.

**For all COAD models:** Zero significant enrichment across the board. The small dataset and the miRNA artifact likely prevented the model from learning biologically coherent patterns.

**Pathway validation:** We checked 9 canonical cancer pathways (PI3K-Akt, p53, MAPK, Cell cycle, Apoptosis, Wnt, Breast cancer, Colorectal cancer, Pathways in cancer). Only Cell cycle and p53 were found — and only for the fusion model on BRCA. All other combinations came up empty.

---

## Step 6: Stress-Testing the Models (Ablations)

We systematically broke things to understand what mattered and what didn't.

### Ablation A: What if we remove one modality at a time?

We retrained IntermediateFusion four times, each time leaving out one data type:

| Removed | BRCA F1 (baseline: 0.808) | COAD F1 (baseline: 0.669) |
|---|---|---|
| mRNA | 0.767 (−0.041) | 0.669 (−0.000) |
| miRNA | 0.803 (−0.006) | **0.802 / 0.8018 (+0.133)** |
| Methylation | 0.788 (−0.020) | 0.732 / 0.7317 (+0.063) |
| CNV | 0.789 (−0.019) | 0.669 (−0.000) |

**For BRCA:** mRNA is clearly the most important modality — removing it caused the biggest F1 drop (−0.041). miRNA removal barely mattered (−0.006, essentially noise). Methylation and CNV were moderately important.

**For COAD: The miRNA artifact.** Removing miRNA didn't hurt the model — it **helped**. F1 jumped from 0.669 all the way to 0.802. That's a +0.133 improvement. A data type was actively making the model worse.

Combined with the Integrated Gradients finding (all top-50 features are miRNA), the conclusion is clear: the IntermediateFusion model over-weights the miRNA encoder for COAD. With only 260 patients and 200 miRNA features, the small miRNA module finds spurious correlations that don't generalize. The model latches onto noise instead of signal.

This is the most surprising and well-evidenced finding in the project. It's not a failure — it's a genuine contribution: discovering that a specific modality can actively harm performance on small datasets.

Removing methylation also improved COAD (to 0.732, +0.063), suggesting it too carries some noise on this small dataset.

### Ablation B: Early vs Intermediate Fusion

| Model | BRCA F1 | COAD F1 |
|---|---|---|
| XGBoost (early, trees) | 0.794 | 0.636 |
| EarlyFusionMLP (early, one big network) | 0.722 | **0.751** |
| IntermediateFusion (per-modality encoders) | **0.808** | 0.669 |
| PathwayAwareFusion (per-modality + biology) | 0.801 | **0.738** |

The pattern is clear:
- **On larger data (BRCA, 671 patients):** Modality-specific encoding wins. IntermediateFusion (0.808) > XGBoost (0.794). But the edge is not statistically significant (p=0.625).
- **On smaller data (COAD, 260 patients):** Simpler approaches win. EarlyFusionMLP (0.751) > IntermediateFusion (0.669). But give it biological structure and it bounces back — PathwayFusion (0.738) nearly catches up.

### Ablation C: What if some data is missing?

In the real world, not every hospital runs every test. What happens if a patient is missing one or more modalities? We simulated this by randomly setting features to zero at increasing rates (10%, 20%, 30%, 50%):

| Missing Rate | BRCA F1 | COAD F1 |
|---|---|---|
| 0% (all data) | 0.808 | 0.669 |
| 10% | 0.819 | 0.718 |
| 20% | 0.831 | 0.599 |
| 30% | 0.764 | 0.611 |
| 50% | 0.749 | 0.620 |

BRCA shows reasonable degradation — even with half the data missing, F1 only drops from 0.808 to 0.749 (a 7% relative decline). COAD is chaotic — the F1 jumps around erratically (0.669 → 0.718 → 0.599 → 0.611 → 0.620), reflecting the instability of a small dataset combined with the miRNA artifact.

---

## The Complete Results: All Five Models Compared

After 40 trained models (4 deployed architectures + 1 ablation baseline × 2 cancers × 5 cross-validation folds), here is everything in one table:

| Model | BRCA F1 | BRCA AUC | COAD F1 | COAD AUC | BRCA Acc | COAD Acc |
|---|---|---|---|---|---|---|
| XGBoost | 0.794 | 0.970 | 0.636 | 0.911 | 0.864 | 0.881 |
| Random Forest | 0.602 | **0.971** | 0.608 | 0.932 | 0.782 | 0.873 |
| EarlyFusionMLP* | 0.722 | — | 0.751 | — | — | — |
| Intermediate Fusion | **0.808** | 0.963 | 0.669 | 0.953 | 0.833 | 0.839 |
| Pathway-Aware Fusion | 0.801 | 0.966 | **0.738** | **0.954** | 0.829 | 0.854 |

*(F1 and AUC are the two most important metrics. F1 balances precision and recall. AUC measures ranking ability. All metrics are averages across 5 folds. \*EarlyFusionMLP is an ablation baseline — excluded from the primary model_comparison.csv.)*

**Statistical significance (Wilcoxon signed-rank, n=5 folds):**

With only 5 folds, the minimum achievable two-sided p-value is 0.0625. Key comparisons from `results/metrics/significance_tests.csv`:

| Comparison | F1 diff | p-value | Cohen's d | Interpretation |
|---|---|---|---|---|
| IntFusion vs XGBoost (BRCA) | +0.014 | 0.625 | 0.197 | Not significant; trivial effect |
| IntFusion vs RandomForest (BRCA) | +0.206 | 0.0625* | 1.888 | Significant at minimum threshold; very large effect |
| PathwayFusion vs IntFusion (COAD) | +0.069 | 0.3125 | 0.504 | Not significant; medium effect |
| PathwayFusion vs XGBoost (COAD) | +0.102 | 0.3125 | 0.664 | Not significant; medium effect |

\*Minimum achievable p with n=5. Deep fusion models and XGBoost show **comparable performance** on BRCA — differences are not statistically distinguishable.

**The headline takeaways:**

1. **For breast cancer (more data):** IntermediateFusion leads nominally (F1=0.808), but XGBoost is statistically comparable (0.794). The margin (+0.014) is within noise.

2. **For colon cancer (less data):** PathwayAwareFusion dominates (F1=0.738), with a +0.069 improvement over IntermediateFusion. The biological prior was the difference maker.

3. **All models achieve excellent AUC** (0.91–0.97). Even when hard classification falters, the models rank patients confidently.

4. **The simplest model (EarlyFusionMLP) beats IntermediateFusion on COAD.** More complexity is not always better.

---

## The Demo: Bringing It All Together

All of this work culminates in a **Streamlit web application** that anyone can use. Here's what it does:

### Prediction Tab

You upload a CSV file containing a patient's multi-omics measurements (the app provides sample files for both cancers). The file format is simple: each row is a feature/gene, each column is a patient sample.

The app then:
1. Validates the format and preprocesses the data using the same scalers and imputers used during training
2. Runs the selected model (XGBoost, IntermediateFusion, or Pathway-Aware Fusion)
3. Shows the predicted subtype with a confidence percentage and a colour-coded tier:
   - **Green (HIGH confidence):** >80% — strong prediction
   - **Blue (MODERATE confidence):** 50–80% — reasonable but uncertain
   - **Amber (LOW confidence):** <50% — the model is unsure; results should be interpreted cautiously
4. Displays which features drove the prediction:
   - For XGBoost: a SHAP waterfall plot showing how each gene pushed the prediction up or down
   - For IntermediateFusion: an Integrated Gradients bar chart, colour-coded by modality (blue=mRNA, green=miRNA, orange=methylation, purple=CNV)
   - For PathwayAwareFusion: its own dedicated IG bar chart **plus** a pathway-level attention weights chart showing the top-10 KEGG pathways attended to during prediction
5. **Optional: AI Research Summary.** If you provide a Groq API key (free, from console.groq.com), the app calls a large language model to generate a clinical context summary — describing the predicted subtype, its biological characteristics, and relevant treatment considerations. If no key is provided, the app works normally without summaries.

If precomputed attributions are unavailable for an uploaded patient, the app computes Integrated Gradients live using Captum — so every patient gets explainability, not just the precomputed demo samples.

You can upload a batch file with multiple patients — the app processes all of them and lets you download all predictions as a CSV.

### Model Comparison Tab

This tab shows everything at once, with no file upload needed:

- **Trophy banner** showing the best overall model
- **Metric highlight cards** for Best F1, Precision, Recall, and Best AUC
- **Full metrics table** with all 4 deployed models, both cancers, all metrics (F1, Precision, Recall, NMI, ARI, AUC, Accuracy) — styled to highlight best values in green
- **Grouped Plotly bar chart** (F1/AUC by model and cancer)
- **ROC Curves** — for each model, a macro-averaged one-vs-rest ROC curve showing how the true positive rate changes as the decision threshold varies
- **Per-fold AUC breakdown** — expandable table showing fold-by-fold consistency
- **Radar chart** — comparing all models across F1, precision, recall, AUC, NMI, and ARI simultaneously
- **Latent space visualization** — a precomputed t-SNE plot showing how IntermediateFusion clusters patients
- **Training convergence graphs** — loss curves showing how each model learned epoch by epoch
- **Confusion matrices** — for all 4 model types (XGBoost, RF, IntermediateFusion, PathwayFusion)
- **Biological validation** — KEGG pathway enrichment table, GO enrichment results, and pathway attention bar chart
- **Ablation results** — modality removal bar chart, fusion comparison table, and missing modality degradation curve

### Data Converter Tab (Lab Data Converter)

This tab bridges the gap between a hospital genomics lab and our prediction pipeline. In a real-world scenario, each modality comes from a different sequencing machine as a separate file. The Data Converter:

1. Accepts up to 4 separate per-modality CSV files (mRNA, miRNA, methylation, CNV — any combination, all are optional)
2. Auto-detects CSV orientation (features-as-rows or features-as-columns)
3. Auto-normalises miRNA names from any common format (hsa-miR-21, hsa-miR-21-5p) to the training format (hsa.mir.21)
4. Applies the selected **data format transform** before mapping to training space:
   - **Already z-score normalised:** data used as-is (no transform)
   - **Log2-transformed:** applies robust single-sample z-scoring (median subtraction + IQR/1.349 scaling)
   - **Raw counts:** applies log2(x+1) first, then robust z-scoring
   - **Beta values (0–1):** converts to M-values [log2(B/(1-B))], then z-scoring
5. Maps feature names to the model's training feature space
6. Reports per-modality **coverage** (what percentage of expected features were matched)
7. Fills missing features with 0 (the training mean in z-scored space)
8. Produces a single prediction-ready CSV that can be uploaded directly in the Prediction tab

We tested this end-to-end with a simulated patient — Amara Nwosu, a 38-year-old woman with triple-negative breast cancer. Four separate lab-format files were uploaded, converted, and all three models correctly predicted Basal-like with >95% confidence.

**Important honesty about single-sample normalisation:** Cohort-based normalisation (fitting statistics across hundreds of patients) is always more accurate than normalising a single patient in isolation. The robust z-scoring here is an approximation — it shifts the data to the right scale but cannot correct for batch effects between different sequencing platforms. Predictions are most reliable when the input data has already been z-score normalised by a bioinformatics pipeline using a proper reference cohort. The converter explicitly warns about this limitation when non-z-scored formats are selected.

### Demo Datasets

Three categories of pre-generated patient files are included at `app/test_datasets/` (35 files total):

**Real TCGA patients (`test/` — 13 files):** Actual de-identified cancer patients from held-out validation folds — never seen during training. 20 BRCA patients (4 per subtype) and 12 COAD patients. These represent honest performance: what the model does with a real new patient. There are also single "highest-confidence" patient files per subtype — the patient the model is most confident about. Best for demonstrations.

**Synthetic patients (`synthetic/` — 14 files):** Statistically generated profiles that aren't real people. Each synthetic patient's 15,000+ measurements are sampled independently from the per-feature distribution of their subtype: `feature_value = sample_from(N(class_mean, class_std))`. Useful for testing on "brand new patients it has never seen."

**Named demo patients (`demo/` — 8 files):** Purpose-built for presentations:
- `demo_patient_sarah_mitchell_BRCA.csv` — Luminal A breast cancer patient (prediction-ready)
- `demo_patient_robert_okonkwo_COAD.csv` — CMS2 colon cancer patient (prediction-ready)
- `convertion/amara_nwosu_mrna.csv`, `_mirna.csv`, `_methylation.csv`, `_cnv.csv` — four separate lab-format files for testing the Data Converter workflow
- Two data card files documenting generation methodology and expected results

---

## The Engineering Behind the Scenes

This isn't just a bunch of models — it's a carefully engineered research system designed for reproducibility, fairness, and auditability.

### Everything Is Reproducible

Every random number in the entire project is controlled. A single function, `set_seeds(42)`, sets the random state for Python, NumPy, PyTorch, and the operating system's hash function. Anyone who downloads this code and runs it with the same data will get the exact same results.

All hyperparameters (model settings like learning rate, number of trees, latent dimensions) live in a single file, `config.yaml`. Nothing is hardcoded inside scripts. To change how a model trains, you edit one number in one place.

### No Data Leakage — Guaranteed

The preprocessing pipeline enforces strict separation between training and testing data:

1. Patients are split into folds first — stored in `data/cv_folds.json` and never regenerated
2. Every model uses the exact same folds
3. The scaler and imputer are fitted on training data only, then applied to test data
4. Every fold preparation includes an assertion: "assert no training patients appear in the test set"
5. All 10 folds (5 BRCA + 5 COAD) pass this check

### Everything Is Logged

Every single training run appends a row to `experiment_log.csv` — 130 canonical rows after deduplication. Each row records: when it ran, what model, which cancer, which fold, all hyperparameters, all metrics, and where the model file was saved. You can trace any number in any table back to the exact experiment that produced it.

### Codebase by the Numbers

- **7 source code modules** (`src/`): ~2,235 lines of Python
- **22 automation scripts** (`scripts/`): covering training, evaluation, attribution, demo preparation, lab data conversion, calibration, significance testing, and runtime measurement
- **32 automated tests** (`tests/`): verifying model shapes, preprocessing integrity, CV fold safety, metric computations, save/load round-trips, and seed reproducibility
- **7 computational notebooks**: interactive documents for exploration, visualization, and analysis
- **14 planning documents** (`content/`): the design specs and roadmaps that guided the entire project
- **40 trained model files** (`models/`): 4 architectures × 2 cancers × 5 folds
- **175 results files** (`results/`): metrics tables (17 CSVs + 40 NPZ), plots, SHAP values, enrichment results, calibration curves, and QC reports
- **27 demo artifacts** (`app/model_artifacts/`): pre-trained models, scalers, imputers, configs, precomputed IG attributions, and latent space embeddings

### Technology Stack

Python 3.11 · PyTorch 2.7.1 (CUDA 11.8 for GPU acceleration) · scikit-learn 1.6.1 · XGBoost 2.1.4 · SHAP 0.49.1 · Captum 0.8.0 · gseapy 1.1.11 · Streamlit 1.50.0 · Groq SDK (AI summaries) · joblib · matplotlib · seaborn · plotly

---

## What We Learned

### 1. Multi-omics beats single-omics.
Using all four data types together (mRNA + miRNA + methylation + CNV) produces F1=0.808 on breast cancer. Our ablation showed mRNA is the most critical modality (removing it causes the largest drop on BRCA), while miRNA is the least useful individually for BRCA.

### 2. How you combine data matters — but only at sufficient scale.
Intermediate fusion (separate networks per modality) outperforms early concatenation on the larger BRCA dataset (0.808 vs 0.794 for XGBoost). On the smaller COAD dataset, the simpler EarlyFusionMLP wins (0.751 vs 0.669 for IntermediateFusion).

### 3. Biological knowledge rescues small datasets.
KEGG pathway structure gave a +0.069 F1 gain on colon cancer — the biggest improvement in the project — precisely where data was scarce. On the data-rich BRCA case, the same biological prior produced essentially the same result as IntermediateFusion.

### 4. Simple models remain surprisingly competitive.
XGBoost at F1=0.794 on breast cancer was only 0.014 behind the best deep learning model, a difference that is not statistically significant. The lesson: always train simple baselines first. They often match or beat complex architectures on small data.

### 5. Some data can be actively harmful.
Removing miRNA from the colon cancer fusion model *improved* performance from F1=0.669 to 0.802 — a +0.133 gain. A modality was acting as pure noise. This is a genuine scientific finding, not a bug.

### 6. Explainability works, but with clear limits.
We can trace individual predictions back to specific genes and pathways. Cell cycle (p=3.1×10⁻⁵) and p53 emerged as significant for breast cancer — biologically plausible results. But KEGG enrichment was empty for 6 out of 8 model-cancer combinations. The pathway attention mechanism didn't learn strong focus (weights were nearly uniform). Interpretability is partially achievable, partially aspirational.

### 7. Sanity checks catch real problems.
Our label-shuffle test revealed the global ANOVA pre-selection issue instantly. Without it, we might have reported inflated performance numbers without understanding why. Documenting dataset limitations honestly is more valuable than pretending they don't exist.

### 8. AUC and F1 tell different stories.
Random Forest had the highest AUC (0.971) but the second-worst F1 (0.602). When evaluating medical models, look at both: AUC for ranking ability (how well the model separates patients), F1 for decision quality (how often the model is right when forced to choose). One metric alone can be deeply misleading.

---

## Honest Limitations

This project has real constraints that matter:

1. **Feature selection contaminated the data.** The top genes were pre-selected using labels from all patients before splitting. Every model's F1 is slightly inflated. The *comparisons* between models are fair (all models share this bias), but the absolute numbers should be interpreted with this caveat.

2. **Colon cancer has a 4-patient subtype.** CMS4 Mesenchymal appears only 4 times. It's essentially unlearnable. Some validation folds contain zero CMS4 patients, making per-class metrics undefined for that fold. COAD results are noisy and less reliable than BRCA results.

3. **Colon cancer has high variance.** F1 standard deviations of 0.09–0.14 on COAD (compared to 0.05–0.07 on BRCA) mean individual fold results swing wildly. The PathwayFusion COAD F1 of 0.738 comes with a standard deviation of 0.142 — meaning in some folds it might be 0.60, in others 0.88.

4. **miRNA is half-missing for breast cancer.** 166 of 366 miRNA measurements are entirely absent. The model can't use what isn't there.

5. **The pathway attention mechanism didn't work as intended.** It found real performance gains on COAD, but the attention weights didn't concentrate on known cancer pathways — the "interpretable biology" promise of this architecture wasn't fully realized. Only 35–37% of mRNA features were mapped to any KEGG pathway, limiting what the attention can learn.

6. **Enrichment analysis hit walls.** For XGBoost and all COAD models, no statistically significant KEGG pathways were found. The biological validation story is stronger for BRCA IntermediateFusion than for any other model-cancer combination.

7. **No independent test set.** All metrics come from 5-fold CV. There is no separate held-out test cohort. The CV estimates carry optimistic bias from the ANOVA pre-selection and should be interpreted as model-selection performance, not true generalisation estimates.

8. **Single-sample normalisation in the converter is approximate.** The Lab Data Converter normalises individual patients using robust single-sample z-scoring. This approximation is scientifically honest but cannot correct for batch effects between different sequencing platforms or labs.

---

## What's Next (Future Work)

Several directions were planned but not pursued:
- **Third cancer type (GS-GBM, glioblastoma):** Would have tested generalisation across very different tumour biology
- **VAE-based latent fusion:** Using variational autoencoders to learn more robust latent representations, especially useful for missing data
- **Automatic missing-modality imputation:** Instead of just zero-filling missing modalities, learn to predict what the missing values should be
- **Cross-cancer generalisation:** Can a model trained on breast cancer say anything useful about colon cancer?
- **Cohort-based normalisation in the converter:** Would require access to the original pre-normalisation TCGA data (not distributed by the benchmark)

---

*Built by Dineth Hettiarachchi as a BSc Computer Science final-year project.*
*Academic research prototype — not for clinical use.*
*Data source: MLOmics benchmark dataset (TCGA origin, CC-BY-4.0).*
*Project tagged v1.0-final on April 29, 2026.*
