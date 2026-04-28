## Summary
<!-- 2-3 sentences: what this PR does and why -->

## Phase Reference
<!-- Which roadmap phase/tasks this covers -->
- Phase: P{N}
- Tasks: P{N}-T{X}, P{N}-T{Y}

## What Changed
<!-- Bullet list of concrete changes. Group by category. -->

### New Files
- `src/{file}.py` — {one-line description}
- `notebooks/{NN}_{name}.ipynb` — {one-line description}

### Modified Files
- `src/{file}.py` — {what changed and why}
- `config.yaml` — {what was added/changed}

### New Artifacts
- `results/metrics/{file}.csv` — {description}
- `results/plots/{file}.png` — {description}
- `models/{subfolder}/{file}` — {description}

## Key Decisions
<!-- Any non-obvious technical choices made. 1-3 bullets max. Skip if none. -->
- {Decision}: {Rationale}

## Verification
<!-- Paste the relevant gate check results -->
- [ ] Tested on toy data first
- [ ] Tested on full data
- [ ] No data leakage (train/val split verified)
- [ ] experiment_log.csv updated
- [ ] All artifacts saved to results/
- [ ] Notebook runs top-to-bottom without error
- [ ] Seeds set (42) — results reproducible

## Metrics (if applicable)
<!-- Paste key numbers. Delete section if no metrics this PR. -->
| Model | Cancer | F1 (mean±std) | Precision | Recall |
|-------|--------|---------------|-----------|--------|
| {name} | {cancer} | {value} | {value} | {value} |

## PR Checklist (from CONTEXT.md)
- [ ] `experiment_log.csv` updated with new results (if any experiments ran)
- [ ] Model artifacts saved to correct `/models/` subfolder
- [ ] No data leakage: confirmed train/val split is patient-level
- [ ] Notebooks run top-to-bottom without error (Kernel → Restart & Run All)
- [ ] No large files (>10MB) committed — check `.gitignore` covers them
- [ ] Random seeds set and documented (seed=42 in all scripts)
- [ ] No hardcoded absolute paths (all paths relative to project root)
- [ ] Code is commented where non-obvious logic exists

## Next Step
<!-- Single sentence: what comes after this merges -->
