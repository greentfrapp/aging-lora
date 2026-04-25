# Paper outline (working draft)

Target venue: bioRxiv preprint after Phase 3, refined for journal submission after Phase 5. Estimated length: 5,000–7,000 words main text + supplementary methods/figures. Edit-in-place as the paper takes shape.

---

## Working title

*"Cell-type-specific immune aging clocks: foundation-model fine-tuning under leakage, chemistry, and ancestry stress-tests"*

(Refine after results land. Alternatives: "Do single-cell foundation models improve immune aging clocks?" — implies null result more openly.)

## Abstract sketch

- Aging clocks for PBMC scRNA-seq exist (sc-ImmuAging LASSO 2025, scAgeClock 2026, Pasta-REG 2025) but no head-to-head leakage-audited benchmark.
- We assemble a 4-cohort, 4-baseline LOCO benchmark with explicit pretraining-corpus leakage flags + chemistry-match flags + detectability flags per result row.
- Fine-tune 3-4 foundation models (Geneformer/scFoundation/scGPT/UCE) with LoRA across 5 PBMC cell types and 3 LOCO folds + 1 ancestry holdout (AIDA).
- Headline finding: [Phase-3 outcome — to be filled].
- Secondary: chemistry-rescue B-cell story; cross-ancestry transfer; zero-shot cell-type transfer; full-fine-tune ablation.
- v1 benchmark harness released alongside the preprint.

## Section structure

### 1. Introduction (~800–1,000 words)

- **Hook**: aging clocks are a major motivating application of single-cell foundation models, but the leakage-audited generalization picture has not been characterized.
- **Setup**: sc-ImmuAging (Li 2025) defines the immune-aging task; scAgeClock (Xie 2026) attempts a deep generalist model; Pasta (Salignon 2025) approaches it bulk-side. None of them are head-to-head benchmarked under leakage + chemistry + ancestry stress.
- **Foundation models hypothesis**: if scRNA-seq pretraining buys anything for aging-signal extraction, it should manifest in low-data regimes (B-cell), cross-chemistry transfer (3'→5'), and cross-ancestry transfer (European→Asian).
- **Contribution**: (1) a leakage-audited 4-cohort × 4-baseline × 5-cell-type LOCO benchmark (75 baseline rows + FM rows); (2) head-to-head FM-vs-baseline comparison with a min-of-4 best-baseline rule; (3) chemistry- and ancestry-robustness sub-figures; (4) v1 benchmark harness for future immune-aging FMs to plug into.

### 2. Results

#### 2.1 Benchmark design (Methods detail in `methods/`, summary here)

- 4 cohorts: OneK1K (981 donors, 10x 3' v2, European), Stephenson (29 healthy controls, 10x 3'), Terekhova (166 donors, 10x 5' v2, European), AIDA (625 donors, 10x 5' v2, 7 Asian populations).
- 4 baselines: LASSO-pretrained-5cohort (sc-ImmuAging upstream), LASSO-retrained-3cohort (apples-to-apples), scAgeClock (deep), Pasta-REG (bulk).
- Leakage audit (20-row table): scGPT/UCE/scAgeClock overlap OneK1K+Stephenson via CELLxGENE Census; scFoundation overlaps Stephenson via HCA-Covid19PBMC mirror; only Geneformer is `clean` everywhere.
- Per-row stratification: `leakage_status` × `chemistry_match` × `detectability_flag`.

#### 2.2 Baseline panel: per-cell minimum-MAE bar (Phase-2 result, already in hand)

- 75-row `loco_baseline_table.csv` headline: Pasta-REG CD4+T MAE=6.3y on AIDA — the toughest cell.
- Pasta wins Terekhova CD4T/CD8T/B (chemistry-rescue); LASSO wins NK/MONO; scAgeClock weaker than LASSO across the board.
- Pretrained vs retrained LASSO essentially equivalent for T cells → FM win cannot be attributed to "more training data."

#### 2.3 CD4+T tri-headline FM result (Phase-3 result — TBD)

- Three headline cells: OneK1K (3' chemistry, 981 donors), Terekhova (5' chemistry shift, 166 donors), AIDA (5' Asian ancestry, 595 donors).
- Per-cell win/match/loss against per-cell minimum-MAE of 4 baselines.
- Aggregate outcome (3/3, 2/3, 1/3, or 0/3 wins) drives the framing.

#### 2.4 Chemistry + ancestry robustness 3×3 subfigure (Phase-3 result — TBD)

- 3-comparator (LASSO-pre, Pasta-REG, Geneformer-LoRA) × 3-context (3' OneK1K, 5' Terekhova European, 5' AIDA Asian).
- Tests: does single-cell FM pretraining add chemistry-invariance + ancestry-invariance *on top of* what rank-normalized bulk modelling already provides?

#### 2.5 Few-shot regime + chemistry-rescue B-cell story (Phase-4 result — TBD)

- Dual-regime curve: CD4+T (chemistry-robust baseline) vs B cells on Terekhova (chemistry-collapsed baseline R=0.08).
- 3-line per cell type: LASSO (count, chemistry-sensitive), Pasta (rank-norm, chemistry-invariant), best FM.
- Crossover thresholds: at what donor count does the FM line beat both baselines?

#### 2.6 Zero-shot cell-type transfer (Phase-4 result — TBD)

- For each (source cell type → target cell type) joint holdout: FM cross-cell-type embedding alignment vs Pasta-on-target (cell-type-agnostic by construction).
- Headline: do FMs transfer better than naive rank-norm-on-pseudobulk?

#### 2.7 Biological readouts (Phase-5 result — TBD)

- Age-axis cosine similarity across cell types and ancestries (AIDA holdout).
- SHAP attribution: top-50 age-driver genes per (FM, cell type); cross-reference with sc-ImmuAging LASSO and iAge cytokine genes.
- In-silico perturbation of top-5 SHAP genes (replication of Tadevosyan 2025).

### 3. Discussion

- Where FMs win and where they don't (cell-type-specific patterns).
- The "scAgeClock weaker than LASSO" result and what it implies for generalist deep clocks.
- The Pasta chemistry-invariance lesson: a strong baseline reframes the FM contribution.
- Caveats: leakage stratification, chemistry shift, detectability floors, scAgeClock systematic bias.
- Limitations: 3-cohort training corpus, no genotype data, EGA-cohort access denied.

### 4. Methods

- Brief in main text, full in supplementary (sourced from `methods/*.md` docs).
- Cohort sources + harmonization (sourced from `methods/datasets/*.md`).
- Leakage audit methodology (sourced from `methods/leakage_audit_notes.md`).
- Pretrained LASSO scoring (`methods/pretrained_lasso_sanity_check.md`).
- Terekhova chemistry-shift analysis (`methods/terekhova_chemistry_shift.md`).
- Detectability floor with bracketed ρ disclosure (`methods/detectability_floor.md`).
- Three Phase-2 baselines + retrained LASSO (`methods/loco_baselines.md`).
- LoRA fine-tuning protocol (Phase-3, TBD).
- Statistical analysis (paired Wilcoxon, random-effects meta-analysis).

### 5. Data and code availability

- v1 benchmark harness on Zenodo (Phase-5 deliverable).
- Frozen LOCO splits + AIDA split + checkpoint hashes.
- 75-row baseline table as a CSV in supplementary.

## Figure-table inventory

See `paper_figures_inventory.md` and `paper_tables_inventory.md`.

## Key claims (in order of priority)

1. **The leakage-audited 4-cohort 4-baseline benchmark is itself a contribution** — useful for future immune-aging FM work regardless of our FM result.
2. **scAgeClock 2026 underperforms a PBMC-specialist LASSO** — informative about generalist vs specialist deep clocks.
3. **Pasta's rank-norm bulk approach is chemistry- and ancestry-invariant** — sets a high bar for FMs.
4. **FMs [win/match/lose] against the per-cell minimum baseline on the tri-cell CD4+T headline** (Phase 3 fills this in).
5. **Chemistry-rescue + cross-ancestry transfer + zero-shot cell-type transfer** are tested on the same panel for the first time (Phase 4).
6. **v1 benchmark harness released** — enables apples-to-apples future evaluation.
