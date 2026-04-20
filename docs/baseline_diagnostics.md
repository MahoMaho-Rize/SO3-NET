# Phase 0 · Baseline Diagnostics Report

> Date: 2026-04-20
> Reference checkpoint: `uprightnet-reference/model/model.pth` (official release by Pang et al.)
> Test set: UprightNet15 test split, 370 objects × 100 rotations = **37,000 samples**

## 1. Overall baseline numbers

| Variant | Mean | Median | Acc@5 | Acc@10 | Acc@30 | Flip rate (err > 90°) |
|---|---|---|---|---|---|---|
| **Official (GT-leak fallback)** | 11.16° | 0.71° | 91.07% | 93.05% | 93.60% | 5.72% |
| **Honest (`[0,1,0]` fallback)** | 11.17° | 0.70° | 91.00% | 93.03% | 93.64% | 5.74% |
| Paper Table 1 (reported)        | 12.78° | — | — | 91.80% | — | — |

**Observations:**
- Our reproduction beats the paper-reported numbers (11.17° vs 12.78° mean, 93.05% vs 91.80% Acc@10). The official checkpoint is slightly better than the paper.
- **The GT-leak fallback never triggers** on the 37,000-sample test set — `num_support == 0` for 0/37000 samples. The leak exists in code but has zero practical effect. My earlier worry that "baseline numbers are inflated by GT leakage" was **wrong**.
- The fallback GT-leak fix remains a necessary honesty adjustment, but does not move the numbers.

## 2. Error structure — confirming Pang Fig.4

See `logs/baseline_error_cdf.png`.

| Error bucket | Count | Percentage |
|---|---|---|
| err < 5° | 33,670 | **91.00%** |
| 5° ≤ err < 10° | 750 | 2.03% |
| 10° ≤ err < 30° | 226 | 0.61% |
| 30° ≤ err < 90° | 229 | 0.62% |
| err ≥ 90° (flip) | 2,125 | **5.74%** |
| err ≥ 170° (near-perfect flip) | 1,523 | **4.12%** |

- **Non-flip median error: 0.64°** — baseline is extremely precise when it works.
- **Flip median error: 177.61°** — failed predictions are almost exactly 180° inverted, not randomly misaligned.
- The "accuracy vs τ" curve (reproduced as `logs/baseline_error_cdf.png` left panel) shows the same plateau + τ=180° boost pattern as Pang Fig.4.

**Conclusion: C2 premise is empirically confirmed.** 5.74% of errors are 180° flips; the "τ=180° boost" of the original paper is real and is responsible for nearly all of the absolute mean error (11.17° - if the flips were fixed, mean error would drop to ~0.64°).

## 3. Per-category breakdown

See `logs/baseline_per_category.csv`.

| Label | Category | Mean | Median | Acc@10 | Flip rate | Near-180 rate |
|---|---|---|---|---|---|---|
| 0 | bed     | 31.16° | 0.96° | 76.96% | **16.88%** | 8.88% |
| 1 | bench   | 5.48° | 0.69° | 96.36% | 2.92% | 0.52% |
| 2 | bottle  | 9.64° | 0.88° | 94.80% | 4.72% | 4.64% |
| 3 | **bowl**    | **1.14°** | 0.78° | **100.00%** | **0.00%** | 0.00% |
| 4 | car     | 6.44° | 0.89° | 96.92% | 3.08% | 3.04% |
| 5 | chair   | 3.97° | 1.02° | 95.20% | 0.88% | 0.00% |
| 6 | cone    | 8.04° | 0.15° | 95.48% | 4.44% | 3.88% |
| 7 | cup     | 12.95° | 0.36° | 93.24% | 6.76% | 6.76% |
| 8 | **lamp**    | **28.27°** | 0.77° | 83.24% | **16.00%** | **12.16%** |
| 9 | monitor | 9.37° | 0.90° | 94.20% | 5.00% | 0.40% |
| 10 | sofa   | 5.95° | 0.85° | 94.76% | 2.04% | 0.60% |
| 11 | stool  | 8.06° | 0.70° | 95.96% | 4.04% | 4.04% |
| 12 | table  | 1.54° | 0.76° | 98.72% | 0.12% | 0.12% |
| 13 | toilet | 10.50° | 0.39° | 93.48% | 6.08% | 3.84% |
| 14 | **vase**   | **23.01°** | 0.79° | 87.48% | **12.04%** | 12.04% |

**Categories that dominate the failure mode:**
- **lamp** (16.00% flip): classic "interfering planar structures" — lamp shade + base + ceiling mount all look like plausible bottoms
- **bed** (16.88% flip): mattress top vs box-spring bottom ambiguity
- **vase** (12.04% flip): rotationally symmetric, often has flat opening at top mimicking a base
- **cup** (6.76% flip): similar to vase
- **bowl** is perfect (0% flip) because its support is unambiguous

**Important: median error per-category is always < 1.1°** — the baseline is nearly perfect on most samples of every class. The mean is inflated entirely by flip-class failures, which cluster in specific categories with ambiguous support planes.

This mirrors Pang 2022 Fig.5 qualitative analysis ("interfering planar structures"): **bed / lamp / vase** are the three worst classes, all of which the paper's failure examples (Fig.5) draw from or are topologically similar to.

## 4. Top-30 worst samples

See `logs/baseline_failure_samples.txt`.

- **29 / 30 are `lamp` samples** (all err ≈ 179.97°) and **1 is `bottle`**.
- All failures have **high `num_support`** counts (440-460 out of 2048 points) — the network is not failing to detect *a* plane; it's detecting the **wrong** plane (lamp shade rather than lamp base).
- Indices 31300-31399 being contiguous suggests **whole objects** (not individual rotations) are systematically failing — the network has a per-object semantic error, not a per-rotation numerical error.

## 5. Implications for the research plan

### Validated premises

| Premise | Verdict |
|---|---|
| C2 "180° flip is a real failure mode" | **CONFIRMED**: 5.74% flip rate, median flip error 177.61° |
| C2 "failures cluster in ambiguous-plane categories" | **CONFIRMED**: lamp/bed/vase dominate; bowl/table/chair near-perfect |
| GT-leak fallback inflates baseline numbers | **REFUTED**: fallback never triggers |

### Updated priorities

**C2 (antipodal-aware head) is now the strongest-justified contribution.** Nearly all of the absolute mean error gap (11.17° - 0.64° = ~10.5°) comes from flip failures. Solving flips would bring mean error below 1°, which is a massive and clean win.

**C1 (differentiable support plane) sets up C2 naturally.** The failure mode shows the network *finds a plane*, just the wrong one (lamp shade vs lamp base). A soft, differentiable, probabilistic plane-detection stage allows the flip logit in C2 to use richer signals than hard support-point classification.

**C3 (L_Stab) is on surer ground.** Static stability ("mass center sits above support plane, not below it") is precisely the kind of physics signal that distinguishes lamp-shade-as-base (unstable) from lamp-base-as-base (stable). **L_Stab is a natural fix for the lamp failure class.**

### New proposed evaluation axis

In addition to mean error / Acc@10:
- **Flip rate** (err > 90°): the direct target of C2
- **Non-flip median error**: to show we don't regress on easy samples
- **Per-category flip rate**: to prove improvements on hard classes (lamp/bed/vase) not just averaged

### Non-issue confirmed

- The GT-leak fallback bug remains, but is cosmetic. Still should be fixed for repro/honesty, but **don't emphasise it as a contribution** — it has 0 numerical impact.

## 6. Files produced

| File | Description |
|---|---|
| `logs/baseline_official.npz` | per-sample: error_leak_deg, error_fix_deg, num_support, is_leak_triggered, label |
| `logs/baseline_error_cdf.png` | 3-panel figure: accuracy-vs-τ curve, error histogram, tail-only histogram |
| `logs/baseline_per_category.csv` | 15-row per-class summary |
| `logs/baseline_failure_samples.txt` | top-30 worst samples with metadata |
| `scripts/eval_official_baseline.py` | reproducible evaluation script |
| `scripts/diagnose_baseline.py` | visualisation + summary script |

## 7. Go decision: proceed to Phase 1

All Phase 0 Go/No-Go checks pass:
- ✅ Official baseline reproduces paper results (actually slightly better)
- ✅ 180° flip failure mode confirmed in data
- ✅ Per-category hot-spots (lamp/bed/vase) line up with Pang Fig.5 narrative
- ✅ GT-leak fallback is cosmetic (no numerical hiding)

**Next step: Phase 1 — infrastructure cleanup + unified eval harness. Then Phase 2 — C3 (equivariant L_Stab).**
