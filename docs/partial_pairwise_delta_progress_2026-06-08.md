# Partial Uprightness Experiments: Pairwise Delta and Adaptive Hierarchy

Date: 2026-06-08

## 1. Problem Restatement

The original UprightNet method estimates upright direction by detecting support
points and fitting a support plane.  This works well on complete ModelNet-style
objects because the bottom support geometry is visible.  In our partial
single-view setting, that evidence is often missing, so the support-plane
mechanism becomes unreliable.

The goal is therefore not simply to improve a complete-point-cloud benchmark.
The goal is to design a classification-based representation that can infer
upright direction from visible partial geometry without assuming that the
support surface is present.

## 2. Fixed Hierarchy Baseline

The first successful partial model used point-wise bottom-to-top hierarchy
classification with K=5 levels.  A DGCNN segmentation trunk predicted a level
for each visible point, then upright direction was recovered by least squares:

```text
level_score_i ~= a + u dot p_i
```

Best recorded K=5 hierarchy result on partial test:

```text
point_acc = 89.83%
mIoU      = 81.28%
mean      = 5.97 deg
median    = 3.07 deg
acc10     = 88.46%
flip      = 0.59%
```

This already greatly improves over original UprightNet on the partial test set,
but its oracle revealed a bottleneck:

```text
K=5 hierarchy oracle:
mean   = 3.09 deg
median = 1.69 deg
acc10  = 94.42%
flip   = 0.00%
```

The K=5 label itself was too coarse.

## 3. Oracle Gap Diagnostics

We added oracle-gap diagnostics to compare model predictions against the best
direction recoverable from ground-truth labels under the same post-processing.

For hierarchy:

```text
predicted logits -> LS direction -> err_pred
GT level labels  -> LS direction -> err_oracle
gap = err_pred - err_oracle
```

For pairwise:

```text
predicted pair labels -> pairwise LS direction -> err_pred
GT pair labels        -> pairwise LS direction -> err_oracle
gap = err_pred - err_oracle
```

This diagnostic answers whether lower CE loss is improving direction-sensitive
structure or only improving direction-irrelevant point classifications.

Implemented in:

```text
scripts/train_hierarchical_uprightnet.py
scripts/eval_hierarchical_uprightnet.py
scripts/train_pairwise_uprightnet.py
```

## 4. Relabeling Experiments: Increasing K

To test whether hierarchy oracle was limited by coarse quantization, we added a
fast relabeling tool:

```text
scripts/relabel_hierarchy_npz.py
```

It reuses the existing partial points in `upright_hierarchy_npz` and recomputes
the level labels with a new K using the complete source mesh bbox.  It does not
regenerate partial point clouds.

Oracle results on full partial test:

```text
K=5:
mean   = 3.09 deg
median = 1.69 deg
acc10  = 94.42%

K=9:
mean   = 1.35 deg
median = 0.78 deg
acc10  = 99.39%

K=17:
mean   = 0.63 deg
median = 0.37 deg
acc10  = 99.98%

K=33:
mean   = 0.33 deg
median = 0.22 deg
acc10  = 100.00%
```

Conclusion:

```text
The data contains enough upright information.
The K=5 hierarchy bottleneck is mainly label quantization.
```

However, K=33 is not a good primary training target.  It is useful as an oracle
upper bound, but it creates many fine-grained classes, sparse levels, and an
implicit height-regression-like objective.  K=9 or K=17 are more practical if
we continue with fixed hierarchy classification.

## 5. Pairwise Sign Classification Failed as an Oracle

We first tested a pure pairwise sign task:

```text
classify pi lower than pj / same level / higher than pj
```

The oracle was poor:

```text
pairwise sign oracle, 8192 pairs:
mean   = 6.85 deg
median = 3.73 deg
acc10  = 79.20%
flip   = 0.00%
```

The reason is that sign-only ordering loses height interval information.  Many
directions can satisfy a finite set of above/below constraints on anisotropic
partial point clouds.  Sign-only pairwise classification is therefore not a
sufficient representation for accurate upright recovery.

## 6. Pairwise Delta Classification

The successful pairwise variant is relative-height delta classification:

```text
delta_ij = (h_i - h_j) / visible_height_span
target   = quantize(delta_ij, B bins)
```

The model predicts a categorical distribution over relative-height bins for
each sampled point pair.  Direction is recovered by converting the softmax
distribution into an expected pairwise margin and solving:

```text
E[delta_ij] ~= u dot (p_i - p_j)
```

This remains a classification loss.  It is not direction-anchor classification
and does not regress a 3D vector directly.

Implemented in:

```text
scripts/train_pairwise_uprightnet.py
```

Key flags:

```text
--label-mode delta
--delta-bins 9
```

Pairwise-delta oracle results:

```text
pairwise delta, 9 bins, 8192 pairs:
mean   = 0.47 deg
median = 0.28 deg
acc10  = 99.98%
flip   = 0.00%

pairwise delta, 17 bins, 8192 pairs:
mean   = 0.22 deg
median = 0.14 deg
acc10  = 100.00%
flip   = 0.00%
```

Current remote DGCNN delta-9 training result:

```text
epoch    = 033
pair_acc = 87.02%
sign_acc = 93.51%
bin_mae  = 0.14
mean     = 4.65 deg
median   = 2.41 deg
acc10    = 93.35%
flip     = 0.49%
oracle10 = 99.98%
gap      = 4.18 deg
```

This is the strongest result so far.  On partial point clouds it exceeds the
original Pang full-point-cloud Acc@10 baseline while keeping flip rate much
lower:

```text
Pang full official checkpoint:
mean   = 11.17 deg
median = 0.70 deg
acc10  = 93.03%
flip   = 5.74%

Pairwise delta-9 partial:
mean   = 4.65 deg
median = 2.41 deg
acc10  = 93.35%
flip   = 0.49%
```

## 7. Interpretation

Pairwise delta works because it combines three properties:

```text
1. It is relative, so it does not require the bottom support surface.
2. It preserves height interval information, unlike sign-only pairwise labels.
3. It uses classification loss, not direct direction regression.
```

This makes it a stronger and more realistic representation for partial point
cloud uprightness than fixed support-point detection or coarse absolute
hierarchy segmentation.

## 8. Current Limitation

Delta-9 still fixes the number of relative-height bins.  It is much better than
K=33 absolute hierarchy, but it remains a manually chosen discretization.

The next methodological question is whether we can remove the fixed bin count
without falling back to direction regression.

## 9. Future Route: Threshold-Conditioned Pairwise Classification

The proposed adaptive method is threshold-conditioned pairwise classification.

Instead of predicting one of B fixed bins, train a binary classifier conditioned
on a continuous threshold:

```text
Given pair (pi, pj) and threshold tau:
predict whether delta_ij > tau
```

Training:

```text
delta_ij = relative height or relative rank difference
tau      ~ Uniform(-1, 1)
label    = 1 if delta_ij > tau else 0
loss     = BCEWithLogitsLoss
```

Inference:

```text
Query multiple tau values.
Estimate E[delta_ij] by numerical integration of P(delta_ij > tau).
Recover upright direction by pairwise LS:

E[delta_ij] ~= u dot (p_i - p_j)
```

This removes the fixed number of layers.  The model learns a continuous
comparison function while still using classification loss.

Recommended implementation:

```text
Extend scripts/train_pairwise_uprightnet.py with:

--label-mode threshold
--threshold-samples-per-pair 1
--eval-threshold-count 17 or 33
```

Recommended target variants:

```text
1. height-threshold:
   delta = normalized visible height difference

2. rank-threshold:
   delta = normalized visible rank difference
```

Rank-threshold is more adaptive to partial visibility because it depends only
on the ordering of visible points.  Height-threshold may preserve better metric
precision.  Both should be evaluated by oracle before full training.

## 10. Immediate Experimental Plan

1. Let the current pairwise-delta-9 DGCNN run to completion.

2. Record best checkpoint metrics and compare against:

```text
Pang full baseline
Pang partial baseline
K=5 hierarchy DGCNN
K=9 or K=17 hierarchy, if trained
```

3. Add threshold-conditioned pairwise mode.

4. Evaluate oracle for:

```text
height-threshold
rank-threshold
```

5. Train the better threshold-conditioned version if oracle is at least as good
as delta-9.

6. Use delta-9 as the strong fixed-discretization baseline and threshold mode
as the adaptive method.

## 11. Current Recommendation

The main technical route should be:

```text
Pairwise delta-9 as the current best model.
Threshold-conditioned pairwise classification as the next main method.
K=33 hierarchy only as an oracle upper-bound analysis, not as a primary model.
```

This gives a coherent progression:

```text
support point classification
-> fixed absolute hierarchy classification
-> fixed pairwise delta classification
-> adaptive threshold-conditioned pairwise classification
```

