# Technical report: Infralibre

## Historical submission and provenance

The challenge evaluated six application classes from header-derived statistics. Task 1 permits full-flow features, and Task 2 uses features from the first five packets. ITU confirmed third place for Infralibre.

| Historical evaluator result | Macro F1 | Accuracy | Points |
| --- | ---: | ---: | ---: |
| Task 1, full flow | 0.964258 | 0.950617 | 48.212898 |
| Task 2, first five packets | 0.935897 | 0.913580 | 46.794872 |
| Combined | | | 95.007770 |

The score belongs to the archived V9 prediction file, SHA-256 `2955aafd7655c3c7aa9d9a93b1ce24b9fd95f0de97dfd7ea012fc7cd7a59ead7`. The V9 builder changed two Task 1 predictions in the previous V8 file and retained V8's Task 2 map. The final V8-to-V9 transformation was verified byte for byte. Earlier submission iterations included manual prediction adjustments informed in part by differences in aggregate evaluator scores. The historical score is therefore a result for the submitted file, rather than a model-only estimate from an untouched test set.

## Reference model

The recovered ensemble recipe encodes transport and uses released statistics for packet counts, byte volumes, packet sizes, direction, timing and burst behavior. It adds log transforms and derived totals, fractions and bytes per packet. An explicit feature allowlist excludes flow IDs, labels and split tags from predictors. Flow IDs align predictions across views and identify output rows.

- **Prefix model:** Random Forest and Extra Trees probabilities are averaged in a 40/60 blend.
- **Full-flow model:** LightGBM, Extra Trees and CatBoost probabilities are averaged in a 35/35/30 blend.
- **Task 1:** 70% prefix probability and 30% full-flow probability after aligning the two views by flow ID.
- **Task 2:** prefix model alone. Full-flow inputs are not loaded on the prefix-only path.

Each model family uses class balancing and three fixed seeds (17, 42, 105). Maximum-log-probability assignment uses published aggregate counts to constrain the batch outputs. These counts are not per-flow labels. The reference does not ingest historical prediction files, evaluator scores or inferred test labels.

The released data contain 247 training and 83 validation flows for each view. The supplied checkpoint is a fresh refit on both public labeled splits. Evaluation on the published validation split uses a separate fit on training rows only. The split was used during earlier development and does not provide a fresh estimate of generalization. No original fitted-weight artifact was located. The recovered V4 training script emitted probability arrays and a report, without serializing its fitted estimators.

| Reference validation result | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Task 1, full flow with prefix fusion | 0.884897 | 0.939759 |
| Task 2, first five packets | 0.769793 | 0.879518 |

## Reproduction checks

Verified on Python 3.12 with the pinned dependencies. The refitted checkpoint reproduced the archived V4 JSON byte for byte across all 162 predictions. Task 2 output was identical when the entire Task 1 directory was absent. Both reference test outputs contained 81 rows with the published class totals. A separate audit replayed the recorded historical adjustments and verified the archive hashes for V6, V7, V8 and V9. The checkpoint alone produces V4 and has no official evaluator score.

- Checkpoint SHA-256: `2ba10ca05da9ba29c22fdafcc0e0a30b28b32727509e1b5665c5e9e89dae00b1`
- Reference prediction SHA-256 (both tasks, packaged checkpoint, original data): `88d0510d0628842622e3b292ac0d6b8c1a6a8e7f721195e7683a6247b735f40f`
