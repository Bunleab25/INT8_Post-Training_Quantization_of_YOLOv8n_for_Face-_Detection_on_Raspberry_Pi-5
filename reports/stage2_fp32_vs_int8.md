# Stage 2 — FP32 vs INT8

Generated: 2026-11-07T19:11:56

INT8 is judged against the frozen Stage 1 FP32 checkpoint on the same splits.

## Accuracy gate

| Set | FP32 mAP50-95 | INT8 mAP50-95 | Drop | Budget | Pass |
|---|---|---|---|---|---|
| Frozen test | 0.6541 | 0.6549 | -0.0008 | 0.0150 | PASS |
| OOD | 0.0007 | 0.0000 | 0.0007 | 0.0200 | PASS |

- Overall accuracy gate: **PASS**

## Small-face recall (reported separately)

| Set / model | small | medium | large |
|---|---|---|---|
| test / FP32 | n/a | 1.0000 | 0.9823 |
| test / INT8 | n/a | 1.0000 | 0.9912 |
| OOD / FP32 | 0.0000 | 0.0000 | 0.0000 |
| OOD / INT8 | 0.0000 | 0.0000 | 0.0000 |

## Next step

- Accuracy gate passed. Run `scripts/bench_target.py` for Gate 5.

