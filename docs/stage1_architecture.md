# YOLOv8 Face Detection – Stage 1 Architecture & Delivery Plan

Nov 7, 2026 · @wayuu

## Executive summary

Stage 1 delivers a single-class YOLOv8 face detector trained on the Roboflow Face-Detection dataset (about 1,242 images), which replaces WIDER FACE entirely. No full training starts until the dataset audit and the data splits are approved.

The plan trains a `yolov8n` baseline at 640 in FP32 first, then runs the addendum's controlled resolution tests (416 and 800). A larger model, a P2 head and augmentation tuning are conditional follow-ups, because the training machine has only 4 GB of VRAM. Each change is tested alone on the same splits, so the result shows which change earns its cost.

The main risk is data, not architecture. About 1,242 images is small for general face detection, and nothing about its diversity or duplicate rate is verified yet. Model changes will not fix a narrow dataset, so the plan treats the audit and an out-of-distribution test set as first-class deliverables.

**Decisions needed from the project owner:** why the export holds about 2,173 images when the overview showed 1,242, which second Roboflow version to compare, the deployment latency target, and approval of Gate 0 once the environment report exists. The Open questions section lists them in full.

## Scope, constraints and assumptions

Stage 1 covers dataset audit, split design, baseline and upgrade experiments, and evaluation. It does not cover deployment, export or a production-grade dataset.

| Item | Decision | Status |
| --- | --- | --- |
| Task | Single-class object detection, class `face` | Approved |
| Framework | Ultralytics YOLOv8 | Approved |
| Dataset | Roboflow Face-Detection Computer Vision Dataset, WIDER FACE removed; image count under investigation (1,242 in the earlier overview, about 2,173 in the selected export) | Approved |
| Initial weights | Official yolov8n.pt (COCO-pretrained); the WIDER FACE checkpoint is not used | Approved |
| Training | No full training before the audit and splits are approved | Approved |
| Existing files | No change or deletion without authorization | Approved |
| Dataset diversity | Not assumed sufficient for general face detection; measured in the audit | Open |
| Training hardware and deployment latency target | Training machine: RTX 3050 Laptop, 4 GB VRAM (from the addendum, to be verified); deployment target not stated | Open |
| Environment | Existing ai-ml Conda environment only; no new environments; no reinstall of PyTorch, Torchvision or CUDA | Approved |
| Baseline settings | FP32 (amp=False), image size 640, batch 4 initially, 100 epochs, patience 20, seed 42, deterministic | Approved as a starting point |
| Data split | Original Roboflow split preserved by default; the dataset files are never renamed, moved or overwritten | Approved |

**Assumptions to confirm:** the selected export is Roboflow version 2 (folder Face-Detection.v2-datasetv2.yolov8) and a second version is still to be identified; the addendum's hardware and package versions match the real machine; AGENTS.md defines the approval gates used in the roadmap section.

## Addendum review and resolutions

Eleven points from the environment addendum were checked against this plan; five change it materially: the VRAM budget, the original split, the experiment IDs, the batch-size confound and the repository layout. I cannot reach the project machine from this session, so none of the addendum's environment values are verified by me.

| # | Addendum point | Finding | Resolution in this plan |
| --- | --- | --- | --- |
| 1 | Environment: Ubuntu, `ai-ml` Conda, PyTorch 2.13.0+cu130, Ultralytics 8.4.162 | Specific and self-consistent, but unverified here | Phase 0 runs the addendum's verification commands and records the output in `environment_report.md` |
| 2 | GPU compute capability 8.1 | RTX 30-series laptop GPUs are normally 8.6, so this value looks wrong | Measure it (`nvidia-smi --query-gpu=compute_cap --format=csv`) and record the real value; no plan change |
| 3 | 4 GB VRAM is a strict limit | My earlier draft's `yolov8s`, P2 head and 960 input could not be assumed to fit | Baseline stays `yolov8n`; 960 is dropped; larger model and P2 head become conditional on memory tests |
| 4 | Image count: 1,242 earlier, about 2,173 now | The export is 1.75 times larger. Roboflow training-set augmentation is one plausible cause; a different version is another. Neither is confirmed | Phase 1 reconciles counts per split and looks for augmented copies; no count is hardcoded |
| 5 | Preserve the selected version and its original split | Conflicts with my earlier re-split by source group | Original split kept by default; a re-split is proposed only if the audit finds leakage, as a new dataset, at Gate 2 |
| 6 | Experiments E0 (640), E1 (416), E2 (800) | My earlier E0 to E6 numbering clashed with these | Addendum IDs adopted; smoke test renamed S0; model size and P2 head become conditional E3 and E4 |
| 7 | E1 and E2 use batch 8 and batch 2 | Resolution and batch size change together, against the addendum's own one-variable rule | Report batch size beside every result; rerun at batch 4 where it fits, as a control |
| 8 | FP32 (`amp=False`), batch 4, deterministic | Conservative and consistent. FP32 raises memory use; deterministic mode can slow some operations or raise errors | Kept as the baseline; AMP becomes a separate labelled experiment |
| 9 | Class name `Face` | My draft used lowercase `face` | Class name is taken from the exported `data.yaml`, verified in Phase 1, and changed only with approval |
| 10 | Existing project and earlier WIDER FACE experiments | Must stay untouched | Repository layout revised: nothing is created inside dataset folders until approved |
| 11 | Out-of-distribution test set | Not in the addendum | Kept as a recommended addition; needs an owner decision |

## Model architecture

Start from `yolov8n` at 640 in FP32, because the training machine has 4 GB of VRAM; larger candidates are tested only if a memory test shows headroom.

| ID | Candidate | Settings | What it changes | Memory and cost |
| --- | --- | --- | --- | --- |
| E0 | Baseline | `yolov8n.pt`, input 640, batch 4, AMP off | Reference | Conservative; batch 4 is a starting point, not yet verified |
| E1 | Lower resolution | Same model, input 416, batch 8 after a memory test | Fewer pixels per face | Faster and lighter; likely worse on small faces |
| E2 | Higher resolution | Same model, input 800, batch 2 after a memory test | More pixels per face | Heaviest case in the plan; may not fit |
| E3 | Larger model (conditional) | `yolov8s`, input 640 | More capacity | Needs a memory test at 4 GB; likely a smaller batch |
| E4 | P2 head (conditional) | `yolov8n-p2`, input 640 | Adds a stride-4 head for small faces | More compute; needs a memory test |

**Why these:** the addendum fixes E0 to E2. Resolution targets small faces directly without a new architecture. E3 and E4 are added only after E0 to E2 are done and a memory test passes. The 960 input from the first draft is dropped; it is unlikely to fit in 4 GB without mixed precision.

**Rules for the experiments:**

- Start every candidate from the official COCO-pretrained weights of the matching size, never from the WIDER FACE checkpoint.
- Change one variable per run; E1 and E2 also change batch size, handled in the experiment section.
- Keep the strict FP32 baseline (`amp=False`); any mixed-precision run is a separate, labelled experiment.
- Keep the default Ultralytics augmentation as the control before any tuning.
- Watch GPU memory with `nvidia-smi` during every run, including validation.

## Data strategy and audit protocol

The audit decides which Roboflow version to use and whether the data can support general face detection at all. Each check below produces evidence in `reports/`, and none of it is assumed.

| Check | Method | Pass condition |
| --- | --- | --- |
| Version comparison | Export both versions; compare image counts, preprocessing, augmentation and split sizes | One version chosen with reasons recorded |
| Export integrity | Count images against label files; validate YOLO format | No missing, empty or corrupt files |
| Class mapping | List all class names; collapse to one class `face` | Exactly one class in the processed set |
| Box quality | Check boxes inside the image, area range, aspect ratio; spot-check a random sample visually | No out-of-range or degenerate boxes |
| Exact duplicates | File hash across all splits | Zero shared files between splits |
| Near duplicates | Perceptual hash across splits and within train | Zero cross-split near-duplicate groups |
| Augmentation copies | Group images by source (the Roboflow `.rf.<hash>` filename pattern) | Whole groups stay in one split |
| Face-size spread | Histogram of box area relative to image area | Small, medium and large faces all represented |
| Scene diversity | Sample review: pose, occlusion, lighting, crowd size, image resolution | Gaps documented, not assumed away |

**Count discrepancy.** The earlier overview showed 1,242 images; the selected export (Roboflow version 2, folder `Face-Detection.v2-datasetv2.yolov8`) is reported at about 2,173. Roboflow can add augmented copies to the training split, which would explain a gap like this, but that is a hypothesis. Phase 1 counts images, label files and faces per split, looks for `.rf.<hash>` filenames, and records the answer before any count is used elsewhere.

**Split policy.** Keep the selected version's original split by default; the dataset files are never renamed, moved or overwritten. Audit leakage first. If near-duplicates cross splits, propose a re-split by source group as a new, separately named dataset and ask for approval at Gate 2. Evaluate only on un-augmented originals, and never tune on the test split.

**Out-of-distribution set.** Collect a small hand-picked set of images from outside the dataset (crowds, profile views, low light, occlusion, small faces) with licensing recorded. It is the real measure of general face detection; Roboflow's own validation split is not.

## Pipeline and repository architecture

Existing files are never renamed, moved or deleted. New files are added beside the Roboflow export only after approval, and every run writes to its own folder. The layout below is a proposal; the real contents are inspected in Phase 0 before anything is added.

```
/home/bunleab/face_yolov/Face-Detection.v2-datasetv2.yolov8/
  data.yaml, train/, valid/, test/   Roboflow export: read-only, unchanged
  outputs/face_detection/            one folder per run; exist_ok=False, never overwritten
  scripts/                           audit and smoke-test scripts (proposed)
  reports/                           environment_report.md, audit report, run comparison (proposed)
```

**Reproducibility rules:**

- Use only the existing `ai-ml` Conda environment (`/home/bunleab/miniconda3/envs/ai-ml`); never system Python for training.
- Do not create environments or reinstall PyTorch, Torchvision or CUDA unless a real compatibility problem is found and approved.
- Record, do not change, versions: Python, PyTorch, Torchvision, CUDA runtime, cuDNN, NVIDIA driver, Ultralytics and OpenCV go into `environment_report.md`.
- Use seed 42 and `deterministic=True`; deterministic mode can slow some operations or raise errors, so check it during the smoke test.
- Save each run's arguments and a manifest of the dataset files with hashes next to its results.
- Keep the dataset version immutable: a changed dataset gets a new name, never an in-place edit.
- Log actual batch size, image size and peak GPU memory for every run in one comparison table.

## Experiment design

Experiments follow the addendum: E0 is the 640 baseline, and E1 and E2 vary resolution. Nothing beyond the S0 smoke test runs before Gate 3, and later runs are proposals, not automatic jobs.

| ID | Run | Settings | Variable changed | Status |
| --- | --- | --- | --- | --- |
| S0 | Smoke test | Short `yolov8n` run in its own folder; results discarded; checks GPU memory, loss and validation | Pipeline check only | Not started |
| E0 | Baseline | `yolov8n.pt`, 640, batch 4, 100 epochs, patience 20, workers 2, seed 42, FP32 | Reference | Not started |
| E1 | Lower resolution | 416, batch 8 after a memory test | Image size (and batch size) | Not started |
| E2 | Higher resolution | 800, batch 2 after a memory test | Image size (and batch size) | Not started |
| E3 | Larger model (conditional) | `yolov8s`, 640, batch set by a memory test | Model size | Not started |
| E4 | P2 head (conditional) | `yolov8n-p2`, 640, batch set by a memory test | Detection head | Not started |
| E5 | Mixed precision (optional) | Best configuration with `amp=True`; recorded separately | Precision | Not started |
| E6 | Augmentation tuning | Mosaic, scale and HSV settings for faces | Augmentation | Not started |

**Controls:** same splits, seed 42, epoch budget and early-stopping rule; official pretrained start; actual batch sizes and training arguments recorded for every run. A combination of winning changes runs last.

**Batch-size confound.** E1 and E2 change batch size as well as resolution, because memory differs. Where a run fits at batch 4, rerun it there as a control; otherwise report the batch size beside every result and treat the comparison as indicative. Batch 4 also gives noisy BatchNorm statistics, so watch validation stability.

**Decision rule:** a change is kept only if it improves validation and out-of-distribution results beyond seed variation, fits in 4 GB, and meets the deployment latency target once that is set.

## Evaluation and acceptance criteria

A model is accepted on out-of-distribution results and small-face recall, not on Roboflow validation score alone. Numeric thresholds are set at Gate 1, once the audit shows what the data can support; none are assumed here.

| Metric | Computed on | Why it matters |
| --- | --- | --- |
| mAP50 and mAP50-95 | Validation and frozen test split | Standard overall detection quality |
| Precision and recall at the chosen confidence threshold | Test split | Shows false alarms versus missed faces |
| Recall by face size (small, medium, large) | Test split | Tests the main reason for the P2 head and higher resolution |
| Hard-case recall | Hand-picked set: crowds, occlusion, profile, low light | Exposes the narrow-dataset risk |
| Out-of-distribution mAP50 and recall | `data/ood/` images | The nearest measure of general face detection |
| Inference speed and memory | Target hardware | Sets which candidates are usable |
| Seed variation | At least two seeds for final candidates | Separates real gains from noise |
| Peak training GPU memory | Every run, including validation | Confirms what fits in 4 GB and sets safe batch sizes |

**Acceptance conditions** (thresholds to be agreed at Gate 1):

- The out-of-distribution result is reported alongside the validation result, and a large gap between them is treated as a finding, not hidden.
- Small-face recall meets the agreed target.
- The model meets the latency and memory target on the stated hardware.
- The audit shows no unresolved cross-split leakage.

## Risk register

The two highest risks are a narrow dataset and leakage between splits; both are checked in the audit before any full training. Ratings are my initial judgment and should be revised with audit evidence.

| Risk | Likelihood | Impact | Mitigation | Check |
| --- | --- | --- | --- | --- |
| Dataset too small or narrow for general face detection | High | High | Out-of-distribution set; document gaps; plan data expansion | Audit diversity checks; OOD results |
| Near-duplicates or augmentation copies across splits inflate scores | High | High | Group by source image; re-split; freeze test set | Perceptual-hash audit |
| Overfitting on about 1,242 images | High | Medium | COCO-pretrained start; early stopping on validation; seed repeats | Train versus validation loss gap |
| Removing WIDER FACE loses crowded and small-face coverage | Medium | High | Measure small-face and crowd recall explicitly; report the gap | Size-bucketed recall; hard-case set |
| Label noise or multiple classes in the export | Medium | Medium | Collapse to one class; visual spot-check of boxes | Box-quality audit |
| No GPU available for training | Medium | High | Confirm training machine before Gate 3 | Phase 0 environment report |
| Roboflow export or licence terms limit use | Low | Medium | Record licence and source per version | Phase 1 dataset notes |
| Scope creep from tuning before the data is trusted | Medium | Medium | Approval gates; one variable per run | Gate reviews |
| 4 GB VRAM limits batch size, resolution and model size | High | High | Start with yolov8n; memory test before E1 to E4; reduce batch size on out-of-memory errors | Smoke test; nvidia-smi during every run |
| Export holds about 2,173 images against 1,242 in the overview; unexplained augmented copies | High | Medium | Reconcile counts per split; evaluate on un-augmented originals only | Phase 1 counts and .rf.\<hash> filename check |
| Addendum values (for example compute capability 8.1) are wrong or stale | Medium | Low | Verify every value on the machine; never invent missing information | environment\_report.md |

## Phased roadmap with approval gates

&#91;embedded content: roadmap · 6 phases, 5 approval gates\]

Work pauses at every gate until the project owner approves; training on the full dataset starts only after Gate 3.

| Gate | Approval criteria |
| --- | --- |
| Gate 0 | environment\_report.md produced; GPU, CUDA, PyTorch and Ultralytics verified in the ai-ml environment |
| Gate 1 | Dataset version chosen; image-count discrepancy explained; audit findings accepted; numeric metric targets set |
| Gate 2 | Original split kept, or a re-split approved as a new dataset; no unresolved cross-split leakage; data.yaml validated |
| Gate 3 | Smoke test passes with memory headroom; loss and validation work; batch size confirmed |
| Gate 4 | Results reviewed against the acceptance criteria in the evaluation section |

## Open questions and inputs required

Hardware and environment now come from the addendum, but six inputs are still open; the first two block Gate 1.

- [ ] Is the difference between 1,242 and about 2,173 images caused by Roboflow augmentation, a different version, or a different dataset?
- [ ] Which two Roboflow versions are being compared, and what preprocessing and augmentation does version 2 carry?
- [ ] What is the deployment latency and hardware target? The RTX 3050 Laptop is the training machine, not necessarily the target.
- [ ] Who runs the verification commands on the project machine and shares the output? This session cannot reach `/home/bunleab/face_yolov`.
- [ ] Are out-of-distribution images available, and under what licences?
- [ ] What numeric accuracy and small-face recall targets define success at Gate 1?
