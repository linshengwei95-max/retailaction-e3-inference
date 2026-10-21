# Pretrained MViT-B with a DETR-style head for RetailAction localization

Author: Linsheng Wei (魏林笙), Communication University of China, linshengwei95@gmail.com. Kaggle team name: Linsheng Wei. Report prepared 21 October 2026.

Result: Kaggle submission **57479032**, official public score **0.1752** (status COMPLETE when checked on 21 October 2026). This is an online score on part of the test data, not a final ranking. The team's earlier submission 57374962 (0.0688) came from a different, much smaller model trained from scratch and is not the model described here.

## Task and approach

Each sample has two camera views (rank0, rank1). For every clip the model outputs up to K = 4 candidate actions; each has a confidence, a normalised start and end time, and one normalised (x, y) interaction point per view. Action categories are not predicted. All four candidates are written to the submission, ordered by confidence, with no confidence threshold.

## Model

- Backbone: torchvision `mvit_v1_b`, initialised from the torchvision Kinetics-400 weights (file `mvit_v1_b-dbeb1030.pth`, SHA-256 `dbeb10308e4fb5e68d793492e93ebf7a1496613a1eae0db91c194d8ffe64ffc5`) and fine-tuned end to end. The classification head is not used; all tokens after the final LayerNorm are kept: for 16 frames at 224x224, 1 class token plus 8x7x7 spatio-temporal tokens of 768 dimensions per view. The same backbone processes both views.
- Fusion: tokens are projected to 256 dimensions, a learned 3-D positional embedding (time, height, width, class token) and a learned view embedding are added for each view, and the two views are concatenated into one memory of 786 tokens.
- Head: a 3-layer transformer decoder (8 heads, feed-forward 1024, dropout 0.1, pre-norm) with four learned queries, followed by a shared two-layer MLP that outputs 7 numbers per query: a confidence logit, two interval parameters and four point parameters. Start = (1 - g)·sigmoid(a), end = start + g + (1 - g - start)·sigmoid(b) with g = 0.001; points use a sigmoid. Total parameters: 39,741,671.
- Pose and the metadata files are not used by the model.

## Training

- Data: the official training split (17,222 clips); the official validation split (1,277 clips) was used only for evaluation.
- Inputs: 16 frames per view. The frame count of each view is obtained by a full decode, n_ref = max(n_rank0, n_rank1), and both views are sampled at the same 16 normalised times idx_k = min(floor((k + 0.5)/16 · n_ref), n_view − 1) (clips with fewer than 16 frames repeat frames); frames are resized to 224x224 RGB and normalised with mean 0.45, std 0.225. No data augmentation. This sampling rule was chosen because the label times were found to lie on a 1/max(n_rank0, n_rank1) grid in the training labels; the official definition of the time base is not documented, so this is an empirical finding.
- Objective: for each clip the four candidates are matched to the ground-truth actions by exhaustive search of the lowest cost (interval L1 weight 4, point L1 weight 2, confidence cost weight 1). Matched candidates get confidence target 1, unmatched ones target 0 with weight 0.25; interval and point losses are SmoothL1 on matched pairs. Clips without actions contribute only the confidence loss.
- Optimisation: AdamW, backbone learning rate 1e-5, head 1e-4, weight decay 0.05, 200 warm-up updates then cosine decay to 10 %, gradient-norm clipping at 1.0, float32, batch 8 (micro-batches of 4 with 2-step accumulation), seed 20261018, 3 epochs = 6,459 updates, one H100 GPU, about 69 minutes. The submitted model is the checkpoint after the third epoch. A single run was made; no seeds were compared.

## Internal validation results (fixed 1,277 clips)

| quantity | value |
|---|---:|
| total training objective | 0.2075 |
| confidence / interval / points components | 0.1404 / 0.01128 / 0.01096 |
| same objective for a prediction that is constant over clips | 0.3138 |
| increase of the objective when predictions are permuted across clips (20 permutations) | +194 % |

These are values of the training objective, not of the challenge metric. For comparison, the same measurements for the model of the first submission gave no increase under permutation, i.e. its outputs did not depend on the input.

A copy of the publicly released scoring code, applied to the stored validation predictions, gives values between 0.07 and 0.22 depending on the choice of confidence-threshold grid, so absolute values from it are not used; it is only used to compare a prediction set with its permuted version, where it is higher for every grid tested (difference 0.027 to 0.125). The official 2027 scorer was not available; its equivalence with that copy is not confirmed.

## Inference and reproduction

`infer.py` reproduces the submitted predictions on CPU (`README.md` gives the single Docker command). The reproduction check against the submitted file is recorded in `REPRODUCTION.md`.

## Known limitations

- Single training run; no ablation was run on the submitted model, so no claim is made about which component helps.
- The four-candidate output is strongly biased: for most clips one query has a high confidence and the others are low (E3: the same query is the most confident for 2299 of 2500 test clips).
- The model does not use pose, and the training objective compares both points to the label, whereas the public scoring code accepts a match when either camera's point is close; the effect of this difference was not measured.
- The time-base convention (frame index divided by the larger frame count) is inferred from the training labels, not from an official definition.
- The initial weights come from the torchvision Kinetics-400 checkpoint; no separate licence statement for the weight file was found, and the team decided to use it.
- Online score and internal metrics are different things; the paragraph above should not be read as an estimate of the leaderboard score.
