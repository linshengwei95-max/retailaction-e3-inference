# RetailAction Challenge (PRAW @ WACV 2027): inference repository

Inference-only code for Kaggle submission **57479032** (public leaderboard score **0.1752**, status COMPLETE when checked on 2026-10-21). The public score is an online result on part of the test data; it is not a final rank.

Model: torchvision MViT-B (`mvit_v1_b`, Kinetics-400 initialisation, fine-tuned) applied to each of the two camera views, a DETR-style 3-layer decoder with four learned queries, and one output per query: a confidence logit, a normalised start/end time, and one normalised (x, y) point for each view. See `METHOD_REPORT.md`.

There is no training code, no data download, no network access and no credential handling in this repository. CPU only.

## Contents

| file | purpose |
|---|---|
| `infer.py` | the single entry point (arguments below) |
| `model.py` | model definition (unchanged from the training code) |
| `contract.py` | strict check of the `id,prediction` CSV format |
| `compare.py` | compares two submission CSVs (exact bytes, then field by field with a stated tolerance) |
| `weights/e3_model_state.pt` | model-only weights, 159,085,129 bytes, SHA-256 `4b4c816268984dc86e9bd8f3aaaa44f0216d1ec8ba306f3a8fc60f7009dfce2e`; it contains no optimizer state |
| `Dockerfile` | CPU image (see below) |
| `LICENSE`, `NOTICE` | Apache License 2.0 for the code |
| `METHOD_REPORT.md`, `REPRODUCTION.md`, `RIGHTS_AND_RELEASE.md` | method report, what was and was not reproduced, open release decisions |
| `SHA256SUMS.txt` | hashes of every other file in this repository |

The weights were extracted from the training checkpoint `epoch3_step006459.pt` (SHA-256 `8164277843a02b66c45b5bbf4f4e5cb7f0b7919f3a7a696c505db94d230d19ce`) after checking that every tensor is identical.

## Input data

Obtain the official competition files yourself under the competition rules; this code never downloads anything. Needed:

```text
<root>/sample_submission.csv                      (SHA-256 826ef8a906dd435d812e06c54f2070e5a82455d087a0f12ac6e0e93d817de34c; only its ids and row order are used)
<root>/challenge_test_set/vam-dataset-challenge_test_no_labels/data/<id>/rank0_video.mp4
<root>/challenge_test_set/vam-dataset-challenge_test_no_labels/data/<id>/rank1_video.mp4        (2500 ids)
```

Only the 5000 `*_video.mp4` files are read. The pose and metadata json files are not used by the model.

## One command (Docker, CPU)

```sh
docker build -t retailaction-e3 .
docker run --rm --network none --shm-size 2g \
  -v <root>/challenge_test_set/vam-dataset-challenge_test_no_labels/data:/data:ro \
  -v <root>/sample_submission.csv:/official/sample_submission.csv:ro \
  -v <empty output dir>:/outputs \
  retailaction-e3 --data-root /data --template /official/sample_submission.csv --output /outputs/submission.csv
```

`--shm-size 2g` is required: the default 64 MB of shared memory is too small for the data-loader workers (a run without it failed with a shared-memory error). Building needs network access once, to install `av==16.0.1` with pip; running does not. The output path must not exist (the program refuses to overwrite).

Without Docker: Python 3.11 or newer, `torch`, `torchvision`, `numpy` and `av==16.0.1`, then `python -B infer.py --data-root ... --template ... --output ...`.

## Getting the weights

The weights file is stored with Git LFS. Install Git LFS (`git lfs install`) before cloning, or run `git lfs pull` afterwards. Without it the file is a ~130-byte text pointer and `infer.py` stops with `WEIGHTS_SHA256_MISMATCH`. After cloning, `sha256sum -c SHA256SUMS.txt` should report every file as OK.

Git LFS bandwidth on a free GitHub account is limited. The same file is attached to the release 1.0: https://github.com/linshengwei95-max/retailaction-e3-inference/releases/download/v1.0/e3_model_state.pt . Download it into weights/e3_model_state.pt if the LFS download fails; its SHA-256 must be 4b4c816268984dc86e9bd8f3aaaa44f0216d1ec8ba306f3a8fc60f7009dfce2e.

## What the program does

1. Checks the SHA-256 of the weights and of the template.
2. For each id in template order: counts the frames of each view by a full decode; `n_ref = max(n_rank0, n_rank1)`; samples 16 frames per view at the same normalised times `idx_k = min(floor((k + 0.5) / 16 * n_ref), n_view - 1)` (a clip with fewer than 16 frames repeats frames); resizes each frame to 224x224 RGB, scales to [0, 1] and normalises with mean 0.45 / std 0.225.
3. Runs the model once per clip (both views), keeps all four query outputs, applies a sigmoid to the confidence, orders the four by confidence, and checks every number (finite, in [0, 1], start < end).
4. Writes `id,prediction` in template order (4 actions per id) and checks the file again with the strict format check.

Runtime: see the measurements in `REPRODUCTION.md`. Use `--threads` and `--workers` (or `RA_THREADS`, `RA_WORKERS`) to match the machine; `--limit N` runs only the first N ids and writes a partial CSV of N rows (marked `PARTIAL_RUN_NOT_A_SUBMISSION` in the program output); it is for smoke tests and must not be uploaded as a submission.

## Reproducing the submitted predictions

```sh
python -B compare.py <submitted E3 csv> /outputs/submission.csv --tol 1e-3
```

The submitted file has SHA-256 `983bc2de18d841251f1b850c1c7e5c98bdf5a16b080ec071979d30b83da98ec9` (2,512,376 bytes). It was produced on a different machine with torch 2.10.0+cu126 (CPU execution), so a byte-identical file is not guaranteed; `compare.py` reports byte identity first and then the largest field difference. The result of this comparison is recorded in `REPRODUCTION.md`.

## Limits

Internal validation losses are not the competition metric. The competition scorer was not available in an official 2027 version; the numbers in `METHOD_REPORT.md` that are marked internal come from a copy of the public scorer and are not official. The weights start from the torchvision Kinetics-400 checkpoint; no separate licence statement for that file was found, and the team decided to use it. The source code is released under the Apache License 2.0 (`LICENSE`, `NOTICE`). That licence covers the code only, not the weights: see `RIGHTS_AND_RELEASE.md`.
