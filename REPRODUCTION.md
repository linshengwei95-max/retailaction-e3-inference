# Reproduction record (21-23 October 2026)

Reference: the submitted file, SHA-256 `983bc2de18d841251f1b850c1c7e5c98bdf5a16b080ec071979d30b83da98ec9` (2,512,376 bytes, 2,500 rows), Kaggle submission 57479032, public score 0.1752.

Tolerance for "same predictions", fixed before the runs: identical ids, order and action counts, and every score / start / end / coordinate within 1e-3.

## 1. Full run with this repository's code, on the lab server (CPU, no Docker there)

- Machine: lab server, torch 2.10.0+cu126 executed on CPU, 32 threads, 6 loader workers; data = the 5,000 official Challenge Test videos (checked file by file against the official file list and with sha256sum).
- Code: `infer.py` SHA-256 `1ded9e05288394a1ebe3980555e749cc9dea777b2df756728d344cb6d62149c8` (see the note below), `model.py` `5985c4b7...`, `contract.py` `4df8dd5c...`, `compare.py` `1e337bd0...`; weights `4b4c8162...dfce2e`.
- Result: 2,500 of 2,500 clips in 1,016 s. The output file is **byte-identical** to the submitted file (SHA-256 `983bc2de...98ec9`); `compare.py` reports maximum differences of 0.0 in every field.
- Log: `server_repo_test.log`, SHA-256 `9f3faf60a117c9dab4b6ecc338ffa29c845d153d200e56205d19d119e51df340` (kept outside this repository).

Note: after this run, `infer.py` was edited so that `--limit N` also writes a partial CSV (15 lines added in the `if a.limit:` branch, 1 print line removed; final SHA-256 `dee5d013a946235081e9a840757c94983e236989d5329aea7c8e596b65b74173`). The code path for a full run is unchanged. The server evidence above is for the version before this edit; the final `infer.py` was run on all 2,500 clips inside the container (section 2).

## 2. Docker image, run on a Windows PC (CPU, 13 threads)

- Built from the `Dockerfile` in this repository (base image digest `sha256:81b1e7c4...cd84a`, plus `pip install av==16.0.1`), torch 2.11.0+cpu, torchvision 0.26.0+cpu, numpy 2.1.3. Run with `--network none`, read-only data mount, `--shm-size 2g`.
- A first attempt without `--shm-size` failed with a shared-memory error in the data loader (the default 64 MB is too small); that is why the flag is in the documented command.
- **Partial run only**: the first 60 ids with the final `infer.py` (`--limit 60`), 320.8 s (about 5.3 s per clip on that PC). Compared with the first 60 rows of the submitted file: ids, order and action counts identical; maximum differences: score 1.0e-5, start 3.0e-6, end 2.4e-6, coordinates below 5e-7; not byte-identical (torch 2.11 here, 2.10 on the server). This is within the tolerance.
- **Full run in the container (23 October)**: the image was rebuilt from the final repository files (the `infer.py` inside the image has SHA-256 `dee5d013...74173`, identical to the repository version), then run on all 2,500 clips with `--network none --shm-size 2g --cpus 13`, 13 threads, 4 workers, read-only data mount. Result: 2,500 of 2,500 clips in 7,833 s (about 2.2 hours); output SHA-256 `5dd1261d871a3252890dfbef8bd4b298b26cbc85a95954fd72dfe8bd4acc15b9`, 2,512,404 bytes, 2,500 rows, 10,000 actions, strict format check passed (no placeholder violations).
- Comparison with the submitted file (`compare.py`, tolerance 1e-3): ids, order and action counts identical in all 2,500 rows; maximum differences over all rows: score 1.0e-5, start 3.0e-6, end 3.3e-6, rank0 x 6.0e-7, rank0 y 1.1e-6, rank1 x 6.9e-7, rank1 y 8.0e-7. Not byte-identical (torch 2.11 in the image, 2.10 on the server, so floating-point results differ in the last digits). Within the stated tolerance.
- Log: `run4_full.log`, SHA-256 `ee20578e3e8bd890e1a59fd0a86cb7900366cd526e82f33658a7ef28f42cb165` (kept outside this repository). An earlier container attempt on 22 October was stopped by us after 200 to 300 clips to quiet the PC; it left no output and is not part of this evidence.
- Partial outputs from `--limit` are labelled `PARTIAL_RUN_NOT_A_SUBMISSION` by the program and must not be uploaded.

## 3. What this does and does not show

Shown: the code in this repository, with the included weights, reproduces the submitted predictions byte for byte on the server's environment (version of `infer.py` before the `--limit` edit), and the Docker image built from the final repository files reproduces all 2,500 submitted predictions to within about 1e-5 in every field on a different machine and torch build.

Not shown: byte-identical output from the container (not expected across torch versions), behaviour on other hardware, or any official check by the organizers. The tolerance is the criterion used here, not an official one.
