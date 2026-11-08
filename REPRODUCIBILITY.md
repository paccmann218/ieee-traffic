# Reproducibility statement

Team Powerbuff Girls, 2026 IEEE Big Data Cup, TrafficFlowBench.

This document is the written account of how the two selected final submissions were produced and how to regenerate
them. It covers the runtime and hardware, the Kaggle Notebook position, model and dependency versions, preprocessing,
random seeds, the external-data and licence declaration, and the exact commands.

## Status at a glance

| Selected submission | Public score | Kaggle ref | Can it be regenerated from this repository and the competition release? | Has that been run? |
| --- | ---: | ---: | --- | --- |
| `submissions/submission_TF3.zip` | 0.86991 | 56825669 | Yes, end to end, using the saved model weights | **Yes.** Clean run on 4 October 2026: all 6,980,503 rows matched, queue labels exactly, continuous values within 1.82 × 10⁻¹² |
| `submissions/submission_ES4.zip` | 0.87187 | 56998048 | Yes, with one difference described below: the EQ1 classifier must be retrained because its saved weights were lost | **Not yet on the real data.** See [What has and has not been verified](#9-what-has-and-has-not-been-verified) |

**The one open item.** ES4 is TF3 plus two changes: 247 ongoing-queue labels from the EQ1 classifier, and a speed
correction on 5,825,667 cells. The EQ1 weights (`transport800.txt`, SHA256 `cb626813…`) existed only on the AWS instance
used during the competition, and that instance no longer exists. The training code, the training-window inventory and
the seeds are all in this repository, so `reproduce.py` retrains EQ1 from the release. Whether the retrained model is
byte-identical to the original is unknown until that run happens. The run's last step compares every row with the
frozen ES4 and reports the result, so any difference will be visible and counted.

## 1. The selected submissions

Both files have 6,980,503 data rows: 6,735,795 state, 174,000 queue and 70,708 ODME.

| File | What it is | Zip SHA256 | CSV member SHA256 |
| --- | --- | --- | --- |
| TF3 | Three-member state ensemble, analogue queues, exact demand | `2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a` | `487b68cf1879984e16f7155b17b8b88da6fcac24ec0a860c0f02284d85430780` |
| ES4 | TF3 → ES1 → ES2 → ES3 → ES4 | `ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb` | `05d288a825236423ea66b04d286a5ce81a27deb519eec964ff0bef2e3c613736` |

ES4 was built in single-change steps. Each intermediate file was submitted once.

| Step | Change | Cells changed | Public score | Zip SHA256 |
| --- | --- | ---: | ---: | --- |
| ES1 | Ongoing queues from the EQ1 classifier | 247 labels | 0.87030 | `f5fac2c81a5061b1fcb589bc24c59dd37b8bc3d4302baaa15c3564f3d22858c8` |
| ES2 | Speed recentering, gap 1 | 1,715,267 speeds | 0.87069 | `98e9ee0bbdb9d42a6a4bed98e99cec5f7e2ffb5cbd871fe89a10f6364b304e4a` |
| ES3 | Speed recentering, gap 2 | 1,918,477 speeds | 0.87113 | `712f772011ede5d63e1ad11d739668ee566074e5687e72224d5c7e6c55c96f71` |
| ES4 | Speed recentering, gaps 3, 4, 5 and 6, each separately | 2,191,923 speeds | 0.87187 | `ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb` |

The ES1 CSV member has SHA256 `69cbc54b22b5c38f171bee0ecfd2a7fa71237fcde3bcc198d9405ed4958188f4`.

## 2. Hardware, runtime and the Kaggle Notebook position

### Hardware used

| Item | Value |
| --- | --- |
| Machine | One AWS EC2 `r7i.2xlarge` instance |
| CPU | 8 vCPU, x86-64. Runs used 6 threads |
| RAM | 64 GiB installed |
| GPU | None. No part of the pipeline uses a GPU |
| Operating system | Ubuntu, Linux kernel 7.0.0-1013-aws, glibc 2.39 |
| Python | 3.12.3 in a virtual environment |
| Network | Needed only to install packages and download the release. Inference makes no network calls |

**Memory.** Peak RAM was not recorded during the competition. Spot checks of single running processes showed 10.4 GiB
during the state-model fit and 1.7 to 2.2 GiB during state inference. Machines with less than 64 GiB were not tested,
so 64 GiB is what was used and not a proven minimum. On Linux, the ES4 driver now records peak child-process memory
in its report.

**Disk.** The release is 7.76 GiB as a zip file (8,335,874,386 bytes). The dense caches built from it take 4.1 GiB.
The reference machine had about 70 GiB free, and free space fell by about 15 GiB across the clean TF3 run and other
jobs running at that time. We recommend at least 40 GiB free.

### Runtime

"Measured" means taken from the job record of that exact run. "Upper bound" means inferred from timestamps in the
research log and includes other work done in that interval.

| Step | Time on the reference machine | Basis |
| --- | ---: | --- |
| Verify archive checksums (`python reproduce.py`) | Seconds | Measured |
| **Regenerate TF3 from the raw release** (rebuilds all caches, then saved-weight inference for all tasks) | **93.2 min** | Measured, job `transfer_tf3_clean`, 4 October 2026 |
| Regenerate TF3 reusing existing caches and components | 44.5 min | Measured, job `transfer_tf3_staged` |
| Build dense caches, if not reused | About 10 min | Estimate: 14 of 30 files took about 3.5 min |
| Retrain EQ1: prepare windows | At most 55 min | Upper bound |
| Retrain EQ1: fit 800 trees on 3,172,453 rows | At most 20 min | Upper bound, includes evaluation |
| Build ES1 (EQ1 inference on all ongoing windows) | At most 28 min | Upper bound |
| Build ES2, ES3 and ES4 (speed recentering) | About 20 min each | ES4 at most 21 min (upper bound). ES2 and ES3 not timed separately |
| Compare the regenerated ES4 with the frozen file | About 5 min | Estimate |
| **Regenerate ES4 from the release, all steps** | **About 3 h or less** | Sum of the rows above, not yet measured as one run |
| **Regenerate both selected files** (`--submission all`) | **About 4.5 h or less** | Sum, not yet measured as one run |

For reference, one retraining stage was also timed: the final refit of the two additional state-ensemble members (four
LightGBM models, 800 rounds, six million rows) took 41.7 min, plus 4.8 min of sample preparation.

### Kaggle Notebook

**The method cannot be run inside a Kaggle Notebook as packaged, and it was never run in one.** As the organizers'
instructions allow, we provide this repository together with this written account of how it was run.

The reasons are:

1. **Environment.** Every result was produced and checked under the exact versions in `requirements.txt`, including
   NumPy 2.5.3, pandas 3.0.6 and LightGBM 4.7.0. A notebook image provides its own versions. The acceptance test here
   is exact queue labels and continuous values within 10⁻⁸, which is only meaningful under the pinned environment.
2. **Resources.** The work was run on 8 vCPUs and 64 GiB of RAM. Kaggle's standard CPU notebook is commonly described
   as 4 CPU cores with at most about 30 GB of RAM, 12 hours per session and 20 GB of saved output. Those figures come
   from secondary sources, so check [Kaggle's documentation](https://www.kaggle.com/docs/notebooks) for the current
   limits.
3. **Structure.** The pipeline is a chain of separate command-line stages, each writing hash-checked results to its
   own new directory. It also needs about 330 MB of Git LFS archives and takes several hours in total. It was built and
   operated this way from the start.

We did not attempt a notebook run, so the points above are the reasons for not using one and not a recorded failure.

## 3. Software and dependency versions

The packages are pinned in [`requirements.txt`](requirements.txt). `reproduce.py` refuses to regenerate a submission if
Python is not 3.12 or if a package version differs, unless `--allow-environment-mismatch` is given.

| Package | Version | Licence |
| --- | --- | --- |
| Python | 3.12.3 | PSF-2.0 |
| numpy | 2.5.3 | BSD-3-Clause |
| pandas | 3.0.6 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| pyarrow | 25.0.1 | Apache-2.0 |
| lightgbm | 4.7.0 | MIT |
| scikit-learn | 1.9.1 | BSD-3-Clause |

The clean TF3 run used exactly these versions (`environment.json` in the bundle records Python 3.12.3 on
`Linux-7.0.0-1013-aws-x86_64-with-glibc2.39`). The ES1 to ES4 stages ran on the same instance, but its package list was
not saved separately and cannot be recovered now.

## 4. Model versions

All models are LightGBM 4.7.0, stored in LightGBM's text format `version=v4`. The full SHA256 of every saved artefact is
in `inference.json` inside `vendor/reproduction_TF3.zip`. All were fitted on TRAIN months only, with
`deterministic=true` and `force_col_wise=true`.

| Model | Role | Objective | Trees | Leaves | Inputs | `seed` | `bagging_seed` | `feature_fraction_seed` | SHA256 prefix |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rs1_state/enhanced_v.txt` | State speed residual, member 1 | L2 | 800 | 255 | 92 | 20261001 | 6489 | 11640 | `ca0ad88e5e51fae0` |
| `rs1_state/enhanced_q.txt` | State flow residual, member 1 | L2 | 800 | 255 | 92 | 20261001 | 6489 | 11640 | `8c4512202f8409d5` |
| `rs1_state/density.txt` | Density for reconciliation | L1 | 800 | 255 | 92 | 20261001 | 6489 | 11640 | `8f1d60972381bd74` |
| `weights/sample20261104_v.txt` | Speed residual, member 2 | L2 | 800 | 255 | 92 | 20261104 | 32231 | 32562 | `dfc0f6846be63072` |
| `weights/sample20261104_q.txt` | Flow residual, member 2 | L2 | 800 | 255 | 92 | 20261104 | 32231 | 32562 | `d2f90fe84d8356a9` |
| `weights/sample20261204_v.txt` | Speed residual, member 3 | L2 | 800 | 255 | 92 | 20261204 | 25728 | 14061 | `a9eb56f102a9e81a` |
| `weights/sample20261204_q.txt` | Flow residual, member 3 | L2 | 800 | 255 | 92 | 20261204 | 25728 | 14061 | `55e5e0f74e5714d8` |
| `final_queue/transport.txt` (Q46) | Ongoing classifier, TF3 | Binary | 400 | 63 | 213 | 20261002 | 17238 | 2936 | `2816e1d3f3c23341` |
| `eq1_fit/transport800.txt` (EQ1) | Ongoing classifier, ES1 to ES4. **Weights lost; retrained by the ES4 command** | Binary | 800 | 63 | 213 | 20261002 | not recoverable | not recoverable | `cb626813e621064594f6956344df81d7d8879e65ebc23ae2f164a2f8f6f9ac9f` |

Shared settings: learning rate 0.05, minimum leaf size 150, feature fraction 0.9, bagging fraction 0.8, bagging
frequency 1 and L2 penalty 2. The three ensemble members are averaged with equal weight.

Other saved artefacts, all in the bundle: 40 temporal factor models (rank 8 and 24, speed and flow, ten panels), 10 S5
spatial covariance files with `models.json`, and 10 S12 covariance files. The 92 state input columns are listed in
`models/rs1_state/fit_manifest.json`. TF3 models were frozen by the TS3 rule
`61499049f1cced66927eff4c9e9c99bb611093c33c3a31de98ebd436940883d5`, and the final model manifest is
`8ed505b54215cd88706e95dedc5b4fbee3f435e228560f8d71d018b546cf6acb`.

## 5. Preprocessing steps

The steps run in this order. Module names refer to the `tfb/` directory inside `vendor/reproduction_TF3.zip`, or to
`code/es_chain/` for ES steps. TRAIN is June 2030 to February 2031, the validation month is March 2031 and the private
month is April 2031. There are ten corridors ("panels"). Task 2 covers the first eight of them in `cache.PANELS`.

**Dense grids (`cache.py`)**

1. Links are sorted by `order_index` in `lwr_mainline_topology.csv`, and each link must have exactly one station.
2. For each panel and split, the parquet files in `mainline_states_masked` and `ramp_states` are read. TRAIN also reads
   `mainline_states`. Target positions come from `task1/<panel>/<split>/sample_submission_state.csv`.
3. Each timestamp becomes a slot, `hour × 12 + minute // 5`, giving arrays of shape `[days, 288, links]`. Missing cells
   are NaN. Masked-layer channels are stored as `float32` (`mspeed`, `mflow`, `mocc`), with `pct_observed`,
   `is_score_eligible`, a target flag, the day's mask regime, ramp flows and ramp eligibility. Validation and private
   caches contain no target channel.

**Task 1: state (`t1_base.py`, `t1_features.py`, `research_state_model.py`, `research_state_final.py`)**

4. A cell is an *anchor* only if it is score-eligible, finite and not a Task 1 target.
5. For each link and day, speed and flow are linearly interpolated in time between anchors, and edge values are held.
   The gaps to the previous and next anchors are recorded in slots.
6. Weekday × slot × link profiles of speed and flow are computed from eligible TRAIN cells. When predicting on TRAIN,
   the day being predicted is excluded.
7. Flow is divided by the lane count. Models predict the *residual* `v − ṽ` and `q/n − q̃/n`, where `ṽ` and `q̃` are the
   interpolants.
8. The 92 inputs are built only from the masked layer, static network files and TRAIN-fitted artefacts: anchor values
   and gaps; adaptive smoothing (Treiber-Helbing, free-flow 100 km/h, congested −18 km/h, σx 0.8 km, τ one slot, ±3 slots
   and ±6 links, the cell itself excluded); ramp flows; neighbouring links at ±1 to ±3, with their leave-one-out
   innovations; box smooths; rank-8 and rank-24 temporal factors; occupancy; density; `pct_observed`; and static link
   context.
9. Output processing: speed is clipped to [5, 140] km/h and flow to at least 0. Density reconciliation applies half of
   the error-weighted projection onto `q = k·v`, with scales 1.5 km/h and 30 veh/h/lane and a density-correction clip of
   10.
10. Free-branch corrections S5, S10 (rank 12, fitted on each split's own visible flow and occupancy) and S12 are blended
    at half strength where `v > 0.85·vf` and `q < 0.9·capacity` (`research_final092_state_build.py`).

**ES2 to ES4 only: short-gap speed recentering (`code/es_chain/`)**

11. A cell's gap is the larger of its distances to the previous and next observed anchor. Separately for each panel,
    split, link and gap shell, the driver computes `δ = mean(ŝ − ṽ)`, needs at least 100 cells, and writes
    `clip(ŝ − δ, 5, 140)`, with values formatted to eight decimals. Shells are 1, 2, 3, 4, 5 and 6, and each is corrected
    on its own. For gaps 3 to 6, a shell is applied only if its link RMS shift is at least 0.40 km/h and its largest link
    shift is at most 15 km/h. No March or April label is read. Flow, queues and demand are copied unchanged.

**Task 2: queues (`research_queue_analog.py`, `research_transport_queue.py`, `research_reopened_decoder.py`, `t2_joint_prior.py`)**

12. Each forecast reads only its own window's rows, exactly the twelve slots `T−60` to `T−5`. Any row outside those
    slots raises an error.
13. Missing history is forward-filled, then backward-filled, then set to free-flow speed for speed or half of capacity
    for flow. Speed is divided by `vf` and flow by capacity, using static network parameters from `fd_parameters.csv`.
14. For ongoing windows, 213 inputs are built from those ratios (distance bands, characteristics, static context and
    calendar). The classifier gives `p(link, horizon)`, and the RQ48 decoder picks a whole mask using a TRAIN analogue
    library of 20 donor futures. For onset windows, the J1 analogue selects the mask with the highest expected IoU.

**Task 4: demand (`research_reproduce_components.py`)**

15. The path-link incidence, the split's own link counts and its prior are read from the release. The problem is solved
    as one non-negative least squares system with `λ = 0.05`: `[A; √λ I] f = [c; √λ b]`, and any remaining negative
    values are clipped to 0.

**Assembly**

16. State, queue and demand rows are merged by the organizers' `merge_submissions.py` against `submission_key.csv`.

## 6. Random seeds

Seeds that affect training:

| Component | Seed |
| --- | --- |
| RS1 state models | LightGBM seed 20261001. Row sample `random_state = 20261001 + panel index` |
| TS3 ensemble members | 20261104 and 20261204. Row sample `random_state = seed + panel index`, LightGBM seed equal to the member seed |
| Q46 training windows | `20261046 + panel index` |
| Q46 and EQ1 classifiers | LightGBM seed 20261002 |
| EQ1 additional training windows | `20261240 + panel index` |

Seed that affects inference:

| Component | Seed |
| --- | --- |
| S10 flow factor model, randomized start | `20261003 + 100 × panel index + month` |

Everything else at inference is deterministic. Seeds used inside evaluation bootstraps and `selftest()` functions affect
only reported intervals and internal checks. LightGBM's derived seeds appear in the table in section 4.

## 7. External data and licence declaration

- **External data:** none. No external dataset, pretrained model, paid or rate-limited API, or proprietary dependency
  was used.
- **Competition data:** used under the competition rules and not redistributed here. The repository contains only small
  derived summaries, such as the 2,093-window Q46 inventory (panel, date, slot and cell counts) that EQ1 retraining
  needs.
- **Software:** open source only, under the licences in section 3. Matplotlib, PyMuPDF and LaTeX were used for figures
  and the report only and are not needed for inference.
- **Our code and documentation:** [MIT licence](LICENSE).
- **Third-party code in this repository:** `repo/src/merge_submissions.py` inside the bundle is the organizers' MIT-licensed
  helper. Its notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 8. Commands

Prerequisites: Python 3.12, [Git LFS](https://git-lfs.com) and the competition release from the competition data page.
`--release-root` is the directory containing `submission_key.csv`, `corridors/`, `task1/` and `task2/`. The output
directory must not exist yet. No command uploads anything.

```bash
git lfs install
git clone https://github.com/NaomiiAP/ieee-traffic.git
cd ieee-traffic
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

# Verify the archives (standard library only)
python reproduce.py

# Regenerate TF3 from the raw release, then compare all rows with the frozen file
python reproduce.py --submission tf3 --release-root /path/to/release --out /path/to/new_tf3 --threads 6

# Regenerate ES4 from the raw release, including EQ1 retraining
python reproduce.py --submission es4 --release-root /path/to/release --out /path/to/new_es4 --threads 6

# Regenerate both selected submissions, sharing one set of caches
python reproduce.py --submission all --release-root /path/to/release --out /path/to/new_all --threads 6
```

Options for ES4:

| Option | Effect |
| --- | --- |
| `--cache-root DIR` | Reuse an existing complete set of 30 dense caches |
| `--eq1-model FILE` | Use a supplied EQ1 model instead of retraining |
| `--ongoing-from-frozen` | Skip EQ1 and copy the ongoing labels from the frozen ES4. This regenerates and checks the speed edits only. The queue labels are then **not** independently regenerated, and the report says so |

Outputs: TF3 writes `submission_TF3_reproduced.csv` and `full_reproduction.json`. ES4 writes `es1/` to `es4/` with
`submission_ES*.csv` and `.zip` files, and `es4_reproduction.json` with the stage timings, package versions, every hash
and the row-level comparison. Acceptance is the same for both: queue labels exactly equal and continuous values within
10⁻⁸. A nonzero exit code means the comparison failed.

### How the ES4 chain works

`code/regenerate_es4.py` runs these stages. The scripts in `code/es_chain/` are the originals that built each file,
copied without edits.

| Stage | Original script | Input | Output |
| --- | --- | --- | --- |
| Caches | `research_rebuild_cache.py` (bundle) | Release | 30 dense caches |
| EQ1 prepare and fit | `research_second_queue.py` | Caches, Q46 inventory | `transport800.txt` |
| ES1 | `research_second_oneshot.py` | Frozen TF3, Q46 control, EQ1 | ES1 (checks that Q46 reproduces all 87,000 TF3 ongoing labels first) |
| ES2 | `research_recenter_state.py` | ES1, caches | Gap 1 recentered |
| ES3 | `research_gap2_speed.py` | ES2, caches | Gap 2 recentered |
| ES4 | `research_gap_ladder.py` | ES3, caches | Gaps 3 to 6 recentered |
| Check | `regenerate_es4.py` | Result, frozen ES4 | Row-level comparison |

The chain starts from the frozen TF3 file, because ES1's script checks that exact parent. Run `--submission tf3` or
`all` to show that this parent is itself regenerated from the release.

### How the models were originally fitted

Regeneration uses saved weights for everything except EQ1, and full retraining of every TF3 model is not packaged as one
command. The final TS3 refit command is in the bundle's `README.md`, its protocol in `TRANSFER_PROTOCOL_S3.json` and its
method in `METHODS_TF3.md`. The fit manifests record fitting months, row counts and input hashes.

## 9. What has and has not been verified

| Claim | Status |
| --- | --- |
| Checksums of TF3, ES4 and the bundle | **Verified** by `python reproduce.py` on 8 November 2026 |
| TF3 regenerated from the raw release with saved weights | **Verified** on 4 October 2026 in a clean pinned environment. All caches and components rebuilt, all 6,980,503 rows compared, queues exact, maximum continuous difference 1.82 × 10⁻¹² |
| TF3 and ES1 differ only in 247 ongoing-queue labels | **Verified** by a row-level comparison on 8 November 2026 |
| ES1 and ES4 differ only in speed, on exactly 5,825,667 cells, the sum of the three recentering steps | **Verified** by a row-level comparison on 8 November 2026 |
| TF3 with ES4's queue column is byte-identical to the original ES1 CSV | **Verified** on 8 November 2026 (`69cbc54b…`) |
| The ES4 driver runs the original stages and produces the correct per-gap result | **Smoke test on synthetic data only**, on 8 November 2026, using the pinned package versions and temporary copies with the row-count and minimum-size checks scaled down. Output was byte-identical to a separately written calculation |
| `reproduce.py --submission es4` on the real release | **Not run.** The original instance no longer exists, and the release is not on the development PC |
| Retrained EQ1 is identical to the original | **Unknown.** Requires the run above. The ES4 report records the model hash, row counts and every differing label. The copy of `research_second_queue.py` packaged here has SHA256 `3480b917…`, and the research log recorded `85886f0b…` when the study was registered, before the fit. The difference between them is not recorded |
| Package versions during the original ES1 to ES4 runs | **Not recorded separately** |
| Independent retraining of every TF3 model from scratch | **Not done.** Saved-weight inference is what was verified |

### How the method was run

All production work ran on a single AWS EC2 `r7i.2xlarge` instance in `us-east-1`, in a Python virtual environment.
Long jobs ran as detached processes that recorded their command, log, exit status and source-file hashes
(`research_jobs.py`). Every model was fitted on TRAIN months only, and selection rules were fixed on held-out TRAIN
months before final refits. Each submitted file was uploaded once and identified by its SHA256. Task 2 forecasts read
only the twelve slots `T−60` to `T−5` of their own window. The instance and its disks were later removed from the AWS
account, so the only surviving artefacts are those in this repository and in the team's local archive.
