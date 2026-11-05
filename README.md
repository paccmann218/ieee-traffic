# TrafficFlowBench

Final package for the 2026 IEEE Big Data Cup. Team **Powerbuff Girls**. Kaggle account `naomiandreapereira`.

Two submissions are selected. Both are in `submissions/`. `reproduce.py` is the only entry point.

| File | Kaggle ref | Submitted (UTC) | Public score | SHA256 |
| --- | --- | --- | --- | --- |
| `submissions/submission_TF3.zip` | 56825669 | 2026-10-04 13:51 | **0.86991** | `2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a` |
| `submissions/submission_ES4.zip` | 56998048 | 2026-10-09 07:13 | **0.87187** | `ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb` |

There is no private score yet. The public difference is +0.00196 for ES4. The report is `docs/TrafficFlowBench_Final_Report.pdf`.

E-mail subject: `Traffic Flow Bench Final Package – Powerbuff Girls`.

## What the two files are

Both files have 6,980,503 rows. Compared cell by cell:

| Column | Cells that differ |
| --- | ---: |
| speed | 5,825,667 |
| ongoing queue | 87,000 (every ongoing cell) |
| onset queue | 0 |
| flow | 0 |
| path flow | 0 |

TF3 is a three-member reconstruction of the masked state, with the earlier queue decoder and a split-specific demand solve. ES4 keeps that demand and that flow, replaces the ongoing queue with the same decoder refit on all nine training months, and subtracts a per-link speed bias on short temporal gaps.

The score is

```text
S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME
```

State is the masked speed and per-lane flow, averaged across the 20, 30 and 50 percent masks. Queue is the overlap of the predicted and latent queues from T+5 through T+30. Physics is read off the state field: one third fundamental diagram, two thirds conservation. Demand is a nonnegative solve against that split's own counts and prior, with penalty 0.05. The history used for the queue is T−60 through T−5. The origin cell at T is not read.

## Run

The competition release is not in this repository. `--release-root` is the directory that contains `submission_key.csv`, `corridors/` and `task2/`.

Check the archives:

```bash
python reproduce.py
```

Regenerate TF3 from the saved weights. This rebuilds the caches. It does not submit to Kaggle. The output directory must not already exist. On the clean run of 4 October 2026 every row matched, queues were exact, and the largest continuous difference was 1.82e-12.

```bash
python -m pip install -r requirements.txt
python reproduce.py --submission tf3 \
  --release-root /path/to/kaggle_public \
  --out /path/to/new_tf3_directory \
  --threads 6
```

The inference code and weights are `vendor/reproduction_TF3.zip` (SHA256 `dce93b3f2a38d27014497ec1a563cc1e2d032da11c499234f226c72093678cd7`). The entry script extracts it and runs `tfb/research_transfer_bagged_state_reproduce.py`. Plan on several hours on an 8-vCPU machine, and on enough disk for the release plus the regenerated caches.

Regenerate the last ES4 speed shell, gaps of three to six slots. The parent must be the two-slot file, not TF3. Starting from TF3 would recenter the speeds and would leave TF3's ongoing queues unchanged, so the result would not be the selected ES4 archive. The uploaded file is `submissions/submission_ES4.zip`.

```bash
python reproduce.py --submission es4 \
  --release-root /path/to/kaggle_public \
  --parent /path/to/submission_ES3.zip \
  --out /path/to/new_es4_directory
```

## Method

### State

Target cells are filled from the masked detectors, not from a copied weekly profile. The features are the nearest observed neighbors, a Treiber–Helbing smooth of those observations, ramp flows, link position, lanes, and a weekday-and-time profile fit on training days. Two boosted trees predict the speed residual and the per-lane flow residual from time interpolation: 800 rounds, 255 leaves, learning rate 0.05, minimum leaf size 150, feature fraction 0.9, row fraction 0.8, L2 penalty 2. A density is formed afterwards and pulled halfway toward the triangular diagram. A fixed free-flow correction then uses flow factors estimated from that split's own masked observations. March and April were not used to fit the trees or to choose the model.

TF3 averages three such trees with equal weight, then applies the density and free-flow corrections once. The members are a frozen parent and two refits with seeds 20261104 and 20261204. Each new member draws up to 600,000 eligible rows per panel, without replacement. The rule was locked on December and January, confirmed once on February, and only then refit on all nine training months. No member was dropped after a score was seen.

### The ES4 speed edit

On February, time interpolation between the nearest observed samples is nearly unbiased by link. The reconstructed speeds are not. ES4 subtracts, for each link, the mean of (submitted speed minus interpolant) on a gap shell.

A shell is accepted on February only when the mean absolute link bias of the interpolant is under 0.15 km/h and the correction does not raise February speed error by more than 0.02 km/h. It is written to the validation and private months only when the root-mean-square link shift is at least 0.40 km/h and no link moves by more than 15 km/h. Speeds stay inside 5 to 140 km/h.

| Shell | Speeds changed |
| --- | ---: |
| At most one slot | 1,715,267 |
| Exactly two slots | 1,918,477 |
| Three to six slots | 2,191,923 |
| Seven slots | 0 (February passed; no cells met the writing rule) |
| Eight slots and more | 0 (February bias test failed) |

The three-to-six shell moved its cells by 0.954 km/h on average, 1.246 km/h root-mean-square, and at most 6.06 km/h. Of those cells, 1,186,051 are gap 3, 671,422 are gap 4, 319,161 are gap 5, and 15,289 are gap 6.

### Queue and demand

Demand is identical in the two files: nonnegative least squares of `||A f - c||^2 + 0.05 ||f - b||^2` on each split's own counts `c` and prior `b`.

The ongoing queue in ES4 is the same neighbor decoder, refit on all nine training months. It replaced TF3's ongoing labels only after a control model reproduced them. All 87,000 ongoing cells differ. Onset does not. The forecast still uses only T−60 through T−5.

## Environment

```text
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
pyarrow==25.0.1
lightgbm==4.7.0
scikit-learn==1.9.1
```

Python 3.12. The checksum command needs only the standard library. Inference needs the packages above. No GPU is required.

## Repository

| Path | Role |
| --- | --- |
| `reproduce.py` | Entry point |
| `submissions/` | The two selected Kaggle archives |
| `vendor/reproduction_TF3.zip` | Saved weights and TF3 inference code |
| `code/recenter_short_gaps.py` | The ES4 short-gap speed edit |
| `docs/TrafficFlowBench_Final_Report.pdf` | The report |
| `tfb/` | Shared estimators used by the recenter and by the bundled inference |
| `requirements.txt` | The pinned inference environment |

The competition dataset, credentials, experiment logs and agent handoff notes are not part of this repository.

## Reading the scores

The public gains are small because the state field was already close to a strong interpolant, and the edit removes a link bias rather than replacing the reconstruction. On February the saved model is ahead of interpolation on every mask rate, including the 50 percent mask. A model fit on visible March cells and written onto the hidden holes scored 0.86725 in public, below ES4, and is not in either selected file.

Local February figures in the report are proxies. They are not Kaggle scores.
