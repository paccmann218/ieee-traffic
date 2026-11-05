# Contributing

Thank you for your interest in this project. This repository is the final package of team Powerbuff Girls for the
2026 IEEE Big Data Cup, TrafficFlowBench. It has two jobs: to let anyone verify and regenerate our selected
submissions, and to document what we learned. Contributions that strengthen either job are welcome.

This guide explains what can change, what must not, and the evidence we ask for. Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md), including its research-integrity addendum.

## Contents

- [Project status](#project-status)
- [Ways to contribute](#ways-to-contribute)
- [Ground rules](#ground-rules)
- [Development setup](#development-setup)
- [Where things live](#where-things-live)
- [Coding standards](#coding-standards)
- [Experiment protocol](#experiment-protocol)
- [Verification before a pull request](#verification-before-a-pull-request)
- [Documentation and the report](#documentation-and-the-report)
- [Commits and pull requests](#commits-and-pull-requests)
- [Reporting a reproducibility problem](#reporting-a-reproducibility-problem)
- [Licensing of contributions](#licensing-of-contributions)

## Project status

- The final Kaggle submission deadline was 6 November 2026. The two selected files, `submissions/submission_TF3.zip`
  and `submissions/submission_ES4.zip`, are **frozen**. Their bytes, and the bytes of `vendor/reproduction_TF3.zip`,
  must never change. `python reproduce.py` checks their SHA256.
- During an active competition the Kaggle rules forbid sharing competition code privately outside a team. Public
  sharing is allowed. This repository is public, and contributions should stay public: issues and pull requests,
  not private messages with code.

## Ways to contribute

| Contribution | Examples | Start with |
| --- | --- | --- |
| **Reproducibility reports** | A run of `reproduce.py` on new hardware or a new OS, with timings and hashes | an issue using the *Reproducibility report* template |
| **Bug fixes** | A path, import or platform problem in `reproduce.py` or `code/` | an issue, then a pull request |
| **Documentation** | Clearer instructions, corrected numbers, better figures | a pull request |
| **Research extensions** | A new reconstruction or queue method evaluated under the protocol below | an issue describing the hypothesis and the gate *before* results |
| **Negative results** | A well-measured idea that did not work | a pull request adding it to the report's negative-results table |

## Ground rules

1. **Never modify frozen artefacts.** That covers `submissions/*.zip`, `vendor/reproduction_TF3.zip` and the checksums
   in `reproduce.py`. New results go in new files and new directories.
2. **Never commit competition data or credentials.** The release is governed by the competition rules and is not
   redistributed. Kaggle tokens, cloud keys and `.pem` files must never enter the repository (see `.gitignore`).
   Small derived summaries without raw records, such as the submission record in `docs/data/`, are fine.
3. **Respect the information boundaries** (README, *Rules compliance*):
   - Task 1 may use the whole masked layer of its own split. The target channels are absent and must stay absent.
   - Task 2 forecasts in this repository use **exactly** the twelve slots $T-60 \dots T-5$ of their own window, static
     network data and TRAIN-fitted models. A method that reads slot $T$, other days, other windows, data after $T$ or the
     offline reconstruction must say so in its title and must not replace the deployed path.
   - Task 4 solves each split on its own counts and prior.
   - Never fit to, tune on, or select with validation or private labels. They are not available, and attempting to
     infer them is prohibited by the competition rules.
4. **Label every number.** A leaderboard score comes only from Kaggle. Everything computed locally is a *proxy*,
   and it must say which instrument it came from: held-out TRAIN month, published TRAIN windows, visible-cell
   pseudo-events, or the local physics proxy.
5. **One change per experiment.** A change that touches two tasks cannot be attributed, and the leaderboard analysis
   in the report depends on single-task contrasts.

## Development setup

```bash
git lfs install                       # the archives are stored with Git LFS
git clone https://github.com/NaomiiAP/ieee-traffic.git
cd ieee-traffic
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce.py                   # must print OK three times
```

- **Python 3.12.** The pinned versions in `requirements.txt` are the inference environment. Documentation tools
  (Matplotlib, PyMuPDF, a TeX distribution) are needed only to rebuild figures and the report.
- **Data paths.** The research toolkit in `tfb/` reads `TFB_REL` (the release root) and `TFB_CACHE` (the dense-cache
  directory) from the environment. Set both before importing `tfb` modules:

  ```bash
  export TFB_REL=/path/to/kaggle_release
  export TFB_CACHE=/path/to/cache
  python tfb/cache.py D12_I5_N          # builds the dense caches of one panel
  ```

## Where things live

| Path | Contents | Editable? |
| --- | --- | --- |
| `reproduce.py` | Single entry point | Yes, carefully: keep the checksums and default behaviour |
| `code/` | The ES4 short-gap edit | Bug fixes only; the edit's logic is part of a submitted file |
| `tfb/` | Research toolkit from the competition | Yes; prefer new modules to rewriting historical ones |
| `vendor/reproduction_TF3.zip` | Frozen TF3 inference code and weights | **No** |
| `submissions/` | The two selected archives | **No** |
| `docs/report/` | LaTeX source of the report | Yes |
| `docs/scripts/`, `docs/data/` | Figure generation and its inputs | Yes; keep every constant sourced |

New research belongs in a new, dated module or directory, for example `tfb/research_20261201_ranker.py`, with
outputs written to a new directory that is never overwritten.

## Coding standards

- Follow PEP 8, use type hints on public functions, and use `pathlib` for paths. Do not hard-code absolute paths.
- Start every module with a docstring that states **what it reads** (which split and which channels) and **what it
  writes**. For Task 2 code, state the information set explicitly.
- Make every command-line tool `argparse`-based, with `--release-root`, `--out` and any seed as explicit arguments.
  Refuse to overwrite an existing output directory.
- Seed every random choice and record the seed in the output. Record input and output SHA256 for anything that
  feeds a submission.
- Keep inference deterministic. Queue outputs must be exactly reproducible, and continuous outputs reproducible to
  within $10^{-8}$.
- Do not add dependencies to `requirements.txt` without a reason in the pull request. New dependencies must be open
  source under a licence that permits commercial use (OSI-approved), as the competition rules require.

## Experiment protocol

Every research contribution follows the protocol the project used during the competition.

1. **Register before you look.** In the issue, state the hypothesis, the single change, the data used to fit, the
   data used to select, the confirmation data, and the numeric gate. For example: "joint proxy gain at least 0.0001
   on December and January, at least 7 of 10 panels improve, no panel loses more than 0.001; one confirmation on
   February."
2. **Fit on TRAIN only.** Development fits use June to November; development scoring uses December and January;
   one confirmation uses February. A final refit on all nine months comes only after the confirmation passes.
3. **No rescue.** If a gate fails, the arm is closed. Do not re-select members, weights or panels after seeing a
   failed result, and do not change the threshold afterwards.
4. **Report everything.** Add the outcome, pass or fail, with the command, source hash and numbers.

## Verification before a pull request

- [ ] `python reproduce.py` prints `OK` for TF3, ES4 and the TF3 bundle.
- [ ] Any change to `reproduce.py` or `code/` is exercised. A smoke test on synthetic data is acceptable when the
      release is not available; say which.
- [ ] If you regenerated a submission, report the row count, the number of differing cells per column, the maximum
      continuous difference, and whether queues match exactly.
- [ ] No data, credentials, caches or large generated files are staged. Check with `git status` and `git lfs status`.
- [ ] Numbers in documentation are traceable to a file, a command or a Kaggle result.

## Documentation and the report

- The README and the PDF report must agree. If you change a number in one, change it in the other.
- Figures are generated, never edited by hand: `python docs/scripts/make_figures.py`. Constants in that script name
  the research-ledger entry they come from; keep that convention.
- Rebuild the report from `docs/report/` with `pdflatex main && bibtex main && pdflatex main && pdflatex main`, then
  copy `main.pdf` to `docs/TrafficFlowBench_Final_Report.pdf`. LaTeX build artefacts are git-ignored.
- Write plainly. State results as measurements, with units and the instrument that produced them.

## Commits and pull requests

- Write commit subjects in the imperative mood and under about 72 characters, for example
  `Fix import path in recenter_short_gaps`. Explain the reason in the body.
- Keep one logical change per pull request, and fill in the pull-request template.
- A maintainer reviews every pull request. Reviews check correctness, the information boundaries, reproducibility
  and documentation.

## Reporting a reproducibility problem

Open an issue with the *Reproducibility report* template and include:

- Operating system, CPU, RAM, Python version and `python -m pip freeze` output.
- The exact command and its complete output.
- The SHA256 printed by `python reproduce.py`.
- For a regeneration, the produced `full_reproduction.json` (it contains hashes, not data).

Do not attach competition data or submission CSVs to issues.

## Licensing of contributions

By contributing you agree that your contribution is licensed under the repository's [MIT licence](LICENSE). Only
contribute work that you have the right to license.
