# TrafficFlowBench 2026 — Research Changelog

This is an honest log of everything tried, with numbers: what worked, what failed, and where the reasoning was wrong.
Competition: Kaggle `2026-ieee-big-data-traffic-flow-bench`. Scoring:
`S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`. The public LB is the validation month (Mar 2031).
**The private LB is a different, independently generated month (Apr 2031, 7 incidents vs 5), and it decides the ranking.**
The daily submission cap is 5.

---

## 0. Status (updated 2026-09-30 21:40 UTC; see section 5j)

- Best **legal** public LB: **0.71192** (R0 = GBDT state + organizer persistence + V1 ODME). 0.82669 read slot T (outside the Task 2 contract): a leaked diagnostic, **never to be selected**.
- **Scheduled 00:01 UTC Oct 1:** U1 then U2 (one change each vs R0), by `tfb/submit_u1u2_v2.sh` on EC2 (checksum-guarded; U2 is withheld if U1 ends in ERROR). Sections 5i/5j.
- **Built, NOT uploaded:** `subs/sub_Rff.csv` = R0 with ongoing rows = legal forward-fill and onset left empty (the robust final candidate; cell-checked: it differs from R0 in 2,196 ongoing cells only).
- **Built for the manual slots, NOT uploaded:** `subs/sub_U3.zip` (U2 + non-learned onset prior) and `subs/sub_L0.zip` (U2 + time-interpolation state). One change each vs U2; critic review found no blockers; questions and decision rules pre-registered in section 5j.
- `sub_safe_v3`, `sub_safe_v2`, `sub_auto_A/B`: cancelled, never submitted.
- Open work, by evidence (section 8): onset location = recurrent-bottleneck prior (section 5j); ongoing = legal forward-fill, then a CTM that must beat it; the Task 1 physics lever is S_LWR only (S_FD is saturated).

---

## 1. Infrastructure

| Item | Detail |
|---|---|
| Compute | EC2 r7i.2xlarge (8 vCPU, 61 GB), `i-0a038adb8b08dbc7e`, us-east-1, ~$0.53/h. The AWS quota is 8 vCPU standard and 0 GPU (user decision: stay CPU for now) |
| Access | `./ec2.sh 'cmd'` and `./ec2.sh --scp files... tfb/tfb/`. The user's ISP rotates CGNAT IPs, so the SG allows 152.57.0.0/16 and 157.50.0.0/16 plus current /32s |
| Code | Local `tfb/`, synced to EC2 `~/tfb/tfb/`. Official repo is at `~/tfb/repo` (evaluators used unmodified) |
| Data | EC2 `~/tfb/data/kaggle_public` (12 GB). Dense caches are at `~/tfb/cache/<panel>_<split>.npz` (`tfb/cache.py`) |
| Gotchas hit | (1) scp to the wrong dir left stale code running; (2) loading the full Task 2 table OOM-killed the box (use `load_cond` sampling); (3) `pkill -f` matched its own ssh command line; (4) the classifier outage blocked the shell for a while |

---

## 2. Data facts established (verified, not assumed)

- One station per link on every panel. **Measurement noise [corrected 2026-10-01]:** robust estimator 1.4826·MAD(second difference)/√6 on Dec–Feb (`tfb/t3_validate.py`): speed median 0.9–1.6 km/h per panel, flow 20–25 vph/lane. The earlier "1.8–2.3 km/h, 30 vph" came from lag-1 differences, which include real traffic dynamics (an overestimate). Adjacent-link flow differences (sd ≈ 190 vph) are **vehicle storage, not noise**, and not a conservation test.
- Masking removes **all** measured channels at Task 1 targets (no leak). **Ineligible cells keep their values** in the masked layer.
- Task 2 horizon cells, plus the 60 min after them, are **fully blanked** (0% visible). Window histories are republished intact in the masked layer (100% match with `window_history.parquet`).
- Queue threshold: `v_cut = 0.6 × free_speed`. fd_parameters and links.csv free speeds are **identical** on all panels. Re-verified 2026-10-01: the organizer helper `queue_utils.thresholds` (links.csv fallback, no `v_cut` column) and our `t2.statics` give the same cutoff vector on all 8 scored panels (max |diff| 0.0 km/h).
- Capacity (99.9th-pct flow) and queue frequency are similar across train, validation and private, so there is no global capacity shift. However, **validation and private ongoing windows hold about half as many queued cells** as train windows (median predicted 39 vs 80).
- Task 4: only **mainline** links are counted (no ramps), giving about 90 distinct segment counts for about 4,300 OD paths. Counts ≈ PM-average mainline flows, averaged per segment.

---

## 3. Task 4 (ODME) — SOLVED

| Variant | Public LB (other tasks = official baselines) | S_ODME |
|---|---|---|
| Official baseline (NNLS λ=.05 on TRAIN counts) | 0.55343 (anchor, matches documented 0.5534) | 0.836 |
| V2: prior × smooth distance correction + L2 | 0.47983 | 0.468 ❌ |
| V4: entropy/KL exact fit on raw prior | 0.55476 | 0.843 |
| V5: gravity seed + KL fit | 0.51071 | 0.622 ❌ |
| **V1: exact NNLS λ=0.05 on the split's OWN counts + split prior** | **0.58501** | **≈0.994** ✅ |

- V1 is byte-identical to the organizer builder's `public_reference.csv`. The public competitor used an approximate L-BFGS solve and sat near 0.80.
- Lesson: the hidden reference is essentially the organizer's own solve, so don't "improve" it. The prior's per-OD deviations carry real signal; smoothing them hurts.
- **Private-LB risk:** low. The same deterministic solve is applied to the private split's own files.

---

## 4. Task 1 (state) + Task 3 (physics)

**Local metric harness** (`tfb/score_local.py`):
- The Task 1 part is exact.
- The Task 3 part is a proxy with rhs = ΔN of the observed field. It reproduced the published baseline physics (0.3243 vs 0.3242).
- ⚠ The proxy is **not exact**: the organizer's perfect answer scores about 0.96 physics, which implies S_LWR ≈ 0.945, not 1.

| Method | Month-blocked CV (train Jun–Nov → test Dec–Feb) | Public LB contribution |
|---|---|---|
| Profile (official baseline) | S_state 0.612, phys 0.324 | 0.6929 / 0.3549 (doc) |
| Same-link time interpolation | 0.906 / 0.674 | ≈0.416 = 0.35S + 0.15P (inferred from the competitor's LB) |
| **GBDT v1 on residuals** (time-interp + ASM free/congested kernels + neighbour innovations + profile + ramps; `tfb/t1_features.py`, `t1_train.py`) | **0.9408 / 0.7292** (J = 0.4387) | **≈0.4226** (derived from probes) |
| GBDT v2 (+16 features: t±2, neighbour flow innovations, ramps t±1, link-day level, 2nd ASM) | 0.9410 / 0.7309 | not submitted |
| Density-aware decoding (blend density with interpolation) | J worse at every blend | ❌ dropped |

- ✅ Worked: the GBDT beats time interpolation. Speed RMSE 1.68 km/h vs 3.0, flow 32 vs 44 vph/lane (CV).
- ⚠ **CV→LB gap:** the CV objective J predicted 0.4387, but the LB gives about 0.4226. The +0.021 gain over time interpolation shrank to about +0.007 on the LB. It is unknown whether S_state or physics is responsible. The physics proxy may be optimistic for smooth GBDT outputs, and the profile/identity features may transfer poorly to a new scenario.
- ❌ My earlier claim of a hard ceiling (0.94 state / 0.72 physics, from the public competitor's notes) was **wrong**: the GBDT beat it.
- Speeds are clipped to ≥5 km/h (29 predictions were below 5; the observed minimum is about 5–8).

---

## 5. Task 2 (queue) — MAIN FAILURE, UNRESOLVED

### What is known about the organizer selector (reproduced on train)
- **Onset windows:** the earliest origin (chronological) with no visible queue in T−60..T−5 and a queue in T+5..T+30. The 360-min spacing is **shared across conditions**. So the first queue is **exactly at T+30** (39/40 train windows reproduced exactly; host-approved, discussion/742328).
- **Ongoing windows:** there is **no** deterministic earliest-qualifying rule. A search over 30+ candidate conditions found that ~70% of windows have a qualifying slot just before them; they are probably sampled.
- Selected ongoing windows all have **persistence-at-T IoU ≤ 0.9** (the state at T, not T−5). They sit in afternoon/evening slots and include young queues.
- Denoised labels made the onset match *worse* (23–29/40 vs 39/40), so the organizer labels ≈ raw observed speeds.
- Post-horizon data (T+95..) is weak: IoU 0.33–0.58 < persistence. It isn't used anyway (**causal-only is a user decision**).

### Local evaluations of the LightGBM queue model (`tfb/t2.py`)
| Eval | Onset | Ongoing |
|---|---|---|
| Day-grouped CV, generated windows | 0.923 | 0.926 (all) / 0.870 (hard-filtered) |
| 80 organizer-selected train windows, dates held out | 0.940 | 0.881 (0.885 hard-trained) |
| Month-blocked (train Jun–Nov → Dec–Feb) | 0.906 | 0.858 (persistence-at-T 0.665) |
| **Leave-one-corridor-family-out** | **0.413** (no_ident 0.241) | 0.781 (no_ident 0.774, no_ident_no_freq 0.752; persistence-at-T 0.677) |

### Public LB probes (Task 1 = GBDT v1, Task 4 = exact ODME)
| Submission | LB | Derived |
|---|---|---|
| v1: LGB queue (onset T+30, ongoing hard) | 0.80613 | S_queue ≈ 0.616 |
| v1 state + organizer persistence queue | 0.71192 | — |
| v1, onset windows zeroed | 0.71024 | **onset IoU ≈ 0.64**, ongoing ≈ 0.59 |
| v1, ongoing model trained unfiltered | 0.80946 | ongoing ≈ 0.61 |
| **v1, ongoing = persistence of state at T** | **0.82669** | **ongoing ≈ 0.73 (best)** |

### Experiments that did not help
- Shockwave/queue-tail velocity features: −0.0001 ± 0.0007 (paired, n=3756) ❌
- Hard-window filter: +0.004 locally, +0.022 on the LB vs the unfiltered model; still far below persistence.
- Bigger GBDT: +0.011 day-grouped. Untested under family blocking; likely memorizes more.
- Realistic masking of the day context: negligible change (0.907 → 0.905). Not the cause.

### Open question (the key one)
On validation, persistence-at-T (0.73) **beats** the model (0.59). In every local split, including leave-family-out, the model beats persistence by 0.10–0.20. **[Resolved 2026-10-01, section 5j: on the same 40 train windows, slot T adds only +0.003 ± 0.002 latent IoU over the legal forward-fill; the leaked probe's gain was the forward-fill of dropped sensors.]**
- Location memorization explains part of the onset gap.
- It does **not** explain ongoing: under family blocking the model still beats persistence by 0.10.
- Something about validation windows differs from anything in train:
  - labels (latent vs observed?)
  - the selection distribution (validation queues are about half the size)
  - the scenario contract (capacity_multiplier 1.2 in `synthetic_release_v1.json`, not visible in flow quantiles)
  - or a bug not yet found

---

## 5b. Critic review (2026-09-30) — verified findings

- ✅ VERIFIED: the contract states the ongoing selector: `competition_contract.json:137` "A window is ongoing when the visible history holds at least two queued observations with at least two on one link, and onset otherwise." I reverse-engineered for hours instead of reading it.
- ✅ VERIFIED, **rule risk**: `repo/src/task2/build_task2_persistence_submission.py:85-87`: "Persistence must use only the released [T-60, T) window history. The old implementation re-read the corridor state at T, which is not part of the Task 2 participant contract and leaked one extra observation."
  → The best LB (0.82669, persistence of the state at T) and every Task 2 model built so far use slot T. **All final Task 2 inputs must be restricted to T-60..T-5.** Rules-safe persistence (T-5) ≈ 0.60 ongoing on the LB.
- ✅ The competitor's real S_queue is ≈0.74–0.76 (their ODME was ≈0.99 as well), not 0.85+. The gap is about 0.02 total.
- Plausible, to test:
  - train features are built from the complete layer while inference uses the masked layer (the `eval_realistic` check was leaky: its models were trained on the same windows' dates)
  - labels: scoring uses the underlying noise-free state, while training uses noisy observations
  - the `known` mask at inference is unverified
- The Task 1 profile uses LOO over all train months, so month-blocked CV still sees same-scenario profiles. This probably explains the smaller LB gain.
- New validation proxy: run the contract selector on the **validation/private masked layers** and score methods on the observed cells there. This measures transfer to an independently generated month. A method must beat rules-safe persistence on both months before it is submitted.

## 5c. Post-critic actions (2026-09-30)

- Contract selector (`t2_simsel.py --contract`, `--contract2`):
  - onset 39/40 exact
  - ongoing 10/40 exact (5 before). D7_I10_W reproduces 5/5 exactly.
  - The persistence-filter variant is irrelevant (10/40 in every variant); the two-pass scan is also 10/40.
  - The remaining ongoing mismatch: the organizer fires 1–3 slots after our trigger (label-noise level). **Stopped** chasing it.
- New `tfb/t2s.py` (rules-safe rebuild):
  - inputs only T-60..T-5, never slot T
  - train rows built exactly like validation (masked day layer + republished history)
  - ongoing windows = first 1–6 slots after the contract trigger
  - proxy on validation/private pseudo-windows (e.g. D7_I10_W validation: 26 onset / 108 ongoing pseudo-windows vs 10 real)
  - Methods are compared to T-5 persistence on both months.
- Auto-submit A/B cancelled (see §0).

## 5d. Proxy validation and the scenario-shift fix (2026-09-30)

- **Pseudo-window proxy FAILED its backtest.**
  - validation: old model 0.768 vs persT 0.728 on the proxy, but the LB says 0.59 vs 0.73
  - Filtering by persistence ≤ 0.9 does not fix the ordering.
  - **Not admissible** (critic rule c).
- Eligibility-mismatch hypothesis refuted: horizon `elig` flags == pct≥75 (76.4%).
- **Episode backtest** (`t2s.episode_backtest`):
  - pseudo-windows 6–8 slots before each real ongoing origin, so the horizon is inside the republished complete history (fully observed labels) and they sample the same queue episodes as the LB
  - old model vs persT: train 0.848 vs 0.773; **validation 0.696 vs 0.685; private 0.721 vs 0.705**
  - The model's edge vanishes on new-scenario episodes, which is directionally consistent with the LB. n = 120 per split, SE ≈ 0.03.
- **Multi-scenario training** (`t2s.scenario_test`, rules-safe, private episodes):
  - persTm5 0.687
  - train-only full 0.719 / no-freq 0.717
  - **train+validation full 0.765 / no-freq 0.753**
  - Validation episodes, train-only: no-freq 0.704 > full 0.684 > persTm5 0.665.
  - The gain is invisible on the public LB (validation predictions must be train-only).
- Critic round 2 (reasoning errors + decision rules) adopted:
  - read the docs first
  - tag claims with evidence type and n
  - a proxy must reproduce the LB ordering
  - ship only if a change beats the rules-safe baseline on both months with a paired SE
- Rules risks now tracked:
  - slot T (organizer says not allowed)
  - day context before T−60
  - training on validation for private
  - A forum question has been drafted for the user.

## 5e. Task 1 proxy validated; Task 2 proxies exhausted (2026-09-30)

- **Task 1 new-scenario proxy** (`tfb/t1_proxy.py`, re-mask 10% of observed eligible val/private cells). S_state:

  | Method | Validation | Private |
  |---|---|---|
  | profile | 0.686 | 0.697 |
  | tinterp | 0.902 | 0.906 |
  | GBDT | **0.932** | **0.934** |

  **Admissible**: profile ≈ LB baseline 0.693; tinterp ≈ the competitor's LB-derived ~0.909. The GBDT transfers (+0.030 S_state; speed RMSE 1.95 vs 2.72). The smaller LB gain is therefore likely physics, not state.
- **Task 2 episode proxy with the contract trigger required** (n≈88): persTm5 0.93 (private) vs LB ≈0.60, so it is NOT calibrated (the pre-origin episode phase is too static). No admissible local Task 2 ongoing proxy exists yet; **the public LB (80 scored Task 2 windows, 40 per condition; corrected from "40") is the only admissible Task 2 evidence for learned models.**
- Multi-scenario training is consistently better in every Task 2 variant:
  - private: trva 0.898 vs tr 0.831
  - earlier: 0.765 vs 0.719
  - The holed-history control (trm 0.830 ≈ tr) shows it is **not** a blank-handling artifact.
  - It stays unverifiable on the public LB, and its legitimacy is pending the forum answer. **[Retired 2026-09-30, section 5i: fitting March to predict April is cross-split fitting. Do not resume.]**

## 5f. Digital twin, Task 1 target-month adaptation, physics hypothesis (2026-09-30)

- **Twin feasibility (train):**
  - Congested cells sit on the released triangular FD: 96–99% within 5 km/h, median deviation ≈1 km/h (noise). Free flow ≈ v_f ± noise.
  - ~~Adjacent no-ramp flow differences have mean ≈0, with sd matching the noise level.~~ **[Corrected: wrong equation. LWR stores vehicles on a link; the sd is ≈190 vph (storage), far above the ~20 vph/lane noise. Not a conservation test.]**
  - Capacity use is identical across splits.
  - So the generator is an LWR/CTM on the released FD.
- **CTM twin** (`tfb/twin.py`, rules-safe, history-calibrated boundaries and bottlenecks):
  - v0 ongoing 0.652, v1 (+ demand trend) 0.648, vs persTm5 0.748 on the 80 train windows. ❌ Not competitive yet: it over-holds queues and misses dissipation.
  - Research continues, but nothing is shipped from it.
- **Hole-filling** (latent queues are contiguous): persTm5 0.726 → **0.748** against a denoised-truth proxy (but 0.748 → 0.733 against noisy labels). ~~Adopted in the rules-safe decoder.~~ **[Not adopted in U1/U2: that proxy hole-fills its own labels, so it is biased toward hole-fill.]**
- **Task 1 target-month adaptation** on the validated proxy (S_state, validation/private):
  - base 0.9337/0.9323
  - own-month profile 0.9357/0.9349
  - self-supervised + own profile 0.9368/0.9358 (+0.003 ≈ +0.001 total). **[Retired, section 5i: one model pooled March and April cells (cross-split fitting); physics was never measured for it.]**
- **Physics "latent reference" hypothesis REFUTED:** with a smoothed reference even the truth scores S_LWR 0, so the organizer reference ≈ the observed field. **[Superseded by section 5j: the local scorer is now validated against the official scorer with a written boundary-flux file, and the ceiling is re-measured with the conditional-mean control.]**
- Building `sub_safe_v1` (not submitted):
  - T1 self-supervised
  - T2 rules-safe: onset full; ongoing no-freq; validation predictions train-only, private predictions train+validation; hole-fill
  - exact ODME
  - It must pass the gate (critic review) before upload.

## 5g. Gate review of the compliant candidate (2026-09-30)

- The critic's gate review of `sub_safe_v1`:
  - no same-cell leak in Task 1 self-supervision
  - train/inference parity OK
  - **expected public ≈0.80 ± 0.015 < 0.82669**, so it does not raise the public LB
  - It is justified only as the compliant baseline for the final.
- **Day-context features stripped** (same-day data before T−60 is a contract risk, like slot T). This gives `sub_safe_v2`.
- **Leak found and NOT used: ramp flows are not blanked inside Task 2 horizons** (79.5% finite, the same as history). Our Task 2 never reads ramps. A forum post was drafted for the user to report it.
- Horizon mainline speeds are verified 100% blank, including ineligible cells.
- Plan: one submission of the rules-safe model ongoing (B). A (rules-safe persistence) is already known from probes: ongoing ≈0.60 on the LB. B − A is the only admissible ongoing evidence.

- **Onset decoders** (validation/private onset pseudo-windows, n≈290):
  - expected-IoU **0.692/0.725** (best)
  - block (grow from argmax) 0.62/0.67
  - top-k fixed size 0.42–0.52
  - expected-IoU + hole-fill 0.686/0.710
  - → keep the expected-IoU decoder; hole-fill for ongoing only. ❌ size-aware decoding killed.

- **`sub_safe_v2` built** (fully rules-compliant; day context stripped; T1 self-supervised; exact ODME). Sanity: 6,980,503 rows, 0 nulls, speeds 5–140.
  - **Scheduled: 1 submission at 00:01 UTC 2026-10-01** (the Sep 30 quota was used).
  - Purpose: the decision-critical measurement of the compliant ongoing model vs compliant persistence (≈0.60 known).
  - Expected public ≈0.80 ± 0.015, below 0.82669 by design; the public-best relies on slot T.

## 5h. Closing the Task 2 gap to the pack (2026-09-30 PM)

- LB position: ~#92 of ~200+ (public 0.82669). About 90 teams are ≥ 0.827. A typical 0.87 implies **S_queue ≈ 0.80**, vs ours 0.684 (slot-T) or ≈0.62 (compliant). ~~The gap is Task 2.~~ **[Corrected: that arithmetic assumes the teams above have our state/physics. Section 5j measures the physics headroom directly (small but nonzero); Task 2 remains the largest legal lever.]**
- **Physics growth decoder** (`tfb/growth.py`, 80 train windows, nothing learned; honest transfer estimate):
  - empirical tail/head trend vs persistence: +0.005 ± 0.009 (latent proxy)
  - kinematic-wave shock speed: **−0.09 to −0.15** ❌
  - Queues mostly persist over 30 min.
- **Competitor code comparison** (`happyc0der` task2_predict / onset2_predict): structurally identical. Differences:
  - They use ineligible cells' values.
  - They train on ineligible cells.
  - Their decoder is over all cells.
  - Onset augmentation k=4,5.
- **Including ineligible cells' values** (`T2S_INELIG=1`), onset proxy: 0.692/0.725 → **0.703/0.732**. A small gain on an uncalibrated proxy. **[Not adopted: the onset proxy is inadmissible (section 5i, critique 5.5); ineligible values stay legal *features* of the ongoing rule only.]**
- Competitor's LB-derived S_queue ≈0.765, split between onset and ongoing unknown. Our onset LB ≈0.64 (proxy 0.69–0.73), our compliant ongoing unknown until `sub_safe_v2` scores.

- **Hybrid ongoing decoder** (`t2s.hybrid_test`, 80 shipped train windows, model trained without those dates, rules-safe features with ineligible cells). Latent-proxy IoU, paired vs persistence-fill 0.748:

  | Decoder | IoU | Δ vs persistence-fill |
  |---|---|---|
  | model alone | 0.789 | +0.040 ± 0.025 |
  | **add-only hybrid: persistence(T−5) + fill ∪ {p ≥ 0.5}** | **0.804** | **+0.055 ± 0.021** |
  | variants that remove cells | 0.772–0.789 | worse |

  ~~This matches the LB evidence: the pure model hurt by dropping persisted cells. Adopted.~~ **[Retired: chosen on training windows whose queues are ~2x the March/April size; add-only cannot represent clearance.]**
- `sub_safe_v3` = T1 self-supervised + T2 (ineligible-inclusive, onset expected-IoU, ongoing add-only hybrid) + exact ODME. **[Cancelled before upload, never submitted: a 5-change bundle with cross-split fitting.]**

## 5i. Response to CRITIQUE.md, plan v4 (2026-09-30 evening)

**Corrections of earlier claims in this file:**
- **Window count.** The public LB has 8 scored panels × 10 windows = **80 windows**, 40 per condition, not "40 per corridor split".
- **Noise level.** Per-window IoU sd is **0.375** (release json), so SE(S_queue) ≈ 0.04 and SE per condition ≈ 0.06.
- **"Neighbour-flow sd matches noise" was the wrong equation.** LWR allows storage between links, so adjacent flows are not copies.
- **Persistence at T is not causal.** Slot T is outside the Task 2 contract (`build_task2_persistence_submission.py:85-87`). **0.82669 is a leaked diagnostic and will never be selected.**
- **§7 recommended "persistence-at-T". That recommendation was wrong.**
- **Self-supervision pooled March and April cells in one model** (`t1_final_ss.py`). That is cross-split fitting. Retired.
- **Private rows were trained on March** (`t2s make_sub`). Retired.
- **Cancelling `sub_auto_A` (no-identity) was a mistake.** The no-identity variant becomes part of the next A/B (B2).
- **CTM v0/v1** were unfinished calibrations (discharge/over-holding bug), not a test of the model class.
- **`sub_safe_v3` was a 5-change bundle.** Cancelled (never submitted).

**Verified premises** (computed 2026-09-30, read-only):
- **No label leak.** Task 1/2/4 templates and `sample_submission.csv` hold NaN/0 values, and the key orders queue rows naturally.
- **Physics ceiling for any method that cannot know the measurement noise.** An ideal latent predictor (truth + fresh noise at the measured per-link sd) scores local S_phys **0.735–0.787**. The GBDT scores 0.725–0.74 and time interpolation 0.67 (3 panels, Dec–Feb).
  - The 0.96 "perfect" physics contains the noise, so the critique's physics-headroom claim is **rejected**: at most ≈ +0.002–0.008 total. **[2026-10-01: that experiment had not been validated against the official scorer and lacked the conditional-mean control; section 5j repeats it properly. The conclusion survives, with a measured ceiling of ≈0.76–0.78 local S_phys.]**
- **State ceiling.** Measured noise sd is speed 0.9–1.7 km/h and flow 17–31 vph/lane, which gives an ideal S_state ≈ 0.94–0.965.
- **Honest ceiling:**
  - ~~total ≈ 0.636 + 0.30·S_queue, so even a perfect queue forecast is ≈0.94~~ **[Corrected in section 5j: 0.636 froze S_physics at the GBDT's 0.73; with the measured ceiling on the LB scale the non-queue maximum is ≈0.64, so S_total ≤ ≈0.64 + 0.30·S_queue.]**
  - > 0.95 is not reachable without the hidden noise or labels.
  - ~~the #1 score (0.924) implies S_queue ≈ 0.96, most likely using out-of-contract inputs~~ **[Retracted: 0.924 needs S_queue ≈ 0.93 with state and physics at their noise ceilings. That is very high, and it says nothing about how any other team got there.]**
  - realistic legal target: 0.85–0.88

**New legal result: persistence done right** (`tfb/t2_persist.py`, 40 released train ongoing windows, non-learned so the result transfers):

| Ongoing variant | IoU (noisy labels) | Paired Δ vs organizer |
|---|---|---|
| Organizer persistence (T−5, eligible only) | 0.590 | — |
| Forward-fill last eligible history obs | 0.748 | +0.158 ± 0.019 |
| **Forward-fill incl. ineligible values** | **0.755** | **+0.165 ± 0.019** |
| Forward-fill incl. ineligible + hole-fill | 0.734 | +0.144 (latent proxy +0.179; that proxy hole-fills its own labels, so it is biased) |

- The organizer rule drops any link whose T−5 cell is missing or ineligible (≈24% of cells).
- Much of the old "slot-T persistence 0.73" was the forward fill, not slot T.

**Scheduled 00:01 UTC Oct 1: two one-change uploads** (`tfb/build_u1u2.py`, `tfb/submit_u1u2.sh`):
- **U1** = R0 (0.71192) with legal onset rows.
  - The model is a train-only LGB on history T−60..T−5, with no day context, h=6 only.
  - Q: onset IoU = (LB−0.71192)/0.15.
- **U2** = U1 with ongoing = forward-fill incl. ineligible.
  - Q: gain = (LB_U2−LB_U1)/0.15. Expected ≈ +0.16 IoU ≈ +0.024 total.
- Checks:
  - R0 queue reproduced exactly (8406 ones)
  - information-set assert passed (12 windows)
  - U1 ongoing rows identical to R0
  - 6,980,503 rows, 0 gaps/nulls
- Critic pre-flight: **no blockers**. Its mandatory checks were run:
  - **U1 vs R0:** 0 state and 0 ODME differences; 225 queue cells changed, all in onset windows.
  - **U2 vs U1:** 0 state and 0 ODME differences; 2196 queue cells changed, all in ongoing windows.
  - The train history values read in the evaluation equal the masked-layer values read at submission (145,224 cells, 0 differing).

## 5j. Critique round 3 (2026-10-01): measurements, corrections, decisions

Scripts: `tfb/t2_audit.py`, `tfb/t3_validate.py`, `tfb/t2_onset_rules.py`, `tfb/build_robust.py`, `tfb/build_next.py`. Everything below is measured, not inferred.

**Pre-flight fixes (the uploads themselves were cleared by the critique):**
- Queue threshold: the organizer helper and our code give identical cutoffs on all 8 scored panels (max |diff| 0.0 km/h). The U2 gain is not a threshold artifact.
- Scheduler replaced by `submit_u1u2_v2.sh`: absolute target 2026-10-01 00:01 UTC (a restart no longer waits a day); zips checksum-guarded; **U2 is withheld if U1 ends in ERROR**; a failed submit call is retried only if the file is not already listed.
- `build_u1u2.py`: the R0 check is now cell-level and keyed through `submission_key.csv` (it catches a single flipped cell in a negative control); midnight-crossing assert added (dormant on the 240 published windows).
- Robust final candidate `sub_Rff` built and **not uploaded**: it differs from R0 in exactly 2,196 ongoing cells and from U2 in exactly 225 onset cells.

**Task 2 ongoing: the full persistence table** (40 released train ongoing windows; nothing fitted; the DIAG rows read slot T and exist only to measure the leak):

| Variant | Noisy IoU | Δ vs organizer | Latent-proxy IoU | Δ vs organizer | Δ vs ffill_inel (latent) |
|---|---:|---:|---:|---:|---:|
| organizer (T−5, eligible) | 0.590 | — | 0.569 | — | −0.164 |
| ffill | 0.748 | +0.158 | 0.726 | +0.157 | −0.007 |
| **ffill_inel (U2)** | **0.755** | **+0.165 ± 0.019** | **0.733** | **+0.164 ± 0.019** | 0 |
| ffill_fill | 0.733 | +0.143 | 0.748 | +0.179 | +0.015 (proxy is hole-filled, biased) |
| ffill_inel_fill | 0.734 | +0.144 | 0.748 | +0.179 | +0.015 (same caveat) |
| mean2 | 0.725 | +0.136 | 0.704 | +0.135 | −0.029 |
| mean2_fill | 0.720 | +0.131 | 0.736 | +0.168 | +0.003 |
| DIAG slot T only (eligible) | 0.613 | +0.023 | 0.593 | +0.024 | −0.140 |
| DIAG slot T, forward-filled (= the 0.82669 probe) | 0.761 | +0.171 | 0.736 | +0.168 | **+0.003 ± 0.002** |

- **The leak was the fill.** On the same windows, reading slot T adds +0.006 ± 0.003 (noisy) and +0.003 ± 0.002 (latent) over the legal forward-fill. This is an upper bound: the DIAG row reads slot T from the complete train layer, while the LB probe read the masked layer, where slot T is partly blank. The leaked probe's gain came almost entirely from not dropping sensors whose T−5 reading was missing or ineligible.
- ffill_inel is the best rule on noisy labels, and the best on the latent proxy among the variants that do not hole-fill. mean2 is worse on both.
- **Missing last reading, per split** (released layer, share of links whose T−5 cell is missing or ineligible): train 0.239/0.233, validation 0.229/0.235, private 0.233/0.238 (ongoing/onset). Raw NaN share 0.16 everywhere. Links flipped by ffill_inel per ongoing window: train 8.8, validation 4.8, private 4.3; queued links +32% / +26% / +26% relative to the organizer rule. The critique's condition for trusting the U2 expectation (equal rates across months) holds.
- **The history ends at T−5.** The index `history_coverage` equals the eligible fraction of [T−60, T−5] on 240/240 windows (all splits); shifted to include T it matches 0–1.3%.
- Expected U2 − U1 ≈ +0.15 × (0.13 to 0.16) ≈ +0.019 to +0.024 total. U2 answers only this question.
- **Forward-fill by horizon step** (40 train ongoing windows): IoU 0.894, 0.861, 0.820, 0.775, 0.726, then **0.542 at T+30**. That drop is not a new recurrent queue: only 0.55 links per window newly queue at T+30 (10/40 windows), and queues clear in 12/40. Adding prior links at T+30 gains +0.005 to +0.010 (≤ 1.6 SE). The remaining ongoing error is clearance and growth dynamics, which is the CTM's job.
- **Prior-modulated forward-fill** (drop historically rare links / add frequent neighbours; frequency leave-one-month-out): dropping hurts (−0.015 to −0.030; ongoing queues often sit where queues are rare, i.e. incidents); adding frequent neighbours +0.011 ± 0.005 latent (best of 3 thresholds, ≈ +0.0016 total). Not worth a submission.

**Task 2 onset: location is a recurrent schedule, not a state visible in the legal hour** (40 released train onset windows):
- The true onset queue is a median of 3 links, first visible at h = 6 on 39/40 windows.
- At T−5 the links that queue at T+30 sit at speed ratio **1.76** (free flow; all links 1.66). Link-level physical rules (lowest speed ratio, hour minimum, flow/capacity, slowing trend, slow downstream link) score **IoU 0.00–0.05**; the best rank of a true link in those rankings is a median of 18–56.
- The train frequency prior (queue rate at T+30 by link and weekday type) puts a true link at **rank 1** (median). Top-3 scores 0.549 latent / 0.651 noisy leave-one-day-out, and **0.558 / 0.621 leave-one-whole-month-out** (paired +0.009 ± 0.017). The recurrence transfers to an unseen month.
- The 40 windows sit on only **9 distinct dates** (4–5 per panel), so the effective n is below 40 and window-level SEs are optimistic. The leave-one-month-out test is within the training scenario; the cross-seed evidence is the public v1 onset ≈ 0.64 (derived, from a model that also saw slot T). Nit: the `dr` signal in t2_onset_rules.py compares T−5 with T−35, not T−30.
- **More onset rules** (`tfb/t2_onset_prior.py`, top-3 at h=6, every prior leave-one-month-out; latent IoU, date-clustered SE): marginal prior 0.558; onset-conditional prior (frequency of being part of a new queue after a queue-free hour) 0.552 (−0.006 ± 0.060); flow over the **downstream** link's capacity 0.064 (the best physical rule, still far below); upstream flow over own capacity 0.021. The marginal prior stays.
- The published clue ("an hour before a queue forms the road is already at 1.28x its queue threshold ... 6912 vph against 4620") is a road-level statement about *whether* a queue forms. The scored onset windows already guarantee that one does. It does not say *where*.
- **U3 defect fixed before any upload:** 7 of 80 validation/private onset windows have fewer than 3 links with a positive frequency at T+30 (four D7_I10_W validation windows have none), so top-3 picked arbitrary links. Ties are now broken by the ±30 min frequency, then the all-day frequency. This changed 38 cells, all in those 7 windows; U3 vs U2 is 0 state rows, 187 onset queue rows, 0 ODME rows.
- **U1's tree is mostly this prior:** 64–70% of its predicted links are the prior's top-k, 0–6% are among the slowest links, and its links sit at free-flow speed (ratio 1.66–1.72). U3 (the pure, non-learned prior; built, not uploaded) shares about 60% of its cells with U1.
- Consequence: the critique's "no-identity or physical onset" next step is closed by measurement. The next onset question is the non-learned prior against the tree (U3 vs U2).
- **What limits the prior is size, not location:** its top-1 link is a true onset link in 80% of windows, and 93% of true links are in its top-10. Top-3 = 0.621 noisy, while an oracle size gives 0.759 (+0.14). The loss sits in the 6 windows with > 5 onset links (0.36 vs 0.80). Corridor flow level barely predicts size (corr 0.11). Adaptive-size decoders on the prior (5 variants; same windows): relative mass f ≥ 0.7·f_top +0.125 ± 0.065 noisy / +0.049 ± 0.054 latent; expected-IoU and upstream-chain variants are worse. Selected from 5 on 40 windows, so it is a hypothesis for a later single-change A/B, not an adoption.

**Task 3: the local scorer is implementation-equivalent to the official one, and the ceiling is re-measured with the right controls** (`tfb/t3_validate.py`; train Dec–Feb, official hash mask R1/R2/R3):
- `make_flux()` writes a `--boundary-flux` file projected onto the observations. With it, `repo/src/task3/score_task3.py` and `score_local.T3` give **identical** S_FD, S_LWR and S_physics on 8 fields × 4 panels (max |difference| 0.0000). Because make_flux constructs rhs = ΔN_obs, the equality is by construction for that flux: it verifies our field composition, filters and aggregation, **not** the organizer flux (critic, 2026-10-01).
- **S_FD is saturated.** The exact observations score S_FD 0.986 (D7_I10_W), and every candidate field scores 0.985–0.991. Projecting any field onto the released triangle at constant density (N and S_LWR unchanged) raises S_physics by only +0.0015 to +0.0024, and costs about +1.3 km/h speed RMSE. **The density-aware / FD-projection decoder is closed with evidence.**
- The spec's "honest S_FD 0.8833" is the pre-anchoring value (SCORING_SPEC says so). Reading the published exact-answer 0.9636 as S_FD 0.8833 with S_LWR 1 is arithmetically impossible: (0.8833 + 2)/3 = 0.9611 < 0.9636. With the measured S_FD ≈ 0.986, 0.9636 implies the exact answer scores S_LWR ≈ 0.95 against the organizer flux. BASELINES.md says the same ("about 0.95 is the practical ceiling"). So the organizer's right-hand side is close to, but not exactly, ΔN of the observations. The repo confirms it: README.md:282 and RUN_LOCAL.md:60 record a perfect answer scoring **0.9598 officially on D12_I5_N**, which with S_FD ≈ 0.986 means an exact-answer S_LWR ≈ 0.947. The local number is therefore optimistic by about 0.05 S_LWR. That it preserves the ranking of methods is an **assumption**: the ~5% residual (plausibly latent ramp or boundary values) could favour smoother fields.
- Predictor ceiling, with the critique's missing conditional-mean control (local = official):

| Field at the targets | D7_I10_W | D7_I210_W | D12_I5_S | D7_I405_S |
|---|---:|---:|---:|---:|
| exact observations | 0.995 | 0.994 | 0.996 | 0.995 |
| observations + fresh noise (old "ideal") | 0.768 | 0.775 | 0.789 | 0.755 |
| **denoised, 3-slot mean (conditional-mean proxy)** | **0.770** | **0.762** | **0.771** | **0.767** |
| time interpolation | 0.680 | 0.668 | 0.674 | 0.675 |

  The two controls differ by +0.002, −0.013, −0.018, +0.012 (mean −0.004): neither is systematically higher, so the predictor ceiling is ≈0.77–0.79 (local scale) whichever control is used. The cleaner argument (critic): the latent (conditional-mean) field and obs+fresh-noise are both one noise σ from the observation at every target cell, so they have the same expected S_LWR against ΔN_obs; and S_FD is saturated. (`denoise3` is not a true conditional mean: it includes the target's own observation, and `np.roll` wraps slot 287 to 0. Neither changes the conclusion.) Per regime, S_LWR for obs+noise is 0.78 (R1), 0.69 (R2), 0.51 (R3); the loss is structural in R3.
- Noise sd (robust second-difference estimator): speed median 1.31 / 1.49 / 0.90 / 1.56 km/h, flow 21.2 / 21.0 / 19.8 / 25.3 vph/lane on these panels. This replaces the lag-1 figures in section 2.
- **Corrected ceiling arithmetic (leaderboard scale):** the local ceiling of ≈0.77 must be moved to the LB scale, where even the exact answer loses ≈0.035 S_physics, giving ≈0.73–0.75. Non-queue maximum ≈ 0.35·0.955 + 0.15·0.74 + 0.20·0.994 ≈ **0.64**, so S_total ≤ ≈ 0.64 + 0.30·S_queue, and 0.924 needs S_queue ≈ 0.93. The GBDT's month-blocked S_physics of 0.729 leaves about 0.04 S_physics (≈ 0.006 total) to the noise ceiling. That headroom is real and small, and it comes only through S_LWR, i.e. lower density error at target cells.

**Pre-registered questions for the next manual uploads** (reviewed by the critic 2026-10-01: no blockers):
- **U3** = U2 with onset = the tie-fixed prior (EC2 `subs/sub_U3.zip`, md5 9147dd3b473aaa20c14fe536de38a83c). Q: (LB_U3 − LB_U2)/0.15 = onset IoU(prior) − onset IoU(U1 tree). Expectation ≈ 0 (about 60% of cells shared). Decision: keep the non-learned prior unless it is clearly worse. That is a robustness choice for April, not a noise band (the paired SE is about half the condition-level 0.06).
- **L0** = U2 with state = time interpolation (EC2 `subs/sub_L0.zip`, md5 040b29b04c2df7142e43c03115da3538; profile fallback on 0 of 6,735,795 cells). Q: LB_L0 − LB_U2 = 0.35 ΔS_state + 0.15 ΔS_physics. Two written expectations: local, GBDT ahead by ≈ +0.02 (re-mask S_state 0.932 vs 0.902; S_physics 0.729 vs ≈0.67); LB-inferred from a competitor's row, ≈ +0.007. Decision: Task 1/3 differences are computed over millions of cells and are nearly noise-free, so **keep the GBDT if L0 scores below U2 by any margin** (the April re-mask proxy agrees, 0.934 vs 0.906). Retire it only if L0 ≥ U2. The result also calibrates the local-to-LB physics mapping.

**Critique points accepted:** stale changelog sections (fixed in place: sections 0, 2, 5e, 5f, 5h, 5i, 7, 8); the paired leak table; the latent column; the missingness rates; the threshold print; the scheduler; the count-level R0 check; the robust file; "0.636 was not a derived ceiling"; the ceiling script now in the repo; retracting the speculation about the leader.

**Critique points rejected, with the measurement:** (1) the 0.8833/1.0 split of 0.9636 (impossible arithmetic; S_FD(exact) measured 0.986); (2) the conditional-mean control lifting the physics ceiling (0.770/0.762/0.771/0.767 vs 0.768/0.775/0.789/0.755, mean −0.004); (3) onset being a propagation problem visible in the legal hour (link-level physical rules IoU ≤ 0.05; onset links at free flow at T−5); (4) the onset location prior being March/training memory that will not travel (leave-one-month-out loses nothing, +0.009 ± 0.017; the public v1 onset ≈ 0.64 agrees).

## 6. Mistakes / wrong turns (so they aren't repeated)
0. **Did not read `competition_contract.json` before reverse-engineering the selector, and used slot T without checking the participant contract.**

1. **Over-trusting local CV for Task 2.** Day- and month-blocked CV share one scenario's bottleneck schedule. The LB exposed it after the first real submission. Scenario-level validation should have come first.
2. Assumed a physics ceiling from a competitor's notes (false), and assumed the Task 3 proxy was exact (false).
3. Spent submissions on Task 4 forms (V2/V5) whose failure was predictable from the competitor's log. The information was still useful.
4. Operational bugs: wrong scp path, an OOM from the full-table load, template merge column clash, pkill self-kill.
5. Queued auto-submission B is essentially a re-run of the 0.82669 config (low information value).

---

## 7. Anti-overfitting policy (private LB ≠ public LB)

- The public LB is **one month, 5 incidents, 80 scored Task 2 windows (8 panels × 10; 40 per condition)** [corrected]. Per-window IoU sd is **0.375** (release json), so SE(S_queue) ≈ 0.04 (≈0.009–0.013 of S_total) and SE of one condition ≈ 0.06 IoU (≈0.009 total). **Do not chase public-LB deltas below ~0.01 total.**
- Prefer methods that are **deterministic or scenario-agnostic** (exact ODME, legal forward-filled persistence from [T−60, T), non-learned rules) over fitted per-link models. **Never slot T** (outside the Task 2 contract; the earlier "persistence-at-T" recommendation here was wrong). A location prior is admissible only after a leave-one-month-out test (section 5j: the onset recurrence passes it).
- Use the LB only for **single-component differential probes** that answer a yes/no question. Do not iterate hyper-parameters on it.
- Final selection: one best-validated submission plus one maximally robust submission.

---

## 7b. SUBMISSION GATE (user directive, 2026-09-30)

Nothing is submitted unless all of the following hold:
1. Every change is validated on an admissible local proxy.
2. The expected LB gain is > ~0.005 total.
3. There are no rules-risky inputs pending the host's answer.
4. The critic has reviewed the exact file and its reasoning.
5. Weak-evidence changes are only bundled with validated gains.

No diagnostic-only probes. The 5 probes on 2026-09-30 were diagnostic and counted as a cost.

## 8. Next candidates (rewritten 2026-10-01; the old list recommended a persistence-at-T backbone, which is the leak)

Each is one change against the current legal file, pre-declared, with the other tasks held fixed.
1. **Read U1/U2 as one question each** (section 5i). Onset IoU = (LB_U1 − 0.71192)/0.15, never negative (R0 onset is empty; a lower total means something else moved). Ongoing gain = (LB_U2 − LB_U1)/0.15. No retuning of either file after the scores.
2. **Onset:** non-learned recurrent-bottleneck prior (train queue frequency at T+30 by link and weekday type, top-3 at h=6; leave-one-month-out 0.558 latent IoU on the 40 train onset windows) vs U1's tree. Link-level physical-signal onset rules are **closed** (IoU ≤ 0.05 on the same windows).
3. **State:** L0 = time-interpolation state vs the GBDT, same queue and ODME (the paired Task 1 measurement that was never made).
4. **Ongoing:** CTM rollout with the discharge fix, scored as a non-learned rule on the 40 train ongoing windows against legal forward-fill (0.755 noisy / 0.733 latent). Upload only if it beats forward-fill there.
5. **Task 1 physics:** S_FD is saturated (section 5j); only S_LWR moves. Measure the GBDT against the ≈0.77 ceiling per regime (R3 carries the loss) before any reconstructor work.
6. Reproducibility: one entry point from the raw release to the selected file (award rerun check).
