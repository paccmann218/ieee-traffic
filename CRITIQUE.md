# Critique of the TrafficFlowBench approach

Written 2026-10-01, after the changelog, the code under `tfb/`, the official repo docs, `competition_contract.json`, `synthetic_release_v1.json`, and the competition pages at https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench.

Updated the same day after changelog §5i, `tfb/t2_persist.py`, `tfb/build_u1u2.py`, `tfb/submit_u1u2.sh`, all 240 rows of `data/small/task2/*/window_index.csv`, `tfb/score_local.py`, `tfb/t3_latent.py`, and `repo/src/task3/score_task3.py`. **Read §0 before the rest of the pre-00:01 critique.** Sections 1–12 still describe the work that produced 0.82669 and `sub_safe_v3`. §4.2 has been corrected in place: an earlier draft of it mis-split the published 0.9636.

Updated again after the honest-score-frontier runs (`tfb/score_frontier.py`, `tfb/t2_oracles.py`, `tfb/t2_joint_prior.py`, `tfb/twin_v2.py`, `tfb/t2_front_analog.py`, `tfb/t1_ablate.py`, `tfb/t1_asm_ablate.py`, `tfb/t1_gbdt_ablate.py`, `tfb/t1_faithful_cv.py`, `tfb/build_frontier.py`). **Read §13 for the current score budget, the measured U1/U2/J1/L0 numbers, the closed branches, and the selection rule.** Do not treat §3’s “0.924 is one physical model” illustration as a reachable plan.

This memo is for the agent that produced `CHANGELOG.md`. The private leaderboard is April 2031, generated independently, with seven incidents against March’s five. It is the ranking.

---

## 0. Pre-flight of U1 and U2, and the reply to the critique

**Verdict (pre-00:01): let both uploads go.** That was correct. They scored **U1 0.80579** and **U2 0.82600**. The later isolated files scored **J1 0.84461** and **L0 0.81733**. **Current legal selection is J1.** Build a third file and do not upload it: R0 with only the ongoing rows replaced by forward-fill, onset left empty (`sub_Rff.zip`, already on disk). Do not treat the new physics paragraph as a result. Do not select 0.82669. Do not retune J1 after seeing 0.84461. Read §13 before acting on any number in §0–12.

The two files are isolated, legal changes. That part of the pre-flight holds. The reply also accepts the leaks, cancels v3, and finds a real bug in the organizer persistence rule. Then it uses a local noise experiment to declare the published physics ceiling unreachable, and from that declares a 0.924 score impossible without cheating. The experiment adds a noise draw to the observations. The published 0.9636 is the score of copying those observations. Neither number is the score of a conditional-mean field. §0.5 has the corrected split of 0.9636. §0.9 is what the window index actually contains.

### 0.1 What was accepted, and should stay accepted

- 0.82669 reads slot T. It is a diagnostic. It is not a final selection.
- `sub_safe_v3` bundled self-supervision, ineligible cells, an onset decoder, an add-only hybrid, and March-trained private rows. Cancelling it before upload was the right call. The state file in U1/U2 is `work/t1/state_submission.csv`, not `state_submission_ss.csv`.
- No training on one hidden month to predict the other.
- The window count is 8 panels × 10 windows = 80, 40 per condition. The release IoU sd is 0.375. SE(S_queue) ≈ 0.04 and SE of one condition ≈ 0.06.
- Adjacent-link flow differences are not a conservation test. LWR stores vehicles on the link.
- Cancelling the no-identity ongoing model was a mistake. It is named as a later A/B, not as tonight’s upload. Keep that promise. Do not “fix” U1’s onset model overnight by putting identity back in after the fact; it is already in.
- CTM v0/v1 over-held queues. That was a discharge bug, not a test of the model class. Nothing tonight tests it either. Leaving it unbuilt for these two slots is acceptable. Closing it because of §0.4 is not.

### 0.2 The files themselves

`build_u1u2.py` builds:

- **R0**, the already scored file at 0.71192: GBDT state, organizer persistence queue, V1 ODME.
- **U1** = R0 with onset rows replaced by a train-only LightGBM. History is `[T−60, T)`. Day-context columns are out of the feature list. Only horizon step h = 6 can be positive. Ongoing rows are the organizer rule.
- **U2** = U1 with ongoing rows replaced by a forward-fill of the last released speed in that history, including ineligible cells (`ffill_inel`).

The external cell check is the one that matters, and it passed: U1 differs from R0 in 225 queue cells, all in onset windows, with 0 state and 0 ODME differences. U2 differs from U1 in 2,196 queue cells, all in ongoing windows, again with 0 state and 0 ODME differences. Train-history values used in the evaluation match the masked layer at submission on 145,224 cells, 0 differing. The information-set assert blanks everything outside `[s0−12, s0)` and the onset probabilities do not move. Twelve windows were checked, six per split. The feature function is shared, so that sample is enough for this model.

Two residuals in the script, neither of which blocks these files:

- The in-script R0 check compares the **count** of queued cells (`queue_pred.sum()`), which a permutation of the same number of ones would pass. The cell-level diff was done outside the script. Put that diff into `build_u1u2.py` before anyone rebuilds the zip.
- `merge_and_check` clips state speeds to ≥ 5 after the merge. The cell diff says it did not move state relative to R0. Leave the clip where it is.

The score arithmetic on the two questions is right. Onset is half of S_queue, so a change that touches only onset moves the total by `0.15 × ΔIoU_onset`, and `(LB − 0.71192) / 0.15` is the onset IoU if R0’s onset IoU is 0. Ongoing-only is the same factor: `(LB_U2 − LB_U1) / 0.15` is the ongoing IoU gain. Do not reinterpret a 0.01 total swing. Condition-level noise of 0.06 IoU is about 0.009 total.

### 0.3 U2 is the first result in this project that the earlier critique asked for

`t2_persist.py` scores non-learned rules on the 40 released training ongoing windows. Nothing is fit. On noisy labels:

| Rule | IoU | Paired gain vs organizer |
|---|---:|---:|
| Organizer: last history slot, eligible cells only | 0.590 | — |
| Forward-fill of the last eligible observation in the 12-slot history | 0.748 | +0.158 ± 0.019 |
| Forward-fill, and if the last value is ineligible, use it anyway | 0.755 | +0.165 ± 0.019 |

The organizer baseline, in `build_task2_persistence_submission.py`, keeps a link queued only when the single timestamp `T−5` is present, eligible, and slow. A missing or low-quality last reading becomes “not queued.” About a quarter of cells are in that state. Carrying forward the last real reading inside the legal hour is still inside `[T−60, T)`. It does not read slot T.

This is why the training number is allowed to predict the board, when every tree CV was not. Organizer persistence on these same 40 windows is 0.590, and the public ongoing inference is about 0.60. The level matches. The earlier proxies did not. The mechanism (a dropped sensor is not a free-flow road) does not memorize a corridor or an incident.

The leaked ongoing score of about 0.73 can be read against this only as an unpaired comparison. `t2_persist.py` never scores persistence at T. The training gain is +0.165 on noisy labels. The public leak gain is about +0.13 on latent labels (0.60 to 0.73). Those are different labels, different months, and different code paths. “Most of what looked like slot T was really the forward-fill” is a hypothesis. It is confirmed only by a paired number this script does not print: on these same 40 windows, latent IoU of forward-fill versus latent IoU of slot T. Until that row exists, do not describe the leak as explained. U2 answers a different question, which is still the right question: does the legal carry-forward beat the organizer rule on March’s latent labels?

If the missingness rate on March matches the training windows, U2 should land near **U1 + 0.024** total, ongoing IoU near 0.76, without ever reading T. A result in that neighborhood would mean the 0.82669 file was not a better persistence rule than a legal one. It would not mean the queue task is finished. 0.755 is still a copy of the recent mask. The release keeps ongoing windows only when persistence IoU is at most 0.9, which is exactly the slice where the mask has to move. Carry-forward repairs the dropped sensor. It does not say where the front goes. A good U2 replaces the organizer baseline. It does not replace a forecast.

Three limits on that sentence, all smaller than “do not upload”:

- The quoted gain is against **noisy** labels. `score_task2.py` scores `queue_true` on eligible horizon cells, and `queue_true` is the latent state. The script already computes a latent column and the changelog does not report it for `ffill_inel`. The same script showed hole-fill flip sign between noisy and latent. Publish the latent column for the variant that was shipped. U2 itself is the latent measurement that counts; until it returns, +0.024 is a training-set estimate, not a promise.
- The ineligible add-on is +0.007, inside the ±0.019. The 8-SE effect is the eligible forward-fill. U2 bundles a significant rule with a non-significant extra. Read `U2 − U1` as the bundle. Do not spend a later submission splitting off +0.007 unless U2 comes back flat and the reason matters.
- The ±0.019 is the paired error on these 40 training windows. It is not the error of a new month. The missing check is the fraction of links whose `T−5` cell is ineligible or missing, on train, on March, and on April. If those three rates agree, the +0.024 expectation is the right one. If March’s last slot is much more complete, the gain shrinks, because there is less for the carry-forward to repair. Queue size being half on March does not by itself kill the rule: the rule copies whatever the history just showed, and March’s history is already the smaller queue.

Hole-fill on top of forward-fill scored 0.734 noisy, worse than forward-fill alone. It stays out. That is the correct drop.

### 0.4 U1 is a legal measurement of a model that still memorizes

`fit_onset` drops `DAYCTX` and drops nothing else. `link`, `pos`, `panel`, `slot`, `dow`, and the training-set frequency columns stay in the matrix. Family holdout of that feature set was onset IoU 0.413, and 0.241 without identity. U1 will report the public score of the memorizing model under the legal information set. It will not report a method that has been shown to travel.

The forecast “≈ 0.80” is the old onset IoU of 0.64, taken from the model that saw slot T, added to 0.71192. The legal model can be lower. A total anywhere from about 0.78 to 0.82 is the same answer at the condition-level noise (0.06 IoU ≈ 0.009 total). Do not train a second onset model because U1 missed 0.80 by 0.01.

What U1 is allowed to decide: whether this legal onset stack is above the empty onset of R0 by enough to keep, as a March measurement. What it is not allowed to decide: that the stack is the April onset model. The next onset comparison, after these scores, is the no-identity model already named, or a non-learned rule using the speed ratio and flow in the legal hour (the clue already printed in `synthetic_release_v1.json`). It is not a change of `num_leaves`, the threshold, or the decoder.

The h = 6 hard zero is still the sampler regularity (first visible queue at `T+30` on 39/40 training windows). It is legal if April uses the same selector. It is wrong when the latent queue occupies more than the last step. U1’s onset IoU includes that error. Do not “fix” it between U1 and U2.

### 0.5 The physics-ceiling rejection does not survive the published table

§5i says an ideal predictor, the true field plus a fresh noise draw at the measured sd, scores local S_physics **0.735–0.787**, the GBDT scores 0.725–0.74, and therefore the headroom is about +0.002 to +0.008 total. From that it builds:

- non-queue maximum ≈ 0.636
- even a perfect queue forecast totals ≈ 0.94
- the leader at 0.924 needs S_queue ≈ 0.96
- 0.924 is therefore out-of-contract
- a realistic legal target is 0.85–0.88, and the only remaining work is Task 2

The 0.636 is not a derived ceiling. It is the current stack with the physics term left at the GBDT’s local score:

```text
0.35*0.94 + 0.15*0.73 + 0.20*0.994 ≈ 0.329 + 0.110 + 0.199 = 0.638
```

S_state 0.94 is a noise ceiling. S_physics 0.73 is the model they already have. Adding those and calling the sum a maximum is how the queue requirement for 0.924 becomes 0.96. Under that same arithmetic a public 0.87 needs S_queue ≈ (0.87 − 0.636) / 0.30 ≈ 0.78, which is an ordinary forecast, not a breach. The “someone above us must be cheating” sentence fits the leader only, and only after the physics term has been frozen.

The state half of this is right, and the earlier critique should be read with it. Speed noise of about 1–2 km/h on a 25 km/h scale, and flow noise of about 20–30 vph/lane on a 600 scale, put a perfect-mean predictor near S_state 0.94–0.96. The re-mask proxy already has the GBDT at about 0.93. State headroom is on the order of 0.01 of the total, not a place to spend the next week of trees. That concession stands.

The physics half uses the wrong perturbation. An earlier draft of this section also mis-read 0.9636. Both are corrected here.

`load_boundary_flux` in `repo/src/task3/score_task3.py` says the organizer flux is projected onto the released observations, so a submission that reproduces those observations satisfies the conservation identity exactly. Deriving the same flux from the topology and from the submission’s own flows put the exact truth at S_LWR 0.0. The conservation signal is 0.75% of accumulated vehicles, and a topology-derived flux misses the simulated one by about 0.22%, a third of the signal. Any local physics number computed from topology boundaries is a different metric, and on that metric the right answer scores zero.

`repo/docs/SCORING_SPEC.md` records an honest S_FD of 0.8833. With S_LWR = 1,

```text
(1/3)*0.8833 + (2/3)*1 = 0.961
```

which is the published exact-answer S_physics of 0.9636 in `repo/docs/BASELINES.md` (naive baseline 0.3549). The gap from 0.96 to 1.0 is the diagram term. `tfb/t3_latent.py` split the 0.96 the other way, S_FD near 0.99 and S_LWR near 0.945, and tuned a smoother until the noisy truth scored 0.945 against a latent right-hand side. That calibration was fitting a misread of the baseline table. The scorer comment says the right-hand side is the observations.

`tfb/score_local.py` implements that projection: S_LWR = 1 − sum|ΔN_sub − ΔN_observed| / sum|ΔN_observed|, so the exact observations score S_LWR = 1. Its docstring says to validate it once against `score_task3.py` with a written `--boundary-flux` file, and it points at `make_flux()`, which is not in the repo. The changelog does not record the check. Until that file is scored, a local 0.73 and a local 0.74 are the leaderboard formula only on the assumption that the reconstruction matches the withheld parquet.

0.9636 is the score of submitting the observations, noise included. A predictor does not observe that noise. The scoring spec says S_LWR falls monotonically as error is injected. The reachable physics score is the score of the conditional-mean field against ΔN of the observations.

The §5i experiment builds a different field: the observations plus another noise draw, scored locally. That field sits one noise sigma away from the labels, which is the right distance, and it also sits further off the diagram than the observations do. Honest S_FD on the observations is about 0.88. A second noise draw pulls S_FD down from there. A denoised field moves S_FD up and S_LWR down. Those two moves are not inside 0.735–0.787. The missing control is that denoised field, scored with `score_local.T3.score` on the same panels, after the boundary-flux check above.

So 0.74 is not a ceiling, and 0.96 is not a reachable one. A previous draft of this section illustrated a legal total near 0.91 by plugging S_physics 0.85, as if a forecast could submit the exact-answer score. Retract that illustration. What remains is narrower. The GBDT’s local 0.73 matching a double-noised field does not measure the March gap. The public state-plus-physics piece of about 0.423, with S_state near 0.93, still implies public S_physics near 0.64. The difference between 0.64 and the conditional-mean score is the open physics question. It has not been computed. It does not authorize “0.924 requires S_queue of 0.96,” and it does not authorize “physics is finished.”

The density-aware decoder that was dropped because local J fell is still an open Task 1 question on this metric: it changes ΔN, so it has to be scored as ΔN against the observations, not as a local J. It is not reopened tonight. It is not closed by §5i.

### 0.6 The changelog now contradicts itself, and the stale half will be followed

§5i corrects the window count, the leak, the pooled self-supervision, and the persistence-at-T recommendation. The following sections still say the old thing:

- §0 still has `sub_safe_v3` scheduled for 00:01 UTC.
- §5e still says the public board is 40 windows.
- §5f still says adjacent-flow sd “matches the noise level,” while §2 still says that sd is about 190 vph.
- §5h still says the gap to 0.87 is Task 2 and that v3 is the submission.
- §7 still prefers “persistence-at-T” and still quotes 40 windows and sd 0.3.
- §8 still lists a persistence-at-T hybrid as the next model.

A later session that reads §7 and §8 will rebuild the leak. Edit those sections in place so they match §5i. A correction that lives only in a later section is not a correction.

### 0.7 What to do with the two scores, and what not to build next

U1 and U2 are one question each. Read them that way and stop.

- `U1 − 0.71192` is the legal onset IoU of this tree, times 0.15. If it is clearly positive, the tree beat an empty onset on March. It is still a link-identity model. The following onset file is no-identity or non-learned, not a retune.
- `U2 − U1` is the forward-fill bundle against organizer persistence. If it is near +0.02, the carry-forward rule is the legal ongoing baseline and it replaces both the organizer rule and slot T. If it is near zero, the training gain did not survive latent labels or March’s missingness, and the rule is retired with that evidence. Neither outcome is a reason to put the add-only hybrid back.
- The robust file to keep for the final selection is the one whose ongoing rows are this non-learned carry-forward, whose state file is the current GBDT only until a reconstructor scores higher S_LWR against ΔN of the observations on the boundary-flux check in §0.5, and whose onset rows are not required to be this tree. A field that conserves more tightly than the noisy observations moves ΔN off the projected flux and pays for it. A March win by U1’s onset model is not, by itself, a reason to prefer U1 over that robust file for April.
- Do not resume self-supervision, March-to-April training, slot T, day context before `T−60`, horizon ramps, or `sub_safe_v3`.
- Do not spend the next submission on Task 4, on ineligible-versus-eligible forward-fill, or on a bigger queue tree.

The simulator remains the open model, on the evidence already in `twin_check.py`: congested cells sit on the released diagram. v0 failed because it held the queue. The next version changes the discharge at a queued-to-free interface, and it is scored the way U2 was scored, as a non-learned rule on the 40 released training windows, against forward-fill rather than against the organizer’s dropped-sensor rule. It is not scheduled tonight. It is not cancelled by a double-noise physics score.

### 0.8 What the first pre-flight missed

**The carry-forward claim was not paired with the leak.** Section 0.3 above. Slot T is not a row in `t2_persist.py`. The latent column for `ffill_inel`, which is the column comparable to the public 0.73, was computed and not written down.

**Three legal variants the script scores were left out of the changelog.** `variants()` also builds `mean2` (mean of the last two forward-filled eligible speeds), `mean2_fill`, and `ffill_fill`. The table in §5i shows organizer, eligible forward-fill, ineligible forward-fill, and ineligible-plus-hole-fill. If `mean2` beat `ffill_inel` on the noisy column, U2 is not the best non-learned rule they already measured. Paste the full printed table, both label columns, before treating `ffill_inel` as the one that won.

**U2 does not isolate the rule that will be selected for April.** U2 = legal onset tree + forward-fill. U1 = legal onset tree + organizer persistence. The public difference cancels the tree. The private score does not, unless both files are selected and subtracted. There is no zip that is R0 with only the ongoing rows replaced. Both uploaded files contain the memorizing onset model. If that model is wrong on April’s seven incidents, both private scores pay for it, and the carry-forward cannot be separated without U1 also being a selected final. Build that third file tonight and do not upload it: same state, same ODME, organizer onset (all zeros), forward-fill ongoing. It is the robust final. It costs no submission.

**U1’s training distribution is still the synthetic one.** The rows come from `t2s.load("train", "onset")`, which keeps a window only when the queue exists at the last horizon step and nowhere earlier, and then fits only that step. That is not the 40 released onset windows. U2 was at least scored on the released ongoing windows. U1 was not trained on the released onset windows. The 0.64 being carried forward as the expectation was itself measured on a different model, one that saw slot T.

**The cell counts are part of the check, not just the isolation.** 225 changed onset cells across validation and private is 80 onset windows (8 panels × 5 × 2 splits), about 2.8 positives per window. The decoder forces at least one, so the model is painting a very small set at h = 6, not a corridor. 2,196 ongoing changes is about 27 cells per ongoing window. A mask repeated for six steps is about four or five links. That matches a small carried queue. It does not match the training ongoing queues of median about 80 cells. Write down whether those 225 onset cells are the same link ids the frequency feature would have picked. If they are, U1 is the memory, and the March score will look better than April.

**If U1 scores below 0.71192, the onset formula is wrong, not negative IoU.** R0’s onset predictions are empty, and IoU against a non-empty truth is 0. Replacing an empty prediction cannot lower onset IoU below 0. A lower total means state, ODME, or the ongoing rows moved, or R0’s onset was not actually empty. Do not report a negative onset IoU.

**`submit_u1u2.sh` submits U2 even when U1 never reaches COMPLETE.** The wait loop exits after 30 minutes and the next command is the U2 upload. That is acceptable only if both scores come back. If only U2 is scored, `U2 − 0.71192` mixes the onset tree and the forward-fill. Do not divide that mixture by 0.15 and call it either answer. Also, `date -d tomorrow` after 01:00 UTC waits until the following midnight. A retry in that window misses the day.

**The noise numbers used for the state ceiling disagree with §2.** §2 says speed noise about 1.8–2.3 km/h and flow noise about 30 vph. §5i says 0.9–1.7 km/h and 17–31 vph/lane, on three panels. The script that produced 0.735–0.787 is not in the repo. It cannot be checked whether “truth + fresh noise” denoised first, which noise channels were shaken, or whether the same seed was used for the GBDT comparison. Three December–February panels are the training scenario. None of that is the published exact-answer score of 0.9636.

**The threshold helper and the persistence script do not read the same column.** `queue_utils.thresholds` uses `0.6 × links.csv` free speed unless `fd_parameters.csv` has a `v_cut` column. The sample network has no `v_cut` column. `t2.statics` uses `0.6 × fd_parameters` free speed. In the sample those free speeds match (114.7 on the same link), so the paired +0.165 is internally consistent. The Task 3 scorer comment still says `links.csv` is a flat 105 and the fd file is 113–117.5. §2 says they are identical. If the full release matches the sample, ignore the comment. If the full release matches the comment, R0’s official threshold and U2’s threshold differ by about 6 km/h and the public gain will not equal the training gain. One print of the two cutoff vectors on a scored panel settles it. It was not done.

**Carry-forward fails in a specific way the training average hides.** Filling a missing tail with the last good reading recovers a queue the organizer called free. It also keeps a queue that has already cleared, whenever the sensor died on the way out. March’s ongoing queues are about half the training size, so a stale carried mask is a larger fraction of the union. The missingness-rate check in §0.3 is the one that says which of those two effects March will see. The training +0.165 alone does not. §0.9 measures the published `history_coverage` column: the hour-average rate does not differ across train, March, and April. The last-slot rate is still unmeasured.

**Reproducibility of these zips is still an EC2 path, not an entry point.** `build_u1u2.py` reads `subs/q_persist.csv`, `subs/odme_ref.csv`, `work/t1/state_submission.csv`, and the `t2s` parquet tables under the home directory. The pre-flight shows the current bytes are a clean diff. It does not show that a fresh clone rebuilds them. The award check reruns the code.

**One credit the first pass skipped.** `merge_and_check` refuses the file if more than a fifth of state flows are below 50 vph. That is the empty-road rule that zeros S_FD. Keep the assert.

**Onset does not see the ineligible speed. It does see a carry-forward.** `fit_onset` asserts `INCL_INELIG` is off, so `day_layer` writes ineligible speeds as NaN. `features()` then forward-fills those NaNs along the hour before it builds `r_lag0`. A missing last slot becomes the previous eligible speed ratio. The ineligible cell’s own speed never enters the tree. The ongoing rule’s measured gain from using that speed was +0.007, inside the SE of the eligible carry-forward. The two halves of U2 disagree on the ineligible number. They agree on the eligible fill. §0.8’s first draft called the onset model blind to the whole hour. That overstated it.

---

### 0.9 What the window index and the scorer comment actually say

All 240 rows under `data/small/task2/*/window_index.csv` were read. That is the scored set: 8 panels, three splits, 10 windows each. It is not a toy subset of the index.

**The clock.** `history_end` equals `forecast_origin` on every row. History is exactly 60 minutes. The forecast starts 5 minutes later and runs 25 minutes, six steps. `future_coverage` is 1.0 on every window, so the horizon label is complete. Missingness is a history property. Origin hours run from 04:00 to 21:35 UTC. The latest horizon ends at 22:05. No scored window crosses midnight.

The submission timestamp in `build_u1u2.py` is `f"{date}T{(s0+k)//12:02d}:{(s0+k)%12*5:02d}:00Z"`. A horizon that crossed midnight would emit `T24:00:00Z`, `assemble()` would fail to match that key, and the organizer value would be kept with no assert. On these 240 windows that path is dormant. It is a rebuild hazard for a later window list. It is not a defect in the zips that were checked.

**Coverage does not move between months.** `history_coverage` runs from 0.721 to 0.801, median 0.764. The eight split-by-condition means sit between 0.760 and 0.767. Train, March, and April look the same, and onset looks like ongoing. The published column is the whole hour. It does not say how often the last slot, T−5, is the missing one. The “March missingness may not match the training +0.165” caveat is not supported by this column, and it is not retired by it either.

**The pairing script was written and not recorded.** `tfb/t2_selector.py` compares `history_coverage` to the eligible fraction of slots `[s0−12, s0)`, which excludes the origin, and it scores persistence of T−5 against persistence of slot T on the same 80 training windows (`pers_iou`, `persT_iou`, `hq_last`, `q_at_T`). The changelog has neither the coverage-match rate nor that paired table. That print is the comparison §0.3 asked for. `t2_persist.py` never scores slot T, so the sentence “the leak was the fill” is still unpaired.

**`history_end` equal to the origin does not by itself put slot T in the file.** The official builder persists `timestamp.max()` of the history parquet, and its comment says that maximum has to fall inside `[T−60, T)`. The index stores the origin as `history_end` on all 240 rows, which is what an exclusive end looks like and also what an inclusive end looks like. The public gap settles the score, not the clock: 0.82669 against 0.71192 is the size of an empty onset plus an ongoing IoU near 0.60, which is the T−5 story. Slot T at about 0.73 would have put the organizer file near 0.73, not at 0.71192. The one line still worth printing, because `t2_selector.py` already computes it, is whether the index coverage matches eligibility on `[s0−12, s0)`. A match says the parquet’s last slot is T−5. A mismatch says U2 is ignoring a timestamp the organizer baseline uses.

**The onset feature already contains the eligible fill.** See the last paragraph of §0.8. Do not build a second onset model to “add forward-fill” without checking that `r_lag0` is already that fill.

---

## 1. What this contest actually pays for

The score is

```text
S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
```

The data were generated by a calibrated traffic-flow model. The overview and `repo/README.md` say the four tasks are one loop: fill the gaps, obey conservation, say where the queue goes, recover the demand. They also say, in the evaluation section, that a method tuned until it fits validation has learned that month’s incidents. `competition_contract.json` calls private “a different month of the same scenario.” Same generator, same corridor physics, different demand draw, different incidents. Memorizing March’s bottlenecks and memorizing June–February’s bottlenecks fail for the same reason.

Two facts in the release define the test the models never trained for.

- `synthetic_release_v1.json` sets `maximum_persistence_iou` to 0.9. Ongoing windows are kept only when copying the current queue is already a bad forecast. The scored Task 2 set is the subset where persistence fails.
- The same file states the onset clue directly: an hour before a queue forms, the road sits at about 1.28× the queue threshold against 1.54× where no queue follows, and it carries about 6912 vph against 4620. Onset is a state that is already visible in the hour of history. It is a propagation problem on a known fundamental diagram.

The practical ceilings, from `repo/docs/BASELINES.md` on the validation split, are S_state 1, S_queue 1, S_physics about 0.96, S_ODME 1, S_total about 0.994. The physics term does not reach 1 even for an exact answer, because conservation is checked against noisy observations. Anything in this repo that scores the ground-truth field at S_LWR = 1 is a different metric from the leaderboard.

A reconstruction that is right cell by cell and wrong between neighbors loses the 0.15 physics weight and also loses the queue forecast, because the queue is the congested region of that same field. Those two weights are 0.45 of the total. The work treated them as separate Kaggle problems.

---

## 2. The public number is not a method

Best public score in the changelog: **0.82669**. The leader is about **0.924**. The changelog’s own decomposition, which is internally consistent, is:

| Piece | Value | Contribution |
|---|---:|---:|
| S_ODME ≈ 0.994 | exact NNLS on the split’s own counts | 0.199 |
| 0.35 S_state + 0.15 S_physics | ≈ 0.423, from the GBDT probes | 0.423 |
| S_queue ≈ 0.684 | onset ≈ 0.64, ongoing ≈ 0.73 | 0.205 |
| **S_total** | | **0.827** |

The 0.73 ongoing number is persistence of the corridor state **at time T**. The organizer’s own baseline says why that is not a Task 2 input. `repo/src/task2/build_task2_persistence_submission.py` lines 85–87:

> Persistence must use only the released [T-60, T) window history. The old implementation re-read the corridor state at T, which is not part of the Task 2 participant contract and leaked one extra observation.

`tfb/t2.py` `features()` builds `hv` from `mv[s0-12:s0+1]`, thirteen slots, and the comment in that function says the window includes T. Every Task 2 model and the best submission were fit and scored with that extra slot. Rules-safe persistence, last history slot T−5, is about **0.60** ongoing on the public leaderboard. With onset left at 0.64, S_queue falls from about 0.68 to about 0.62, and the total falls by about **0.02**, to roughly **0.81**. The changelog’s own expectation for the compliant file is 0.80 ± 0.015.

So the rank on the public board is the leak. Strip it and the entry sits with the pack that copied the last legal observation and ran a generic regressor. The later write-up still calls 0.82669 the best score and treats the compliant file as a sacrifice. The sacrifice is the first legal submission. The 0.82669 file is a diagnostic that should never be selected for the final.

This also poisoned the science. Every comparison of the form “the model loses to persistence” was a comparison against the leaked state. The model lost to an input it is not allowed to see. That fact was then used to justify an add-only hybrid that refuses to delete persisted cells (§8). The backbone of that hybrid is the weaker T−5 field, and the decision to never delete was learned from the stronger illegal field.

Using T will probably still “work” on April, because the same cell is present in the private masked layer. It is not overfitting. It is an information-set violation that transfers, and it will be visible in a code review. Awards are 0.85 × private rank + 0.15 × report, and the report rubric gives 40 points to technical soundness. Reproducibility has to match the selected file to within 1% of the private score. A finalist whose gain depends on reading T will have to either reproduce the breach or lose the points when the breach is removed. Select the legal file.

---

## 3. The missing score was mis-assigned to Task 2

Section 5h says the gap to a typical 0.87 is Task 2, because 0.87 implies S_queue ≈ 0.80 if the other components stay at the GBDT-probe values. That arithmetic is only true if the teams above have the same state and physics scores. The benchmark was built so that they would not. A method that reconstructs a conserved field lifts S_state and S_physics together, and the queue forecast is then a short rollout of that field.

Put numbers on the room that was left on the table, using the changelog’s own probe split and the published ceilings:

- Physics. If the probe split is right, S_physics on the public board is about **0.64** (0.423 − 0.35×0.93, using the new-scenario proxy S_state ≈ 0.932). The published perfect answer is 0.964. Closing even half of that gap is about **0.024** of S_total. Closing it the way the README describes, by making the reconstruction conserve, is also the Task 2 model.
- Queue, legal. From ≈0.62 to a serious forecast around 0.80 is **0.054** of S_total. From the leaked 0.68 the same target is 0.036, and 0.02 of that “have” disappears the moment T is removed.
- State. The GBDT’s entire advantage over time interpolation on the public board is about **+0.007** total (0.423 versus ≈0.416). Month-blocked CV said the advantage was +0.021 in the combined state+physics objective J. Two thirds of the CV gain did not survive one new month.

The project then spent the remaining submissions, the CTM, the hybrids, and the decoder search on Task 2 classifiers, and adopted a Task 1 change worth about **+0.001** total (self-supervision, +0.003 S_state). The largest unused lever that the local tools could actually see — conservation error between neighboring links — was left on a proxy the author already did not trust.

A top score near 0.924 is what you get from roughly S_state 0.94, S_queue 0.88, S_physics 0.85, S_ODME 0.99. That is one physical model, run as a reconstruction and as a 30-minute forecast. It is not a better LightGBM.

---

## 4. Task 1 was optimized for the metric that does not see the failure

### 4.1 The objective was per-cell RMSE

`tfb/t1_train.py` fits two regressors to `y − interpolation`, speed and per-lane flow, with `num_leaves=255`. `tfb/t1_features.py` feeds them temporal interpolation, a Treiber–Helbing ASM smoother, neighbor innovations, a train profile, ramp flows, link position, weekday, regime, and panel id as a categorical. V2 added sixteen more features for a CV gain of **+0.0003 J** and was, correctly, dropped. The model class had already saturated.

S_LWR is not a sum of per-cell errors. It is

```text
S_LWR = 1 - sum |ΔN_submitted - rhs| / sum |rhs|
```

over link-to-link transitions. Two adjacent links can each have a small speed error and a small flow error and still create or destroy vehicles. The README says this is the failure mode RMSE cannot see, and that it is why a perfect answer scores about 0.96 physics while the historical-mean baseline scores about 0.32. The GBDT was selected because it reduced speed RMSE from about 3.0 to 1.68 km/h and flow RMSE from about 44 to 32 vph/lane on a month-blocked slice of the **training scenario**. That is the metric the physics score is designed to ignore.

### 4.2 The local physics number is a different metric, and it flatters smooth models

`tfb/score_local.py` sets the conservation right-hand side to ΔN of the observed field, so the ground truth scores S_LWR = 1. That agrees with `load_boundary_flux` in `score_task3.py`: the organizer flux is projected onto the released observations, and reproducing them satisfies conservation exactly. `SCORING_SPEC.md` gives an honest S_FD of 0.8833. With S_LWR = 1 that is S_physics 0.961, the published exact-answer 0.9636. An earlier draft of this section said those two facts contradict each other and that S_LWR of the exact field is about 0.95. They do not. The 0.04 below 1.0 is the diagram term. §0.5 has the arithmetic and the reason 0.9636 is still not a predictor’s ceiling.

`tfb/t3_latent.py` assumed the reverse split, S_FD near 0.99 and S_LWR near 0.945, and calibrated a smoother so the noisy truth would score 0.945 against a latent right-hand side. The scorer comment says the right-hand side is the observations. Killing that proxy was right. The docstring of `score_local.py` still asks for one check that has not been recorded: score the same file with `score_task3.py --boundary-flux`. `make_flux()` is not in the repo. A topology-derived flux, the fallback in the official scorer when the parquet is absent, put the exact truth at S_LWR 0.0. A local number that took that fallback is not this metric.

What the numbers still show:

- Month-blocked CV reported S_state 0.941 and S_physics 0.729, J = 0.439. The public board returned about 0.423 for the same pair. With S_state near 0.93 that implies public S_physics near 0.64. If `score_local` is the leaderboard metric, the gap is transfer to March.
- The new-scenario re-mask proxy later showed the state half does transfer (S_state 0.932 / 0.934 on validation / private, against time interpolation at 0.902 / 0.906). The physics half of that proxy was not the gate.
- A reconstruction that denoises each link changes ΔN relative to the observations and pays for that here. Under the scorer comment the leaderboard does the same. The earlier claim that this proxy rewards smoothness the leaderboard punishes does not survive that comment. What the proxy has not been shown to do is rank two reconstructions the way March ranks them. It reproduced the baseline floor of about 0.32, which is S_LWR near zero. Matching the floor does not validate the ranking above the floor.

Model selection for Task 1 continued on S_state alone after that. The adopted change (own-month profile + self-supervision) was accepted because S_state moved from 0.9337 to 0.9368 on the re-mask proxy. Physics was not computed for it. A +0.003 S_state move is +0.001 total, below the changelog’s own noise floor of ~0.01, and it can lower S_LWR by more than that if the extra fit makes neighboring links less consistent. There is no measurement either way.

### 4.3 The cross-validation shares the scenario it claims to hold out

Month-blocked CV trains on June–November and tests on December–February. Those months share one bottleneck schedule, one incident process, one capacity. The profile features are leave-one-out over training months of that same scenario, so the test months still see the scenario’s weekday pattern. Panel id is a categorical, so the trees are free to keep a separate model per corridor. Leave-one-corridor-family-out was never the Task 1 gate. The first genuinely new scenario was the public leaderboard, and the gain over interpolation shrank by a factor of three.

The later proxy (`tfb/t1_proxy.py`) re-masks 10% of observed eligible cells in March and April. It is admissible for a coarse S_state check: the historical profile lands near the published 0.693 and time interpolation lands near the competitor-implied ~0.91. It is not admissible for the decision that was then made.

- The real mask is 20%, 30%, or 50%, one regime per day, scored as an equal average of regimes. A corridor can have a third of its state score sitting on a handful of R3 days (`MASK_SPEC.md`). A 10% random hole with almost all neighbors present is the easy regime.
- Task 2 cuts a 90-minute hole (horizon plus buffer) out of the day. Interpolation and ASM across the remaining field never have to cross that hole in the proxy, because the proxy’s targets are random observed cells, not the real template.
- Incidents are a few hours on five days. They are a small fraction of random cells, so they barely move RMSE. They dominate conservation error and they dominate Task 2. A proxy that says “GBDT transfers” is a statement about typical cells.

### 4.4 Self-supervision pools the two hidden months into one model

`tfb/t1_final_ss.py` trains a single pair of GBDTs on 30% of the training rows plus re-masked cells from **both** validation and private, then predicts both splits with that model. Task 1 is offline inside a split: the other observed cells of the month being reconstructed are legal inputs. April’s cells are not a legal training set for the March reconstruction, and March’s incidents are not a training set for April.

The proxy that “validated” this (`tfb/t1_ownmonth.py`) does the same pooling and then scores each month’s held-out re-mask. The evaluation cells of March are predicted by a model that has been fit to April and to the rest of March. The +0.003 is the model seeing more of the two test months, measured on random cells drawn from those same months. That is the public-board version of fitting the test set. It will look stable on the proxy for exactly that reason. On April’s incidents, the trees have been pulled toward March’s incident residuals and toward whatever spatial pattern the private month happened to show in the sampled cells.

The own-month profile is a weekday-by-slot mean of the month being filled, leave-one-day-out. That part is a legitimate offline feature. With about four samples of each weekday, one incident day moves the profile for every other day of that weekday. The feature is usable. It is not worth a submission by itself, and it was not tested under the real mask or under S_LWR.

There is already a structural baseline in the repo, `baseline_task1_structural_kf.py`, and the feature code already computes an ASM estimate along free-flow and congested characteristics. The production model uses ASM as one more column for a tree, then adds a residual that is free to break `q = k v` and free to break conservation. The residual is what CV rewarded. The characteristic smoother is what the physics score rewards. Shipping the residual and adopting a third-decimal adaptation of it is the leaderboard decision.

### 4.5 What Task 1 should have been selected on

A candidate reconstruction is worth submitting when, on train months that were not used to fit it, all three of these move, and the same direction appears on a re-mask of March and of April:

- S_state, on the real template, reported separately for R3.
- The conservation residual across mainline neighbors, with ramp flows treated as missing when `pct_observed < 75`, never as zero.
- The FD residual `q` versus the released triangular diagram, per lane.

Time interpolation is the robust baseline until a physical reconstructor beats it on the conservation residual on **both** hidden months’ re-masks. The GBDT does not have that evidence. Keeping it because it won a June-versus-December RMSE comparison is how the public board got +0.007 and how April can take it back. That +0.007 is not even a paired submission: the interpolation number was inferred from a competitor’s leaderboard row, not from an upload of `time_interp` with the same Task 2 and Task 4.

### 4.6 The one experiment that coupled speed and flow was dropped because the proxy disliked it

`tfb/t3_decode.py` rebuilds the pair so they share a density: `k` mixed between `q_hat/v_hat` and interpolated observed density, then either flow or speed is rewritten to match. Every blend made local J worse on December–February, and the decoder was dropped.

That is the FD constraint the physics score is built on, rejected by the metric in §4.2, the one that scores the ground truth at S_LWR = 1 and that had already overstated the GBDT’s physics by enough to explain the public gap. A constraint whose only evaluation is “RMSE fell on the training scenario” will lose to an unconstrained tree every time. Dropping it closed the only Task 1 path that treated `(v, q)` as one physical state.

The same proxy killed the latent-reference hypothesis in `tfb/t3_latent.py`. That file set the right admission bar: a smoothed rhs is acceptable only if truth scores near 0.945 and the GBDT and interpolation land near the leaderboard’s physics. Smoothing the reference sent truth to S_LWR 0, so the bar failed. The conclusion written down was “the ceiling stands, stop.” The conclusion the bar supports is “this rhs is not the organizer’s flux, so stop selecting models by it.” A reconstruction that is conservative by construction does not need that proxy. It needs S_state under the real mask, and an internal residual that uses the released ramp flows and topology.

### 4.7 Training does not match the aggregation the score uses

- `S_state` is an unweighted mean of R1, R2 and R3. `MASK_SPEC.md` says a corridor can put a third of that mean on a handful of R3 days (four days on one validation panel). The GBDT is fit by row, so a regime with more target rows dominates the loss. R3 is both the hardest mask and the one a row-weighted fit under-represents when it has fewer days.
- Corridors are equal-weighted after directions are averaged. `t1_train.py` concatenates every panel’s rows. A long corridor contributes more gradients than a short one. The leaderboard will not.
- The CV and the production fit (`t1_predict.py`, `t1_final_ss.py`) both train on a **30% subsample** of target rows (`frac=0.3`), 800 trees, `num_leaves=255`. The number that was compared to the leaderboard is a subsampled model. Part of the CV-to-board gap is that variance. A bigger tree was the response to it.
- ASM in `t1_features.py` uses a hardcoded free-flow speed of 100 km/h and a congested wave of −18 km/h (`c_cong=-12` in the second kernel). The released diagram has free-flow around 113–117.5. The wave speed is an input from `fd_parameters`, not a constant. The smoother that was supposed to carry the physics does not use the released diagram.
- Changelog §8 asked for a leave-family-out ablation of the Task 1 profile and the panel identity. It was never run. Panel stays a categorical in the production model.
- Twenty-nine training predictions fell below 5 km/h and were clipped to `[5, 140]`. The clip hides the tree leaving the diagram. It does not put it back. The empty-road rule (S_FD = 0 if more than a fifth of submitted cells are below 50 vph) was never checked on the file.

`twin_check.py` already showed congested cells sit on the released triangle (about 96–99% within 5 km/h). That was the evidence to constrain the reconstruction. It was used as a feasibility note and then left behind.

The conservation half of that same script compares simultaneous flows on adjacent ramp-free links. LWR does not set those flows equal: vehicles accumulate on the link. The changelog says that spatial difference has sd ≈ 190 vph early on, and later that the sd “matches the noise level” (flow noise is about 30 vph). Those two sentences disagree, and 190 vph is the storage term, not a license to treat neighbor flow as a copy. The FD check is the real evidence. The neighbor-flow check is the wrong equation, which is also why a per-cell model can look accurate and still fail S_LWR.

---

## 5. Task 2 trained the wrong function on the wrong windows, then trusted the CV

This is the main failure. It is also the place where the post-critic work repeated the original mistake with more machinery.

### 5.1 The label the model fits is not the label the leaderboard uses

The scorer labels a link queued when **latent** speed is at or below `0.6 × free_speed`. Measurement error is not allowed to flip the label. Training in `tfb/t2.py` and `tfb/t2s.py` sets

```text
y = observed_speed <= v_cut
```

on eligible cells. Observed speed noise is about 1.8–2.3 km/h. The threshold sits near 68 km/h. The cells that flip are the cells at the queue front, which are the cells that decide whether the queue grew, shrank, or stayed. Intersection-over-union penalizes a wrong front twice, once as a miss and once as an extra.

The project knew this. Denoised labels made the onset-selector match worse, which says the **selector** uses raw speeds. The **score** uses the latent state. Those are different objects. Hole-filling was then accepted because it raised a denoised-truth proxy from 0.726 to 0.748 and was kept even though the same decoder fell from 0.748 to 0.733 against noisy labels. The decoder was chosen by the proxy that agrees with the author’s hope about the label, after the author had just shown the two labels disagree.

A model that regresses speed, with the known noise, and only then applies the threshold, is aimed at the scored label. A binary tree on noisy threshold crossings is aimed at the sensor.

### 5.2 Identity and frequency features are memory of where queues lived

`tfb/t2s.py` puts `link`, `pos`, `panel`, `slot`, and `dow` in the feature frame, and `freq` / `freq_nb` are the training-set queue rate by weekday-type, slot, and link. The production fit (`fit_models`) drops frequency for ongoing and drops **nothing** for onset. Day-context columns are dropped. The link index stays in both models.

The changelog’s own leave-one-corridor-family-out result:

| | With identity | Without identity | Persistence at T |
|---|---:|---:|---:|
| Onset | 0.413 | 0.241 | — |
| Ongoing | 0.781 | 0.774 | 0.677 |

Onset collapses as soon as the corridor family is new. Removing the identity columns makes it worse, which means there is no transferable onset mechanism in the feature set at all: the 0.41 was memory, and the rest of the features do not locate a new queue. Ongoing still beats persistence under family holdout, on **training-scenario** episodes. That result was then treated as evidence the ongoing model generalizes. Family holdout changes the roads. It does not change the scenario. March is the same roads and a new incident draw. The failure mode family-holdout cannot see is “this link queued every afternoon in the training months, and the March incident is two interchanges away.”

The shipped onset model is the full feature set, identity included. That is a decision to submit the memorizing model after the ablation that exposed the memorization. Private has seven incidents on those same links. The frequency prior points at the old bottlenecks. Each false link in the predicted set costs IoU. This is the transfer failure that produced onset ≈ 0.64 on the board against 0.92 in day-grouped CV, and it is the same failure April will score.

### 5.3 The training windows are an easier problem than the scored windows

The scored windows are published in `window_index.csv`. Reproducing the selector is useful for one purpose only: building extra training windows with the same bias. It is not useful as a forecast. The changelog records hours of reverse engineering before `competition_contract.json` line 137 was read. The line says a window is ongoing when the visible history holds at least two queued observations with at least two on one link, and onset otherwise. After the contract was read, the reimplementation hit onset 39/40 and ongoing **10/40**, and the search was stopped.

Stopping was right. Training on the windows that search produced was not.

`windows_for_day` in `tfb/t2s.py` yields an ongoing sample at the first one to six slots after the contract trigger. It does **not** apply `maximum_persistence_iou ≤ 0.9`. The scored ongoing windows are exactly the ones that filter keeps, and the changelog says those windows sit in the afternoon and evening and include young queues. The training sample is dominated by the moments just after a queue becomes visible, which is when persistence is correct. A classifier trained there learns to copy the mask. On the public board the pure model scored about 0.59 ongoing against 0.73 for leaked persistence and about 0.60 for legal persistence. The model was not a forecast of change. It was a worse copy, trained on windows where copying works.

The old `t2.py` builder did apply a persistence filter, but it applied it to persistence **at T**, the leaked slot, and it sampled every two steps from the complete training layer. The CV numbers that justified the first submissions (onset 0.92, ongoing 0.87–0.93) were computed on that self-generated distribution. The 80 real training windows, when finally scored, were already lower, and the public board was lower again. Each evaluation was easier than the next one, and the model was chosen on the easiest.

There is a second distortion inside the ongoing filter they did use. Windows with persistence IoU above 0.9 are removed. The remaining label is “something changes.” A tree whose best feature is the current queue bit will still predict the current queue, because most cells do persist, and the loss is cellwise binary logloss. IoU of the set is not the loss. The model can be better than persistence on a mixture that still contains easy cells and worse than persistence on the hard slice. That is the pattern in the table: local ongoing 0.86–0.93, public ongoing ≈ 0.59.

### 5.4 Onset was reduced to a selector trick plus a location prior

On the training split, the earliest legal origin with a queue somewhere in the horizon has that queue at T+30 on 39 of 40 windows. The code then does two things:

- `windows_for_day` yields an onset example only when the last horizon slot is queued and the earlier five are not.
- `predict_real` sets the onset probability to 0 for every horizon step other than h = 6.

The timing regularity is a property of the sampler. If private uses the same contract, the first **selector-visible** queue is at the end of the horizon, and hard-zeroing the earlier steps is fair. The scored label is the latent queue on all six steps. A latent queue that occupies T+25 and T+30, or a selector that fires one to three slots off, turns those zeros into misses. The changelog records that the organizer’s ongoing trigger lands one to three slots after the reimplementation, attributes it to label noise, and stops. The onset decoder assumes the timing error is zero.

IoU arithmetic: if the true onset queue covers two timestamps and the submission covers one, with the same links, IoU is 0.5 before any spatial error. The public onset estimate of ≈ 0.64 is what a one-slot, partly-right location prior looks like. CV onset of 0.92 is what the same prior looks like when the locations are the ones it memorized. April’s seven incidents are new locations. The timing hack can survive. The location prior should not be expected to.

The contract already says what predicts onset: speed ratio and flow in the hour before the break. Those are in the feature list as `r_mean`, `r_min`, `qc_max`. They are then handed to a tree that is also allowed to split on `link`. On a single scenario the link split wins. The physical split never has to.

### 5.5 The proxies were used after they failed their own admission test

The changelog states a good rule and then does not follow it: a proxy must reproduce the leaderboard ordering before it is allowed to choose a submission.

| Proxy | What it said | What the board said | Used afterward |
|---|---|---|---|
| Day-grouped and month-blocked CV | model 0.86–0.93, far above persistence | model ≈ 0.59, persistence-at-T ≈ 0.73 | yes, for the first five probes |
| Pseudo-windows on the masked layer | old model 0.768 > persT 0.728 | the opposite | rejected, correctly |
| Episode backtest, pre-origin | validation 0.696 vs 0.685, SE ≈ 0.03 | direction only; the gap is inside the SE | yes, to justify multi-scenario training |
| Episode backtest with the contract trigger | persTm5 0.93 on private | legal persistence ≈ 0.60 | declared inadmissible, then more variants were still ranked by it |
| Hybrid on 80 train windows, latent proxy | add-only +0.055 ± 0.021 vs persistence-fill | not yet measured; pure model already lost on the board | adopted into `sub_safe_v3` |
| Onset decoders on pseudo-windows, n ≈ 290 | expected-IoU 0.692 / 0.725 | public onset ≈ 0.64 | adopted |
| Ineligible cells on that same onset proxy | +0.01 | unknown | adopted because a competitor does it |

The pseudo-window failure had a mechanical cause, visible in `iou_by_window` and in the way labels are built. Proxy IoU is computed on cells where the masked layer is observed (`known`). The real horizon is blanked. The proxy therefore scores the model on the cells the leaderboard does not ask about, and it drops the cells the leaderboard does ask about. An episode placed six to eight slots before a real origin so that “the horizon falls inside republished history” scores a quieter part of the same queue. Persistence there is 0.93 because nothing is moving yet. The scored window was chosen because something is moving. No reweighting of that sample turns it into the leaderboard.

After writing “no admissible local Task 2 ongoing proxy exists; the public leaderboard is the only admissible evidence,” the project kept accepting and rejecting decoders with local proxies. If the board is the only evidence, the only honest Task 2 experiment left is one pre-declared comparison, legal persistence against one physically specified alternative, on a single submission. Everything else in §5g–§5h is another local number from a distribution that has already been shown to rank methods backwards.

### 5.6 Training on March to predict April is the overfitting the overview warns about

`make_submission_queue(private_uses_val=True)` fits the validation rows with the train-only model and the private rows with a model trained on train **plus validation pseudo-windows**. The validation labels in that fit are noisy observed speeds, not the hidden queue labels, so this is not direct answer copying. It is still fitting March.

The justification in §5e is a private-episode proxy: train+validation 0.765 against train-only 0.719, and a later variant 0.898 against 0.831. Those are the proxies the same section says are uncalibrated (persistence 0.93 versus the board’s 0.60). A gain measured on the wrong conditional distribution does not have a sign on the real one. The public board cannot check it, because the March rows of the file were deliberately scored with the train-only model. The file contains two different Task 2 models, chosen so that the check the contest gives you is blind to the model that will be ranked.

March and April share a generator and do not share incidents. Capacity use was measured to be the same across splits, so there is no global capacity shift for the extra month to teach. What March contains that the training months do not is five specific incidents. April contains seven others. A tree with a link index, trained on March’s pseudo-labels, stores those five. On April it predicts them. The overview’s sentence about fitting validation’s incidents is this function, with the public board blinded by construction.

The same split-specific logic is in the Task 1 file: one model sees both months’ re-masked cells before either month is predicted (§4.4). Two tasks, same pattern. The pattern is “the test month is a training source, and we will not have a measurement until the private board.” That is the decision to decline.

### 5.7 The add-only hybrid locks in the train-set error mode

`hybrid_test` chooses the decoder on the 80 shipped **training** ongoing windows, labels denoised toward a latent proxy, model trained without frequency features. Add-only (keep every persisted cell, union any cell with p ≥ 0.5) scores 0.804 against 0.748 for persistence-fill, a paired gain of +0.055 ± 0.021. Variants that delete cells score worse. This was adopted, and the queued-cell share on ongoing windows rose from 0.116 to 0.139.

Read that result against the sampling design.

- The test that selected it is the training scenario, where ongoing queues are about twice as large as March and April (median predicted 39 versus 80). A decoder fit to large queues over-predicts small ones.
- Add-only can represent growth. It cannot represent dissipation. The 0.9 persistence cap means the scored window was kept because the set changes. Young afternoon queues in the training selector grow. A queue downstream of an incident also clears. Private has more incidents, not more copies of the training afternoon.
- IoU treats an extra cell and a missing cell the same. Raising the predicted share is a gain only while the model is under-calling. The changelog adopted the raise because the pure model was dropping cells on the **public** board. That is one month, forty ongoing windows, standard error on S_queue about ±0.03. A one-sided decoder is now frozen into the file that has to face a different month.
- The threshold 0.5 was not cross-validated under the legal information set on March. It is the default that won on the training latent proxy.

The pure model’s failure (it deletes the queue that is already there) is real. The repair is a propagation model whose state at the first horizon step is the legal history, and whose later steps move the upstream front at a shock speed estimated from that history. A union with a memorizing classifier is a patch for logloss. It has the sign of the training residual baked in.

### 5.8 Shockwave features failing inside a tree is not a test of kinematics

The paired test of shockwave features was −0.0001 ± 0.0007 on the tree (n = 3756). `tfb/growth.py` then applied a kinematic-wave decoder and lost 0.09 to 0.15 IoU to persistence, and the conclusion was that queues mostly persist for 30 minutes.

Both results are about those implementations.

- A tree that already has the current queue bit gains nothing from a noisy velocity column, because the loss does not care where the front is, only whether each cell is right, and most cells are unchanged. A null ablation means the column was unused. It does not mean the front is stationary.
- The scored windows are the ones where persistence IoU is at most 0.9. “Queues mostly persist” and “the test set is defined as the windows where persistence is a bad answer” are in the file at the same time. The second sentence is the sampling design. The first is an average over the windows the sampler throws away.
- A kinematic decoder that loses 0.10 IoU has the shock speed, the front, or the discrete link length wrong. `twin.py` documents the companion failure: the CTM over-holds queues and misses dissipation, 0.652 versus persistence-at-T−5 of 0.748 on the 80 training windows. Over-holding is a capacity and discharge bug. The calibration note in that file already points at the right observation: where a queued link is followed by a free link, the interface capacity is the measured discharge, which is how an incident appears without storing its location. One version that over-holds is an unfinished calibrator. It was treated as a class of models that had been tried.

The CTM is the model class that generated the data. It uses only legal history, ramp flows, and the released diagram, so a score on the training windows is a real estimate for April in a way no tree CV in this repo is. It was abandoned because version 0 lost to persistence on the easy windows, and nothing was submitted from it. The trees, which lose to persistence on the hard windows that are actually scored, were submitted repeatedly.

### 5.9 Ramp flows were dropped for the wrong reason

A real queue on these corridors is a discharge rate below the arriving mainline plus on-ramp flow. `twin.py` reads ramp flows on the history slots. The shipped Task 2 model uses on-ramp and off-ramp **indicators**, static, and not the flows. The changelog notes that ramp flows inside the horizon are not blanked (about 79.5% finite) and treats “we never read ramps” as cleanliness.

History ramps are inside the contract. Horizon ramps are the same kind of extra observation as slot T: they are released, and they are not part of the forecast information set. Refusing the horizon ramps is required. Refusing the history ramps removes the demand term from the only model that can move a queue. The forum question about the horizon leak is a rules question. It is not a modeling result.

### 5.10 Copying the public competitor copies their ceiling

`happyc0der`’s Task 2 stack is described as structurally the same tree: the differences adopted were ineligible-cell values, training on ineligible cells, a decoder over all cells, and onset augmentation at k = 4, 5. Their implied S_queue is about 0.74–0.76. The ineligible-cell flag (`T2S_INELIG`) moved the onset proxy from 0.692/0.725 to 0.703/0.732. Ineligible cells are `pct_observed < 75`. They are noisier observations. Using them as features is a small, legal source of evidence. Training on them as labels trains the threshold on the worst sensors. Either way the gain is a hundredth of IoU on a proxy that reads 0.70 where the board reads 0.64.

The top of the board is about 0.10 of S_total above this entry, on the order of 0.15–0.20 of S_queue if the rest were equal, and more if their physics is also better. A hundredth of IoU from a competitor’s feature flag does not close that. Matching their structure means matching a public notebook, which is how a crowd of teams ends up with the same March-specific errors and the same April surprise.

---

## 6. Task 4 is the one place a “don’t be clever” decision was right, and it was then over-generalized

V1 is NNLS with λ = 0.05 on the split’s own released counts and the split’s own weak prior. It is byte-identical to the organizer builder’s `public_reference.csv`. Public S_ODME ≈ 0.994. The official baseline at 0.836 is that solver pointed at the **training** counts. The entire Task 4 gain is “solve the split you were given.”

That transfers. Private ships its own counts and its own prior. The solver does not contain March’s incidents except through April’s own counts, which are inputs. The public competitor’s L-BFGS approximation sat near S_ODME 0.80. V4 (entropy / KL projection onto the raw prior) reached only 0.843 and a total of 0.55476, a hair above the train-count baseline 0.55343. V2 (smooth the prior, then L2) fell to S_ODME 0.468. V5 (gravity seed, then KL) fell to 0.622. The code also contains V3 (smooth, then KL), V6 (geometric shrink of the prior toward gravity), and the forensics in `t4_gravity.py`, `t4_decay.py`, `t4_structfit.py`, `t4_ramps.py`. All of them ask whether a low-dimensional distance or gravity correction can replace the prior’s per-OD residuals. The submitted ones say no. The prior’s per-OD deviations are signal. Smoothing them is the same instinct as smoothing a queue front: it looks like regularization and it deletes the answer.

`build_task4_odme_artifacts.py` records a separate trap: `base_od.csv` is one file per corridor and holds the private month’s prior, and an earlier builder regularized every split toward it. V1 as described uses `synthetic_weak_prior.csv` from the split’s own folder. Keep that path. Pointing the solver back at `base_od.csv` would fit April’s prior into the March submission.

Two errors still sit around this success.

- Submissions were spent on V2, V4 and V5 after the failure mode was already predictable from the competitor’s solve. The daily cap is 5. A submission is a measurement. V4’s total moved by about +0.001. V2 and V5 moved it down. That was three uploads to re-learn that the reference hugs the unsmoothed prior.
- The lesson recorded in the changelog is “the hidden reference is the organizer’s solve, so don’t improve it.” The lesson that is actually supported is “when the estimator is determined by released inputs and a convex objective, solve that objective on each split.” It does not say “stop improving the reconstruction” or “stop improving the forecast.” Those are not references the organizer hands you. The physics ceiling at 0.96 and the queue score moving from 0.30 to 1.0 are the room. Treating Task 4’s solved status as a reason to stop modeling is how a 0.99 ODME term got paired with a 0.64 physics term.

Also, local Task 4 scoring against the builder’s own reference is circular. `RUN_LOCAL.md` says so. The public 0.994 is the real evidence, and it is enough. No further Task 4 variant belongs in the remaining submissions unless a change beats V1 on a quantity the board has already shown it cares about (`S_dev` punishes both the untouched prior and an exact crazy fit). It will not.

---

## 7. How the thinking produced these decisions

The individual model choices follow from a few habits. They are still in the later sections of the changelog, which is why a new feature pass will repeat them.

**The contract was a source of features, read after the model existed.** The selector, the ban on slot T, the 0.9 persistence cap, the onset speed-ratio clue, the capacity multiplier, and the instruction that tasks share a physical state were all in `competition_contract.json`, `synthetic_release_v1.json`, or the README before the first GBDT was trained. The selector was reverse-engineered for hours. Slot T was the best submission. The persistence cap was discovered empirically and then not applied in the rules-safe training sampler. The onset clue was never the training target. Reading documents after a leaderboard result, to explain the result, is how the project learned the rules.

**Local CV was treated as a population.** Day-grouped CV, month-blocked CV, and family holdout are three different conditionings, and only the last one even changes the roads. None of them is “a new draw of incidents.” The first number that was a new draw was the public board, and it inverted the Task 2 ranking. After that, the correct update is that every CV number in the changelog is an upper bound from the training scenario. The update that was made instead was to build another proxy on March and April and to keep the tree.

**A proxy was defined as anything that returns an IoU.** Admission was “does this number look lower than CV.” The stated rule — the proxy must reproduce the board’s ordering — was applied once, to kill the pseudo-window proxy, and not applied to the hybrid, the onset decoder, the ineligible-cell flag, or the train+validation fit. Those were adopted because the new number was higher than the previous number on a sample the author had already called uncalibrated.

**The anti-overfitting policy prescribes the leak.** Changelog §7 says to prefer “exact ODME, persistence-at-T, physics features” over location memory. Persistence-at-T is the slot the organizer removed. The policy also says the public board is “40 Task 2 windows per corridor split (4 families).” The release has 8 scored panels and 10 windows each, 80 windows, 40 per condition. The published per-window IoU sd on the persistence baseline is 0.375, not 0.3 (`synthetic_release_v1.json`). The standard error of S_queue is about 0.04, and of one condition about 0.06. The policy’s own threshold (“do not chase deltas below ~0.01 total”) is tighter than what 80 windows can resolve, and the text undercounts the windows while quoting an SE that assumes the larger count. Section 8 then lists the next experiment as “a persistence-at-T backbone plus a model that only predicts changes.” After the critic had named the leak, the written plan was to build on it. The same section’s robust submission — one maximally simple file beside the best — was never built. Interpolation plus legal persistence plus V1 ODME is still not a file.

**The leaderboard was used as a coordinate-wise optimizer.** The scored uploads in the changelog are the Task 4 set (train-count baseline 0.55343, V2 0.47983, V4 0.55476, V5 0.51071, V1 0.58501) and the Task 2 probes on a fixed GBDT state (full model 0.80613, organizer persistence 0.71192, onset zeroed 0.71024, unfiltered ongoing 0.80946, persistence at T 0.82669). That is on the order of ten measurements. Five of them were spent the day the cap is five, to back out onset ≈ 0.64 and ongoing ≈ 0.59. A derived gap of 0.05 between conditions is inside the error of 40 windows. The research plan was then organized around those point estimates: “onset is 0.64, so change the onset decoder,” “ongoing is 0.59, so change the ongoing decoder,” “the gap to 0.87 is queue.” The position that number bought is about **#92**, with roughly 90 teams at or above 0.827 and the leader at 0.924. The anti-overfitting section says not to chase total deltas under ~0.01 and not to iterate on the board. The probe program is that iteration. The organizer-persistence upload at 0.71192 is the legal queue baseline sitting under the same state model. It was recorded as a weak score. It is the score the legal information set actually had, because onset persistence is empty by construction and the script uses T−5. Beating 0.71192 with slot T did not produce a forecast.

**Failure was repaired inside the same hypothesis.** The model drops persisted cells, so forbid dropping. The CV–board gap might be physics, so add a self-supervised residual and do not measure physics. The CTM over-holds, so return to the tree. Shockwave features do nothing inside the tree, so conclude that queues persist. Each repair is locally responsive and leaves the hypothesis — “a per-cell tree can be the traffic model” — untouched. The hypothesis is what family-holdout, the physics gap, and the inverted Task 2 ranking all contradict.

**Competitor artifacts were used as ceilings and then as templates.** An early ceiling of 0.94 state / 0.72 physics, taken from a competitor’s notes, was false, and the GBDT beat it. Good correction. The later move was to diff `happyc0der` and port the flags that differed. A false ceiling causes under-ambition. Copying a public stack causes a shared overfit. Both are substitutes for a generative model.

**Two information sets were mixed and then called a robustness check.** Training rows were built from the complete layer; inference sees the masked layer. The realistic-masking check trained and tested on the same dates, so the model had already seen those days’ unmasked targets. Slot T, day context before T−60, and horizon ramp flows were all used or considered because they are correlated and present on disk. Presence in the parquet file is not membership in the task’s information set. The participant contract for Task 2 is the released history `[T−60, T)`. The participant contract for Task 1 is the masked month plus the network. Private evaluation will apply those contracts. A feature that only exists in the training cache will look like skill until the cache is gone.

**The submission gate encodes the public board as the objective, and v3 breaks the gate.** Section 7b requires an admissible proxy, an expected public gain above ~0.005, no unresolved rules risk, a critic review, and no weak change bundled alone. Rule 3 is the right rule and was already being broken by the queued 0.82669 file. Rule 2 says a legal method that might lose 0.02 on March and gain it back on April cannot be submitted, and a +0.001 cosmetics change can, if someone rounds the expectation. `sub_safe_v3` was then scheduled in the open knowledge that its expected public score is about 0.80, below 0.82669. Training the private rows on March was still “pending the forum answer,” which is an open rules risk, so rule 3 fails. Self-supervision, ineligible cells, the onset decoder and the hybrid are four weak changes bundled together, so rule 5 fails. `build_safe3.sh` turns on `T2S_INELIG=1`, merges `state_submission_ss.csv` with `queue_safe.csv` and `odme_ref.csv`, and does not submit a single-change control. v2, the fallback, has already stripped day-context and still carries self-supervision and the expected-IoU onset decoder. Neither file is a clean test. If v3 scores 0.80, there is no way to know which piece moved the number.

**The de-memorized model was cancelled for losing to persistence.** `sub_auto_A` was the no-identity ongoing model. `t2.py` `auto_build` picks the onset feature set that **maximizes** family-blocked IoU, which selects `full` (0.413) over `no_ident` (0.241) and therefore keeps the memorizer whenever memory beats an empty mechanism. Ongoing in that script is hardcoded to `no_ident_no_freq`. The run was cancelled because that family “scores below persistence on validation.” The proxy that cannot rank the main model was allowed to kill the one variant that had removed link identity. `sub_auto_B` was a second copy of 0.82669. Cancelling B was right. Cancelling A was the research result being thrown away.

**Compute was used to scale the tree.** The box is an 8-vCPU instance. A corridor CTM for a 30-minute horizon is a small CPU job, and `twin.py` already runs. The constraint that mattered was not the GPU quota. It was that the iteration loop was “more leaves, more features, more windows,” which saturates at the CV numbers already in §4 and §5, and which cannot represent a shock.

---

## 8. Decisions that should stand

These are listed so they are not “fixed” in the next pass.

- Task 4 V1, per split, λ = 0.05, the split’s own prior, no smoothing. Do not spend another submission on ODME.
- The rules-safe information set for Task 2: history `[T−60, T)` only, no corridor state at T, no same-day context before T−60, no ramp flows from inside the horizon. History ramp flows stay available.
- Cancelling the auto-submit that repeated 0.82669. Do not select that file for the final.
- The admission that the pseudo-window proxy ranks methods backwards. Do not revive it under a new name.
- Speeds clipped to a physical range, finite outputs, full row coverage. Keep the sanity checks. They are not the model.
- The measurement that validation and private ongoing queues are about half the training size, and that capacity quantiles do not shift. Use it as a reason to reject decoders fit on training queue sizes. Do not go looking for a global capacity multiplier the quantiles have already refused.
- Not training the reported selector-match into a claim about labels. Onset 39/40 used raw speeds; denoising made it worse; the score uses the latent state. Hold those three facts apart.
- Post-horizon reads (`t2_leakcheck.py`): the state at T+95 does not recover the horizon (IoU about 0.33–0.58, below persistence). The 60-minute buffer does its job. Do not reopen future interpolation. The same file labels persistence at T as “causal.” It is not. T is the extra slot.
- The eligibility check that horizon `elig` matches `pct_observed ≥ 75` (76.4%). That hypothesis is closed.
- Frequency features dropped for ongoing. Keep them dropped, and drop `link` / `panel` / `pos` with them. Dropping frequency and keeping the link index leaves the larger memory in the model.
- Selector search stopped at ongoing agreement 10/40. Every variant tried (persistence filter, two-pass scan) was also 10/40, and `D7_I10_W` was 5/5. The residual is not a filter bug. Do not resume the search. The windows to forecast are already in `window_index.csv`.

---

## 9. What another iteration has to do differently

The private month will not reward a finer tree, a new proxy on random cells, or a decoder threshold chosen on the 80 training windows. The following is the bar for the next submission, given the evidence already in the changelog.

**One state, used twice.** Fit a reconstruction whose unknowns are a density (or speed) field on the released diagram, with flows implied by the diagram and by conservation, ramp flows entering only where they are eligible, and blanks filled by the characteristics rather than by a residual tree. Score it on train by S_state and by a conservation residual that does not give the ground truth S_LWR = 1 for free. The GBDT remains a baseline. The new model replaces it only when the conservation residual improves on a March re-mask and an April re-mask under the real regime rates, including R3, and S_state does not fall by more than the physics gain is worth. A gain of +0.003 S_state with physics unmeasured fails this bar.

**Task 2 is a rollout of that state.** Initial condition from the legal history only, forward 30 minutes, queue bit from latent speed against `0.6 × v_f` after the noise model, not from a second classifier. Bottlenecks come from measured discharge where a queued link meets a free link, which is the incident signal that does not store a link id. No `link`, `panel`, or training-set frequency column. No fit on March’s labels for April’s rows. One model, both splits.

**The training distribution has to be the hard one.** If synthetic windows are used, apply the published filters: history coverage, the contract definition of onset versus ongoing, spacing, and persistence IoU ≤ 0.9 computed at T−5. Then check, before any fit, that the synthetic windows match the released training windows on queue size, time of day, and persistence. The current sampler fails that check (ongoing selector agreement 10/40, and the 0.9 filter is absent). A model selected on the failing sampler is not a candidate.

**Stop when the proxy and the board disagree.** The standing disagreement is: every local Task 2 score says the tree beats persistence; the board says legal persistence is at least as good as the tree and leaked persistence is better. Until a proxy reproduces that ordering, it cannot approve a Task 2 change. The CTM is allowed to be bad on the first version. It is not allowed to be discarded for losing to persistence on windows the sampler defined as easy, while the tree is kept after losing on the windows the contest defined as hard.

**One submission, one question.** With a cap of five per day and an S_queue error of ±0.03, the next upload should change one component against the legal baseline: either the conserved reconstruction against the current GBDT, or the rollout against legal persistence, with the other tasks held at V1 ODME and the legal queue or the current state. Bundling self-supervision, ineligible cells, a new onset decoder, an add-only hybrid, and a March-trained private model into `sub_safe_v3` makes the result uninterpretable. If that file is already queued, treat its score as a single contaminated measurement and do not iterate on it.

**Do not select 0.82669.** It is the leaked information set. April will score it, and a code check will see `s0+1` in `t2.py` and the state-at-T persistence that produced the number. The file that can be defended is the one whose Task 2 inputs are the published history and whose Task 1 field conserves vehicles on a corridor it was not fit to.

The public 0.82669 is what this process was optimized to produce: a legal ODME solve, a cellwise regressor that wins a same-scenario RMSE, and a queue answer that reads one observation past the contract. The private month scores the generator’s next draw. Nothing in the current stack is a model of that generator, and every selection step after the first leaderboard inversion was a way to avoid building one.

---

## 10. Task 2 results that were measured and then folded into the same model

These are easy to “retry” because each one looks like a small positive. They are already decided.

- **Hard-window filter.** Training only on windows with persistence IoU ≤ 0.9 gained about +0.004 locally and about +0.022 of S_queue on the public board against the unfiltered tree, and still lost to persistence. The public gain is the training distribution being nudged toward the 40 scored windows. April uses the same filter rule and a different incident draw. The filter is a property of the sampler, already in the json. Re-applying it does not create a propagation model, and +0.022 on the board is inside the condition-level error.
- **Bigger tree.** `num_leaves=255`, 1000 rounds, gained about +0.011 day-grouped IoU. Untested under family blocking. Day-grouped CV is the split that rewards memory. A larger tree on that split memorizes more of it. `compare()` in `t2.py` also tried `num_leaves=31`. Neither replaces the feature set.
- **Realistic masking.** Forcing the training day to look masked moved a local score from 0.907 to 0.905. The check trained on the same dates it tested, so the model had already seen those days’ labels. A null result from a leaky check does not clear the train/serve skew in `t2.py` `visible()` (complete layer at train, masked layer at inference). The `known` mask used at inference was flagged in the critic pass and is still an assertion, not a tested equality with the scorer’s eligibility, on the horizon cells.
- **Shockwave columns inside the tree.** Paired delta −0.0001 ± 0.0007 (n = 3756). The columns were ignored. See §5.8.
- **`growth.py`.** Empirical tail motion versus persistence: **+0.005 ± 0.009** on the latent proxy, a zero. Kinematic wave speed: **−0.09 to −0.15**. The head was held at the downstream end of the queued region. A null empirical velocity on training windows, where queues are large, does not license “queues do not move” on the windows selected for persistence IoU ≤ 0.9. It licenses “this estimator of the tail is noise.”
- **CTM v0 / v1.** History-calibrated Godunov scheme: ongoing 0.652. Adding a demand trend made it **0.648**. Both against T−5 persistence 0.748 on the 80 training windows. The second version changed a boundary time series and left the discharge bug in place. Over-holding means the bottleneck capacity stays too low, so the queue never clears. That is the parameter the calibration comment already names (measured discharge at a queued-to-free interface). One failed trend is not a test of that parameter.
- **Hole-fill.** On a denoised proxy, T−5 persistence went 0.726 → 0.748, and against noisy labels 0.748 → 0.733. Onset expected-IoU on the pseudo-windows went **0.692 / 0.725 → 0.686 / 0.710** when hole-fill was added, so it was correctly kept off onset. It stayed on for ongoing, where the add-only hybrid hole-fills the persisted mask and then unions p ≥ 0.5. The sign of hole-fill depends on which label you believe. The scored label is latent. The training label is noisy. The decoder mixes them.
- **Onset decoders on pseudo-windows (n ≈ 290).** Expected-IoU 0.692 / 0.725, contiguous block 0.62 / 0.67, top-k of a fixed size 0.42–0.52. Expected-IoU was adopted. Those pseudo-windows are the proxy that reads ~0.70 where the board reads ~0.64, and they were built with the hard rule that only the last horizon step may be queued. Top-k and block lost on that distorted sample. They should not be revived on it. They also should not be treated as having been rejected on the leaderboard. They were never submitted.
- **Ineligible cells.** Onset proxy +0.01 on both months. Adopted via `T2S_INELIG=1` because the competitor trains on them. Same proxy problem. Ineligible cells remain legal features. They are a bad training label.
- **Multi-scenario numbers, all on the inadmissible episode proxy.** Private episodes: T−5 persistence 0.687, train-only 0.719 (no-freq 0.717), train+validation 0.765 (no-freq 0.753). A later cut: 0.898 versus 0.831. The holed-history control (train with holes, 0.830 ≈ train 0.831) says the gain is not an artifact of blank handling. It does not say the proxy matches the board. Validation episodes, train-only: no-freq 0.704 > full 0.684 > persistence 0.665, which is why frequency was dropped, on a proxy whose persistence sits at 0.93 in the contract-trigger version against a board value near 0.60. A control that rules out one bug does not admit the proxy.
- **Queue size.** Training ongoing windows carry about twice the queued cells of March and April (median about 80 versus about 39). The add-only hybrid raises the predicted share from 0.116 to 0.139 and was chosen on the training windows. Smaller true sets make extras more expensive. April has seven incidents to March’s five, so the share of windows where the queue is a new object, rather than the afternoon bottleneck the frequency table remembers, goes up.

---

## 11. Ledger

Every experiment in the changelog and the `tfb/` scripts, with the decision it actually supports. “Closed” means do not spend another submission or another proxy on it. “Open, but not as a tree tweak” means the question stands and the previous method cannot answer it.

| Experiment | Result | Verdict |
|---|---|---|
| Task 4 NNLS λ=0.05 on the split’s own counts and prior (V1) | Public total 0.58501, S_ODME ≈ 0.994, matches `public_reference.csv` | Keep. Closed. |
| Task 4 on training counts (official baseline) | 0.55343, S_ODME 0.836 | The bug V1 fixed. Do not regress. |
| Task 4 V4, KL on the raw prior | 0.55476, S_ODME 0.843 | Closed. Prior residuals matter; entropy projection does not recover them. |
| Task 4 V2, smoothed prior + L2 | 0.47983, S_ODME 0.468 | Closed. |
| Task 4 V5, gravity seed + KL | 0.51071, S_ODME 0.622 | Closed. |
| Task 4 V3, V6, distance decay, ramp-count forensics | Local fits, not the board | Same hypothesis as V2/V5. Closed. |
| `base_od.csv` as the prior | Holds the private month | Do not use. V1’s per-split file is the legal input. |
| Historical-mean profile | S_state ≈ 0.69 | Floor. |
| Same-link time interpolation | CV far above the profile; public contribution inferred, not submitted | The robust Task 1 baseline. Still no direct upload. |
| GBDT v1 on interpolation residuals | CV S_state 0.941 / phys 0.729; public state+physics ≈ 0.423 | RMSE win, physics does not travel. Not the final reconstructor. |
| GBDT v2, +16 features | +0.0003 J on CV | Closed. |
| Density-aware `(v, q)` decoding | Local J worse at every blend | Killed by the wrong physics proxy. The constraint is still required. Not closed as an idea. |
| Speed clip to [5, 140] | 29 cells had been below 5 | Sanity check only. |
| Own-month profile | Proxy S_state +0.002 | Legal offline feature. Too small to submit alone. Unmeasured on S_LWR and on R3. |
| Self-supervision on re-masked March and April cells, one pooled model | Proxy +0.003 S_state | Test-month fit, physics unmeasured, both hidden months in one tree. Do not ship. |
| Task 1 re-mask proxy | Profile ≈ 0.69, interpolation ≈ 0.90, GBDT ≈ 0.93 on both hidden months | Admissible for coarse S_state only. Silent on incidents, R3, and conservation. |
| Latent smoothed rhs for S_LWR | Truth falls to 0 | Proxy rejected. Does not end the physics task. |
| Structural Kalman baseline in the repo | Not run as a candidate | The published hint that Task 1 is a state estimator. |
| Digital-twin FD check | Congested cells on the released triangle, 96–99% within 5 km/h | Evidence for a constrained reconstructor. Wasted as a note. |
| Neighbor-flow sd check | Changelog contradicts itself (≈190 vph vs “noise level”) | Wrong conservation equation. Ignore the “matches noise” sentence. |
| Capacity quantiles across splits | No shift; `capacity_multiplier` 1.2 not visible | Closed. Do not hunt a global capacity change. |
| Slot T persistence as the ongoing answer | Public 0.82669, ongoing ≈ 0.73 | Illegal information set. Do not select. |
| Organizer persistence script (T−5, both conditions) | 0.71192 under the GBDT state | Legal queue floor. Onset half is ~0 by construction. |
| Rules-safe ongoing persistence ≈ 0.60 | Inferred, not a standalone ongoing-only upload | Treat as an inference from the 0.71192 file. A direct A/B is the only clean measurement left, and only for one pre-declared model. |
| LightGBM queue, day-grouped / month-blocked / 80 train windows | 0.86–0.94 local | Training-scenario upper bound. Not evidence. |
| Leave-family-out | Onset 0.413 (0.241 without identity); ongoing still above persistence | Onset is memory. Ongoing family-holdout is not a new incident draw. |
| Public full queue model | 0.80613, S_queue ≈ 0.62; onset ≈ 0.64, ongoing ≈ 0.59 | The transfer result. The tree loses to persistence. |
| Onset zeroed, unfiltered ongoing | 0.71024 and 0.80946 | Diagnostic probes. Condition split is inside noise. |
| Onset hardcoded to T+30 only | 39/40 train windows; host-cited regularity | Sampler fact. Legal if the contract is unchanged. Does not locate the queue. Latent label can occupy more than the last step. |
| Denoised labels for the selector | Match fell to 23–29/40 | Selector uses raw speeds. Score uses latent speeds. Do not mix them. |
| Selector reimplementation | Onset 39/40, ongoing 10/40 in every variant, one corridor 5/5 | Stop. Forecast the published windows. |
| Pseudo-window proxy | Ranked the model above persistence; the board did the opposite | Closed. Inadmissible. |
| Episode proxies, with and without the contract trigger | Persistence from 0.67 to 0.93 depending on placement; board ongoing near 0.60 | Closed as a selection tool. |
| Train on March pseudo-labels, predict April | Proxy gain, invisible on the public board, forum unanswered | Overfit by construction. Do not ship. |
| Day context before T−60 | Used, then stripped in v2 | Stay stripped. |
| Horizon ramp flows (not blanked, ~80% finite) | Not used; forum draft | Keep unused. History ramps are a different input and belong in a rollout. |
| Post-horizon T+95 | IoU 0.33–0.58 < persistence | Closed. Buffer works. |
| Shockwave features, `growth.py`, CTM v0/v1 | Null or large loss versus persistence | The implementations failed. The model class did not get a real test. See §5.8 and §10. |
| Add-only hybrid, p ≥ 0.5, hole-filled T−5 | +0.055 ± 0.021 on 80 train windows, latent proxy | Fit to large training queues. Cannot represent clearance. Do not carry forward on that test. |
| Expected-IoU / block / top-k onset decoders | Expected-IoU won a bad proxy | Not leaderboard evidence. See §10. |
| Ineligible-cell features and labels | +0.01 onset proxy | Not worth a bundle. |
| Hard-window filter, bigger queue tree, “realistic” mask | +0.022 board, +0.011 CV, ~0 local | See §10. Closed as ways to save the tree. |
| `sub_auto_A` no-identity | Cancelled; below persistence on the local proxy | The cancellation is the mistake. Identity stays out. |
| `sub_auto_B` | Copy of 0.82669 | Correctly cancelled. |
| `sub_safe_v1` → v2 → v3 | v3 is self-sup state + ineligible Task 2 + expected-IoU onset + add-only ongoing + V1 ODME, private rows trained on March | One contaminated measurement if it has already been sent. Do not iterate on the difference between v2 and v3. |
| Operational: scp to the wrong directory, OOM on the full Task 2 table, `pkill` matching its own ssh line, template column clash | Lost time, one stale run | Process. The OOM pushed the code toward `load_cond` subsampling, which is how several CV numbers were computed on 6–12% of windows. |
| U1 legal onset tree | Public 0.80579; onset IoU 0.6258 | Keep as the March measurement of the memorizing tree. Replaced for selection by J1. |
| U2 legal ffill ongoing | Public 0.82600; ongoing IoU 0.7381; +0.020 vs U1 | Keep as the ongoing rule. Matches latent train ffill. Closed as a search. |
| J1 joint onset analogue | Public 0.84461; onset IoU 0.7499 | Selected legal file. Train noisy 0.96 did not transfer; median 0.79 did. Do not retune. |
| L0 time-interpolation state | Public 0.81733; −0.00867 vs U2 | GBDT stays. Interpolation remains the local baseline, not the submission. |
| Onset size / joint-mask oracles | Latent size 0.784; joint analogue latent 0.785 | Remaining onset headroom ≈ +0.005 total. Closed. |
| Ongoing front oracle / analogue / CTM v2 | Oracle +0.06 to +0.10 IoU; analogue CI covers 0; CTM expected 0.59, LODO always ffill | Dynamics class failed the gate. Closed. |
| Structural KF / ASM / FD waves | All lose to time interpolation on Dec–Feb | Graph-smoother branch closed. |
| Regime-balanced GBDT | J 0.439 vs 0.4387 | Zero. |
| Full-row GBDT | J 0.4405 vs 0.4387 | Below the 0.005 gate. Closed as a submission. |
| Leak-free month-blocked GBDT | J 0.4390, same as leaky 30% | Profile leak was not the gain. Closed as an expansion. |

---

## 12. What this file is for

The next session does not need another leaderboard theory. It needs to stop treating 0.82669 as the score to defend, stop selecting Task 2 changes on proxies that invert the board, and stop fitting March to predict April. The open scientific problem is a single conserved state on the released diagram, rolled forward 30 minutes from the legal history, with bottlenecks taken from measured discharge. Every row in §11 that is marked closed is a result this project already paid for. Repeating it is how the private month gets a more elaborate version of the same March fit.

---

## 13. Honest score frontier (measured 2026-10-01)

This section is the decision record for the frontier plan. Arithmetic is from `tfb/score_frontier.py`. Task 2 oracles are from `tfb/t2_oracles.py` with date-clustered 95% intervals. Models were chosen on train-only grouped validation. March was used only as a pre-registered one-change probe. April was not fit.

### 13.1 What the public probes actually identified

| File | Public | Isolated change | Identified quantity |
|---|---:|---|---|
| R0 | 0.71192 | GBDT + organizer persistence + V1 ODME | non-queue bundle + organizer S_queue 0.3017 |
| U1 | 0.80579 | onset only | onset IoU **0.6258** |
| U2 | 0.82600 | ongoing only vs U1 | ongoing IoU **0.7381**; S_queue **0.6820** |
| J1 | 0.84461 | onset only vs U2 | onset IoU **0.7499**; S_queue **0.7440** |
| L0 | 0.81733 | state only vs U2 | GBDT − interpolation = **+0.00867** total |

Non-queue identified from R0 is

```text
0.71192 − 0.30 × 0.3017 = 0.62141
```

Checks: `0.62141 + 0.30 × 0.6820 = 0.82601` (U2) and `0.62141 + 0.30 × 0.7440 = 0.84461` (J1). U2 − U1 = +0.0202, inside the +0.024 training expectation for legal forward-fill. Slot T at 0.82669 is not better than legal U2; J1 is the first legal file above that leak. L0 lost, so the GBDT stays.

Do not read U2 − R0 or J1 − R0 as one condition.

### 13.2 Feasibility, not a target

`S = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`.

| Non-queue premise | Value | Q for 0.90 | Q for 0.92 | Q for 0.94 | Max at Q=1 |
|---|---:|---:|---:|---:|---:|
| Measured from R0 | 0.62141 | 0.929 | 0.995 | **1.062** | 0.921 |
| §5j working ceiling | 0.64405 | 0.853 | 0.920 | 0.987 | 0.944 |
| Exact-answer components | 0.69454 | 0.685 | 0.752 | 0.818 | 0.995 |

A total above **0.94 cannot be promised** on the measured non-queue: it requires S_queue > 1. Even the §5j working ceiling needs S_queue ≈ 0.987, which is not a forecast, it is the label. The published 0.924 leader needs S_queue ≈ 0.93 at that working ceiling, or a higher physics term than we have identified. That is no longer an accusation of an out-of-contract input. It is a statement that we have not measured their physics.

A total above **0.90 is not supported by any legal method that survived validation.** J1 is 0.84461. The remaining representation oracles do not add 0.055.

### 13.3 Task 2 information-set ceilings (80 train windows, date-clustered)

Onset, 40 windows, 9 dates:

| Method | Noisy | 95% date CI | Latent / median proxy | 95% date CI |
|---|---:|---|---:|---|
| Frequency prior, fixed top-3 | 0.6211 | 0.383–0.722 | 0.5579 | 0.381–0.761 |
| Same prior, oracle size | 0.9325 | 0.637–0.997 | 0.7843 | 0.558–0.935 |
| Exact h=6 mask | 1.000 | — | 1.000 | — |
| Joint analogue `r6_analog10_noisy` (leave-month) | 0.9597 | se 0.020 | 0.7852 | se 0.051 |

Public U1 onset 0.6258 matched the noisy top-3. Public J1 onset 0.7499 matched the **median-proxy / size-oracle band**, not the 0.96 noisy leave-month number. The noisy analogue overfit the training scenario. The conservative number transferred. Remaining onset headroom against the latent size oracle is about +0.03 IoU ≈ **+0.005 total**. That is below the 0.005 upload gate once date noise is counted. Stop onset search.

Ongoing, 40 windows, 13 dates:

| Method | Noisy | Δ vs ffill (95% date CI) | Latent | Δ vs ffill (95% date CI) |
|---|---:|---|---:|---|
| Legal ffill | 0.7551 | — | 0.7330 | — |
| Front family, leave-date rule | 0.7336 | −0.051 to −0.006 | 0.7483 | +0.002 to +0.053 |
| Front family oracle (not a model) | 0.8114 | +0.030 to +0.161 | 0.8327 | +0.095 to +0.221 |
| Front analogue (chosen leave-date) | 0.7799 | +0.025 ± 0.033 | 0.7658 | +0.033 ± 0.038 |
| CTM v2 expected ensemble | 0.5878 | −0.167 ± 0.049 | 0.5785 | −0.155 ± 0.047 |
| CTM v2 leave-date | 0.7551 | 0 (always picked ffill) | 0.7330 | 0 |
| CTM v2 oracle | 0.7730 | +0.018 ± 0.004 | 0.7507 | +0.018 ± 0.004 |

Public U2 ongoing 0.738 sits on the latent ffill, not the noisy 0.755. The front analogue’s interval covers zero. The CTM recovery ensemble is worse than ffill and its representational headroom is +0.018 IoU ≈ +0.003 total. Both branches are closed. Legal forward-fill remains the ongoing answer.

Contract audit, read-only: train template targets equal eligible blanks exactly. Validation has 134,965 extra eligible blanks not in the template; private has 135,887. Those cells are Task 2 horizon/buffer blanks. They are not Task 1 targets. Do not fill them as a scorer exploit.

### 13.4 Task 1 / 3 ablations and the graph-smoother gate

Faithful June–November profiles, December–February targets, official R1/R2/R3 masks, family-then-regime macro:

| Model | S_state | S_physics local | J_local |
|---|---:|---:|---:|
| Weekday profile | 0.7208 | 0.3646 | 0.3070 |
| Official-style structural KF / RTS | 0.8952 | 0.5298 | 0.3928 |
| Time interpolation | 0.9058 | 0.6742 | 0.4182 |
| ASM, default waves (100, −18) | 0.7995 | 0.4055 | 0.3407 |
| ASM, FD waves (~115, −25) | 0.7983 | 0.4045 | 0.3401 |
| GBDT v1, 30% rows (changelog) | 0.9408 | 0.7292 | 0.4387 |
| GBDT, 30%, regime-balanced | 0.9411 | 0.7304 | 0.4390 |
| GBDT, all June–Nov rows | 0.9430 | 0.7361 | 0.4405 |
| GBDT, leak-free profiles, 30% | 0.9413 | 0.7305 | 0.4390 |

The structural Kalman in `t1_ablate.py` is the same local-level + RTS smoother as `repo/src/task1/baseline_task1_structural_kf.py`. It loses to interpolation on every regime, including R3. Adaptive smoothing as a reconstructor is worse still; switching the waves to the released fundamental diagram does not help. The plan’s instruction was: implement a directed graph/RTS smoother only if these cheap tests established signal. They did not. That branch is closed. Do not spend another week building a fancier smoother that is the same estimator.

Regime balancing is a +0.0003 J zero. Training on every row instead of 30% is +0.0018 J, about +0.002 total if every bit transferred, and historically about a third of J survives a new month. Both are below the 0.005 upload gate. R3 remains the hard slice (tinterp physics 0.535; full GBDT 0.610). Existing `t1feat` parquet files do not contain `iv`/`iq`; the ineligible-anchor ablation needs a feature rebuild and is not justified by a 0.002 effect. `full_balanced` was not run.

L0 on the public board is the Task 1 decision that counts: interpolation lost **0.00867** to the current GBDT under identical queue and ODME. Keep the GBDT. Retire interpolation as a submission, not as the robust baseline in local tests.

The leak-free feature rebuild (`t1_faithful_cv.py`) rebuilds weekday profiles from June–November only, so December–February do not help fill their own profile features. The 30% plain eligible model scores **0.9413 / 0.7305 / J 0.4390**, the same as the leaky tables. The profile leak was not the CV gain. The GBDT’s local +0.021 J over interpolation is real on a clean month split; L0 says about **+0.009** of that survives March. That is already in the selected file. Ineligible-anchor features were not rebuilt: `t1feat` does not contain `iv`/`iq`, and nothing here is large enough to pay for another feature pass.

### 13.5 Combined budget after J1

Plausible legal pieces in hand:

```text
non-queue  0.62141   (identified)
onset      0.7499    (J1, public)
ongoing    0.7381    (U2, public)
S_queue    0.7440
S_total    0.84461
```

If every remaining **representation** oracle were realized and the §5j working non-queue were free:

```text
onset  0.785 (latent joint / size oracle)
ongoing 0.833 (latent front oracle)
S_queue 0.809
non-queue 0.644
S_total 0.887
```

That is still below 0.90, and we do not have models that realize those oracles. The CTM and front-analogue attempts failed the 0.005-total gate. Physics local 0.73 versus the public state+physics implication near 0.64 is a mapping gap, not a stored +0.013 we can pick up with another tree.

The only arithmetic path to 0.90 that does not invent a component is: non-queue near the exact-answer 0.695 **and** S_queue near 0.81. Exact-answer non-queue includes held-out measurement noise. It is not a predictor ceiling. Treating it as one is how the earlier drafts promised 0.91.

### 13.6 Selection rule

**Current legal selected file: `sub_J1.zip` (md5 `78837f4bfaf68d06694b3e6895bb94ed`), public 0.84461.**

Keep with it:

- State: current GBDT (`work/t1/state_submission.csv`). L0 rejected it on the board.
- Ongoing: `ffill_inel`. U2 confirmed the rule. CTM v2 and the front analogue did not beat it.
- Onset: joint analogue `r6_analog10_noisy`. Do not retune neighbours, radius, or label after seeing 0.84461.
- ODME: V1. Closed.
- Robust unuploaded file: `sub_Rff.zip` (organizer onset, ffill ongoing, same state/ODME). Keep it. Do not upload it unless J1’s onset is later shown to be illegal or unreproducible.

Do not upload:

- Slot T, `sub_probe_persT`, or any file that reads the origin cell.
- CTM v2, front analogue, add-only hybrid, no-identity retune of U1, March-to-April trees.
- Another onset decoder because J1 missed the 0.96 leave-month number. The 0.75 public number *is* the result.
- Task 4 variants.
- A physics “fix” that scores a topology-derived flux or a latent rhs.

Daily cap leftover after J1 and L0: one slot. Leave it unused unless a pre-registered file that was already built, isolated, and above the 0.005 gate is sitting on disk. None is.

### 13.7 What this does not authorize

It does not authorize “we will get 0.94.” Measured non-queue makes that impossible.

It does not authorize “we will get 0.90” as a forecast. The legal frontier that can be defended from grouped validation and the public probes is **0.845**, with a generous oracle envelope near **0.87** that no implemented model reached.

It does not authorize closing Task 3 forever. The official exact-answer physics is still 0.96 and the public implication is still near 0.64. That gap is real. The failed ASM/KF/CTM tests say the next attempt has to change the right-hand side it is scored against, not add another smoother that loses to linear interpolation.

It does not authorize using the extra eligible blanks on validation/private as Task 1 targets.

U1 and U2 were the right uploads. J1 was the one new queue candidate that cleared the gate. L0 was the one Task 1 A/B. That is the whole honest movement of the day: 0.71192 → 0.84461, all legal, one change at a time.

### 13.8 Code review of the frontier scripts

Two read-only reviews of the new scripts, after J1 and L0 were already scored. Neither found a contract leak in the uploaded files. Both explain why the train 0.96 was never a promise.

**J1 path is legal.** [Onset analog legality](2ba95e49-9a1d-4b66-a15d-0cf70b37517a) checked `t2_joint_prior.py` and `build_frontier.py` against the official Task 2 scorer. Validation/private labels are not read. Inference features are `mspeed`/`mflow` on `[T−60, T)`. Slot T and the horizon are not in the query. Isolation still holds: 120 queue rows, 0 state, 0 ODME. Do not rebuild J1 from this list.

**The 0.96 / 0.785 leave-month numbers are optimistic proxies.** Same review, plus [frontier experiment code](d8493183-aa67-43d9-96c0-555a615da8ba):

- `evaluate()` scored query features on the unmasked train `speed`/`flow`. The zip uses the masked layer. Public onset 0.750 is the deploy number.
- Leave-month evaluation drops the query month; the zip uses the full train library. That is allowed. It is also why LOO is not a leaderboard forecast.
- `noisy` / `latent` are not `queue_true`. `latent` is a centered 3-slot median, so it sees t+1 inside the horizon. Rank methods with those columns. Do not quote them as official IoU.
- Onset `prior_top3` is an h=6-only submission scored with full-window IoU, the same shape as U1/J1. It is the right comparison for those files and the wrong comparison for a six-step persistence mask.
- `t2_front_analog.py` picks a front by neighbour *oracle* IoU. That is a ceiling, not a deployable policy. The branch was already closed.
- `twin_v2.py` `expected` is consensus among ensemble members, not E[IoU | data]. Leave-date already picked ffill in all 40 windows.
- Structural KF uses same-day RTS, as the official baseline does. It still lost to interpolation.

**Fixes applied after the reviews, not shipped.** Empty-known analogues are dropped from the Bayes mean. The train event filter now also requires no queue at the origin slot. `build_rows` asserts the midnight bound and falls back to an empty onset if the candidate pool is empty. `evaluate()` now reads query features from the masked layer. `t2_oracles.py` documents the proxy labels and the h=6 submission shape. None of that changes `sub_J1.zip`. Do not spend the leftover daily slot on a rebuild that exists only to absorb these notes.
