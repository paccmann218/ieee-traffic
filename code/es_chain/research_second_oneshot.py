"""One ES1 file: frozen TF3 with ongoing queues from the EQ1 all800 model.

The Q46 control must reproduce TF3's ongoing labels exactly before any cell
is replaced. Onset, state, flow, speed and demand text stay unchanged.
No evaluation mainline file is read. One submit, no retry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import cache
import score_local as sl
from research_combine_candidates import archive_csv
from research_jobs import snapshot_sources
from research_leaderboard_once import COMPETITION, listing
from research_queue_analog import build_library, encode_history, explicit_history
from research_queue_candidate import digest, raw_train
from research_queue_dynamics import parameters
from research_reopened_phase_experts import quarter_action
from research_second_queue import Q46_SHA
from research_sprint_queue_build import check_template
from research_transport_queue import context, transport_features


EQ1_SHA = "cb626813e621064594f6956344df81d7d8879e65ebc23ae2f164a2f8f6f9ac9f"
TF3_SHA = "2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a"
MARKER = Path("work/research_20261004_second/one_submission_attempted.json")


def now():
    return datetime.now(timezone.utc)


def ongoing_forecasts(release, control_path, model_path, threads):
    control_model = lgb.Booster(model_file=str(control_path))
    model = lgb.Booster(model_file=str(model_path))
    controls, candidates = [], []
    for panel in cache.PANELS[:8]:
        z, _ = raw_train(release, panel)
        p = parameters(panel, z["links"])
        ctx = context(p)
        vf, cap = p.free_speed_kmh.to_numpy(), p.capacity_vph.to_numpy()
        library = build_library(z, p, "queue_ongoing", stride=2)
        for split in ("validation", "private"):
            root = release / "task2" / panel / split
            windows = pd.read_csv(root / "window_index.csv")
            history = pd.read_parquet(root / "window_history.parquet")
            for window in windows[windows.condition == "queue_ongoing"].itertuples():
                isolated = history[history.window_id.astype(str) == str(window.window_id)]
                arrays = explicit_history(isolated, window, z["links"])
                for left, right in zip(arrays, explicit_history(history, window, z["links"])):
                    np.testing.assert_array_equal(left, right)
                origin = pd.Timestamp(window.forecast_origin)
                if origin.tzinfo is not None:
                    origin = origin.tz_convert("UTC").tz_localize(None)
                slot, weekend = origin.hour * 12 + origin.minute // 5, origin.dayofweek >= 5
                assert 12 <= slot <= 281
                frame = transport_features(arrays, p, ctx, slot, weekend)
                qr, qf = encode_history(arrays[0][None], arrays[1][None], vf, cap)
                times = [(origin + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(1, 7)]
                keys = dict(window_id=str(window.window_id), timestamp=np.repeat(times, len(p)), link_id=np.tile(z["links"], 6))
                for booster, bucket in ((control_model, controls), (model, candidates)):
                    probability = booster.predict(frame[booster.feature_name()], num_threads=threads).reshape(6, len(p))
                    assert np.isfinite(probability).all()
                    prediction = quarter_action(probability, library, qr, qf, slot, weekend)
                    bucket.append(pd.DataFrame(dict(**keys, queue_pred=prediction.ravel().astype(np.int8))))
            print(panel, split, "both models done", flush=True)
        del z, library
    control = pd.concat(controls, ignore_index=True)
    candidate = pd.concat(candidates, ignore_index=True)
    check_template(release, candidate, "queue_ongoing")
    return control, candidate


def build(args):
    release = args.release_root.resolve()
    cache.REL = sl.REL = release
    assert digest(args.base) == TF3_SHA
    assert digest(args.control_model) == Q46_SHA
    assert digest(args.model) == EQ1_SHA
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(Path(__file__), args.out)
    control, candidate = ongoing_forecasts(release, args.control_model, args.model, args.threads)
    control.to_csv(args.out / "control_ongoing.csv", index=False)
    candidate.to_csv(args.out / "candidate_ongoing.csv", index=False)
    compose(args)


def compose(args):
    release = args.release_root.resolve()
    assert digest(args.base) == TF3_SHA
    control = pd.read_csv(args.out / "control_ongoing.csv", dtype=str)
    candidate = pd.read_csv(args.out / "candidate_ongoing.csv", dtype=str)
    for frame in (control, candidate):
        frame["queue_pred"] = frame.queue_pred.astype(int)
    assert len(control) == len(candidate) == 87000
    key = pd.read_csv(release / "submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "timestamp", "link_id", "window_id"])
    with zipfile.ZipFile(args.base) as archive:
        assert archive.namelist() == ["submission_TF3.csv"]
        base = pd.read_csv(archive.open("submission_TF3.csv"), dtype=str)
    assert list(base.columns) == ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
    assert len(base) == len(key) == 6980503
    assert (base.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    assert (base.task.to_numpy() == key.task.to_numpy()).all()
    stamp = pd.to_datetime(key.timestamp, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ongoing = key.task.eq("queue") & key.window_id.isin(set(candidate.window_id))
    assert int(ongoing.sum()) == 87000
    wanted = pd.DataFrame(dict(
        window_id=key.loc[ongoing, "window_id"].to_numpy(),
        timestamp=stamp[ongoing].to_numpy(),
        link_id=key.loc[ongoing, "link_id"].to_numpy(),
        queue_pred=base.loc[ongoing, "queue_pred"].map(lambda value: int(float(value))).to_numpy()))
    paired = wanted.merge(control, on=["window_id", "timestamp", "link_id"],
                          suffixes=("_tf3", "_control"), validate="one_to_one")
    same = paired.queue_pred_tf3.to_numpy() == paired.queue_pred_control.to_numpy()
    if not same.all():
        raise SystemExit(f"TF3 ongoing control mismatch on {int((~same).sum())} rows; no file written")
    updated = wanted.drop(columns=["queue_pred"]).merge(candidate, on=["window_id", "timestamp", "link_id"], validate="one_to_one")
    new_text = np.where(updated.queue_pred.to_numpy() == 1, "1.0", "0.0")
    changed = int((new_text != base.loc[ongoing, "queue_pred"].to_numpy()).sum())
    if changed == 0:
        raise SystemExit("Candidate matches TF3 on every ongoing cell; no upload")
    kept = base.drop(columns=["queue_pred"]).copy()
    base.loc[ongoing, "queue_pred"] = new_text
    assert kept.equals(base.drop(columns=["queue_pred"]))
    csv_path = args.out / "submission_ES1.csv"
    base.to_csv(csv_path, index=False, lineterminator="\n")
    archive = archive_csv(csv_path)
    report = dict(candidate="ES1", base_sha256=TF3_SHA, model_sha256=EQ1_SHA, control_sha256=Q46_SHA,
                  rows=len(base), ongoing_rows=87000, ongoing_cells_changed=changed,
                  candidate_sha256=digest(archive), control_exact=True,
                  preserved="speed, flow, onset queues, demand, and unchanged ongoing text",
                  history="T-60 through T-5", uploaded=False)
    (args.out / "build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def submit(args):
    path = args.candidate.resolve()
    if digest(path) != args.expected_sha:
        raise SystemExit("Candidate hash changed")
    if MARKER.exists():
        raise SystemExit("ES1 upload already attempted; will not retry")
    record_dir = args.out
    record_dir.mkdir(parents=True, exist_ok=False)
    kaggle = Path(sys.executable).with_name("kaggle")
    payload, submissions = listing(kaggle)
    (record_dir / "submissions_before.csv").write_text(payload)
    today = now().date().isoformat()
    used = sum(row["date"].startswith(today) for row in submissions)
    if used >= 5 or any(row["fileName"] == path.name for row in submissions):
        raise SystemExit(f"No upload: daily used {used} or filename already listed")
    report = dict(status="upload_attempt_started", at=now().isoformat(), path=str(path),
                  candidate_sha256=args.expected_sha, daily_used_before=used)
    with MARKER.open("x") as stream:
        json.dump(report, stream, indent=2)
    message = "ES1: frozen TF3 with one ongoing-queue model trained on all nine TRAIN months. SHA256 " + args.expected_sha
    try:
        result = subprocess.run([str(kaggle), "competitions", "submit", COMPETITION, "-f", str(path), "-m", message],
                                capture_output=True, text=True, timeout=600)
        (record_dir / "upload_stdout.txt").write_text(result.stdout)
        (record_dir / "upload_stderr.txt").write_text(result.stderr)
        report["upload_returncode"] = result.returncode
    except Exception as exc:
        report["upload_error"] = str(exc)
    (record_dir / "submission_record.json").write_text(json.dumps(report, indent=2))
    for attempt in range(80):
        try:
            payload, submissions = listing(kaggle)
            (record_dir / "submissions_latest.csv").write_text(payload)
            found = [row for row in submissions if row["fileName"] == path.name]
            if len(found) > 1:
                raise RuntimeError("Duplicate filenames; no retry")
            if found:
                row = found[0]
                report.update(status="listed", submission=row, checked=now().isoformat())
                if "COMPLETE" in row["status"].upper() and row.get("publicScore"):
                    report["status"] = "complete"
                    report["verified_public_score"] = float(row["publicScore"])
                elif "ERROR" in row["status"].upper():
                    report["status"] = "evaluation_error"
                (record_dir / "submission_record.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
                if report["status"] in ("complete", "evaluation_error"):
                    return
            elif attempt >= 8:
                report["status"] = "uncertain_no_retry"
                (record_dir / "submission_record.json").write_text(json.dumps(report, indent=2))
                raise SystemExit("Upload outcome uncertain; do not retry")
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            report["last_poll_error"] = str(exc)
            (record_dir / "submission_record.json").write_text(json.dumps(report, indent=2))
        time.sleep(30)
    report["status"] = "wait_timeout_no_retry"
    (record_dir / "submission_record.json").write_text(json.dumps(report, indent=2))
    raise SystemExit("Score not ready; check read-only, do not resubmit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "compose", "submit"], required=True)
    ap.add_argument("--release-root", type=Path)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--control-model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--expected-sha")
    args = ap.parse_args()
    if args.stage == "build":
        build(args)
    elif args.stage == "compose":
        compose(args)
    else:
        submit(args)
