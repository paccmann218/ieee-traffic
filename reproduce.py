"""Single entry point for the two selected TrafficFlowBench submissions.

    python reproduce.py
    python reproduce.py --submission tf3 --release-root PATH --out PATH
    python reproduce.py --submission es4 --release-root PATH --out PATH
    python reproduce.py --submission all --release-root PATH --out PATH

See REPRODUCIBILITY.md for hardware, runtime and what has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SELECTED = {
    "tf3": {
        "path": ROOT / "submissions" / "submission_TF3.zip",
        "sha256": "2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a",
        "kaggle_ref": "56825669",
        "public_score": "0.86991",
        "submitted_utc": "2026-10-04T13:51:11Z",
    },
    "es4": {
        "path": ROOT / "submissions" / "submission_ES4.zip",
        "sha256": "ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb",
        "kaggle_ref": "56998048",
        "public_score": "0.87187",
        "submitted_utc": "2026-10-09T07:13:22Z",
    },
}
BUNDLE = ROOT / "vendor" / "reproduction_TF3.zip"
BUNDLE_SHA = "dce93b3f2a38d27014497ec1a563cc1e2d032da11c499234f226c72093678cd7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check() -> int:
    failed = False
    for name, item in SELECTED.items():
        path = item["path"]
        if not path.is_file():
            print(f"MISSING {name}: {path}")
            failed = True
            continue
        found = sha256(path)
        ok = found == item["sha256"]
        print(f"{name.upper():<9} sha256 {found}  {'OK' if ok else 'MISMATCH'}")
        failed = failed or not ok
    if BUNDLE.is_file():
        found = sha256(BUNDLE)
        print(f"{'TF3-CODE':<9} sha256 {found}  {'OK' if found == BUNDLE_SHA else 'MISMATCH'}")
        failed = failed or found != BUNDLE_SHA
    else:
        print(f"MISSING tf3 code bundle: {BUNDLE}")
        failed = True
    return 1 if failed else 0


def environment_problems() -> list[str]:
    """Differences from the pinned environment in which the results were produced."""
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {sys.version.split()[0]}; expected 3.12")
    for line in (ROOT / "requirements.txt").read_text().split():
        name, wanted = line.split("==")
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed; expected {wanted}")
            continue
        if found != wanted:
            problems.append(f"{name} {found}; expected {wanted}")
    return problems


def run_tf3(release: Path, out: Path, threads: int) -> int:
    release, out = release.resolve(), out.resolve()
    if out.exists():
        raise SystemExit(f"Output directory already exists: {out}")
    if check():
        raise SystemExit("Selected archives failed the checksum check")
    work = ROOT / "vendor" / "reproduction_TF3_extracted"
    if not (work / "tfb" / "research_transfer_bagged_state_reproduce.py").is_file():
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        with zipfile.ZipFile(BUNDLE) as archive:
            archive.extractall(work)
    command = [
        sys.executable,
        "tfb/research_transfer_bagged_state_reproduce.py",
        "--release-root", str(release),
        "--models", "models",
        "--inference-spec", "inference.json",
        "--out", str(out),
        "--threads", str(threads),
        "--reference", str(SELECTED["tf3"]["path"]),
        "--expected-reference-sha", SELECTED["tf3"]["sha256"],
    ]
    return subprocess.call(command, cwd=work)


def run_es4(release: Path, out: Path, threads: int, cache_root: Path | None = None,
            eq1_model: Path | None = None, ongoing_from_frozen: bool = False) -> int:
    if check():
        raise SystemExit("Selected archives failed the checksum check")
    command = [
        sys.executable,
        str(ROOT / "code" / "regenerate_es4.py"),
        "--release-root", str(release.resolve()),
        "--out", str(out.resolve()),
        "--threads", str(threads),
    ]
    if cache_root is not None:
        command += ["--cache-root", str(cache_root.resolve())]
    if eq1_model is not None:
        command += ["--eq1-model", str(eq1_model.resolve())]
    if ongoing_from_frozen:
        command.append("--ongoing-from-frozen")
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce or verify the two selected submissions.")
    parser.add_argument("--submission", choices=("tf3", "es4", "all"))
    parser.add_argument("--release-root", type=Path, help="Directory with submission_key.csv, corridors/, task1/ and task2/")
    parser.add_argument("--out", type=Path, help="New output directory")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cache-root", type=Path, help="ES4 only: complete dense-cache directory to reuse")
    parser.add_argument("--eq1-model", type=Path, help="ES4 only: use this EQ1 model instead of retraining it")
    parser.add_argument("--ongoing-from-frozen", action="store_true",
                        help="ES4 only: copy ongoing labels from the frozen ES4 and regenerate speeds only")
    parser.add_argument("--allow-environment-mismatch", action="store_true",
                        help="Run even if Python or a package differs from the pinned environment")
    args = parser.parse_args()
    if args.submission is None:
        return check()
    if args.release_root is None or args.out is None:
        raise SystemExit("--release-root and --out are required to regenerate a submission")
    problems = environment_problems()
    if problems and not args.allow_environment_mismatch:
        raise SystemExit("Environment differs from requirements.txt:\n  " + "\n  ".join(problems))
    if args.submission == "tf3":
        return run_tf3(args.release_root, args.out, args.threads)
    if args.submission == "es4":
        return run_es4(args.release_root, args.out, args.threads, args.cache_root, args.eq1_model, args.ongoing_from_frozen)
    if args.out.exists():
        raise SystemExit(f"Output directory already exists: {args.out}")
    code = run_tf3(args.release_root, args.out / "tf3", args.threads)
    if code:
        return code
    return run_es4(args.release_root, args.out / "es4", args.threads, args.out / "tf3" / "cf2_base" / "cf1_base" / "cache",
                   args.eq1_model, args.ongoing_from_frozen)


if __name__ == "__main__":
    raise SystemExit(main())
