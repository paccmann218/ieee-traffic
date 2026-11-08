"""Regenerate the selected ES4 submission from the competition release.

Reads:  the competition release (--release-root), the frozen TF3 and ES4
        archives in submissions/, the TF3 bundle in vendor/, and the original
        stage scripts in code/es_chain/.
Writes: only under --out, which must not exist yet. Nothing is uploaded.

ES4 was built as a chain of single-change files:

    TF3 -> ES1 (EQ1 ongoing queues) -> ES2 (gap 1) -> ES3 (gap 2) -> ES4 (gaps 3 to 6)

The scripts in code/es_chain/ are the ones that built those files, copied
without edits. Each checks the SHA256 of its parent zip. A rebuilt zip can
differ in compressed bytes under another zlib, so this driver first records
whether the parent matches, then passes the actual hash to the stage. The
decisive check is the last one: every row of the regenerated ES4 is compared
with the frozen submissions/submission_ES4.zip.

The EQ1 classifier weights from the competition were not preserved. Three routes
produce the ongoing queues of ES1, and the report states which one was used:

    default                 retrain EQ1 from the release with the original code
    --eq1-model PATH        use a supplied EQ1 model file
    --ongoing-from-frozen   copy the ongoing labels from the frozen ES4 archive.
                            This regenerates the speed edits only. The queue
                            labels are then not an independent regeneration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    import resource
except ImportError:  # not available on Windows
    resource = None


ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "code" / "es_chain"
BUNDLE = ROOT / "vendor" / "reproduction_TF3.zip"
TF3_ZIP = ROOT / "submissions" / "submission_TF3.zip"
ES4_ZIP = ROOT / "submissions" / "submission_ES4.zip"
INVENTORY = CHAIN / "q46_training_inventory.csv"
PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "lightgbm", "scikit-learn")
CACHE_FILES = 30

# Recorded when each file was built and submitted.
EXPECTED = {
    "TF3": {"zip": "2ae94d422399c7248297487fc8f45d737de0e024fbad32b6a360c290bbabca7a",
            "csv": "487b68cf1879984e16f7155b17b8b88da6fcac24ec0a860c0f02284d85430780"},
    "ES1": {"zip": "f5fac2c81a5061b1fcb589bc24c59dd37b8bc3d4302baaa15c3564f3d22858c8",
            "csv": "69cbc54b22b5c38f171bee0ecfd2a7fa71237fcde3bcc198d9405ed4958188f4"},
    "ES2": {"zip": "98e9ee0bbdb9d42a6a4bed98e99cec5f7e2ffb5cbd871fe89a10f6364b304e4a"},
    "ES3": {"zip": "712f772011ede5d63e1ad11d739668ee566074e5687e72224d5c7e6c55c96f71"},
    "ES4": {"zip": "ce03386b42e33bc6c726b09c0b64b791f294ea3abeb88d7e310552bb080c04bb",
            "csv": "05d288a825236423ea66b04d286a5ce81a27deb519eec964ff0bef2e3c613736"},
}
EQ1_SHA = "cb626813e621064594f6956344df81d7d8879e65ebc23ae2f164a2f8f6f9ac9f"
EQ1_ROWS = 3_172_453
EQ1_WINDOWS = 3_766
INVENTORY_SHA = "6f3f45a7a06c37248f1d108afcf2881321050fba1c11f81ae5772b98595ddc53"

# Runs in a child process: applies the parent-hash constants, then calls the
# original stage entry point.
LAUNCHER = r"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path("tfb").resolve()))
spec = json.loads(sys.argv[1])
module = __import__(spec["module"])
for name, value in spec.get("constants", {}).items():
    if not hasattr(module, name):
        raise SystemExit(f"{spec['module']} has no constant {name}")
    setattr(module, name, value)
if "namespace" in spec:
    values = {key: Path(value) if key in spec["paths"] else value for key, value in spec["namespace"].items()}
    getattr(module, spec["call"])(argparse.Namespace(**values))
else:
    sys.argv = [spec["module"] + ".py"] + spec["argv"]
    getattr(module, spec["call"])()
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_csv(path: Path) -> Path:
    """Same container settings as research_combine_candidates.archive_csv."""
    target = path.with_suffix(".zip")
    info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 2, 0, 0, 0))
    info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        with archive.open(info, "w", force_zip64=True) as out, path.open("rb") as source:
            shutil.copyfileobj(source, out, length=1 << 20)
    return target


def peak_child_rss_gib() -> float | None:
    if resource is None:
        return None
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2)


def prepare_runtime(out: Path) -> Path:
    """Extract the frozen bundle and add the original ES stage scripts beside its modules."""
    runtime = out / "runtime"
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        archive.extractall(runtime)
    for script in sorted(CHAIN.glob("research_*.py")):
        shutil.copy2(script, runtime / "tfb" / script.name)
    return runtime


class Chain:
    def __init__(self, args):
        self.release = args.release_root.resolve()
        self.out = args.out.resolve()
        self.threads = args.threads
        self.runtime = prepare_runtime(self.out)
        self.cache = args.cache_root.resolve() if args.cache_root else self.out / "cache"
        self.env = dict(os.environ, TFB_REL=str(self.release), TFB_CACHE=str(self.cache),
                        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        self.report = dict(
            candidate="ES4", release_root=str(self.release), threads=self.threads, uploaded=False,
            environment=dict(python=platform.python_version(), platform=platform.platform(),
                             packages={name: importlib.metadata.version(name) for name in PACKAGES}),
            stages={}, files={})
        self.save()

    def save(self):
        (self.out / "es4_reproduction.json").write_text(json.dumps(self.report, indent=2))

    def timed(self, name, action):
        started = time.time()
        result = action()
        self.report["stages"][name] = dict(minutes=round((time.time() - started) / 60, 2),
                                           peak_child_rss_gib_so_far=peak_child_rss_gib())
        self.save()
        print(f"[{name}] done in {self.report['stages'][name]['minutes']} min", flush=True)
        return result

    def launch(self, spec):
        subprocess.run([sys.executable, "-c", LAUNCHER, json.dumps(spec)], cwd=self.runtime, env=self.env, check=True)

    def caches(self):
        found = sorted(self.cache.glob("*.npz")) if self.cache.is_dir() else []
        if len(found) == CACHE_FILES:
            self.report["cache"] = dict(path=str(self.cache), rebuilt=False)
            return
        if found:
            raise SystemExit(f"{self.cache} holds {len(found)} of {CACHE_FILES} cache files; give a complete or new directory")
        subprocess.run([sys.executable, "tfb/research_rebuild_cache.py", "--release-root", str(self.release),
                        "--out", str(self.cache)], cwd=self.runtime, env=self.env, check=True)
        self.report["cache"] = dict(path=str(self.cache), rebuilt=True)

    def train_eq1(self) -> Path:
        if sha256(INVENTORY) != INVENTORY_SHA:
            raise SystemExit("The Q46 training inventory does not match its recorded SHA256")
        prepared, fitted = self.out / "eq1_prepare", self.out / "eq1_fit"
        self.timed("eq1_prepare", lambda: self.launch(dict(
            module="research_second_queue", call="main",
            argv=["--stage", "prepare", "--out", str(prepared), "--inventory", str(INVENTORY),
                  "--threads", str(self.threads)])))
        self.timed("eq1_fit", lambda: self.launch(dict(
            module="research_second_queue", call="main",
            argv=["--stage", "fit", "--out", str(fitted), "--prepared", str(prepared),
                  "--threads", str(self.threads)])))
        record = json.loads((fitted / "fit.json").read_text())
        self.report["eq1_retrain"] = dict(
            rows=record["rows"], recorded_rows=EQ1_ROWS, windows=record["windows"], recorded_windows=EQ1_WINDOWS)
        return fitted / "transport800.txt"

    def es1_with_model(self, model: Path) -> Path:
        target = self.out / "es1"
        control = self.runtime / "models" / "final_queue" / "transport.txt"
        self.timed("es1", lambda: self.launch(dict(
            module="research_second_oneshot", call="build", constants=dict(EQ1_SHA=sha256(model)),
            namespace=dict(release_root=str(self.release), base=str(TF3_ZIP), model=str(model),
                           control_model=str(control), out=str(target), threads=self.threads,
                           candidate=None, expected_sha=None),
            paths=["release_root", "base", "model", "control_model", "out"])))
        return target / "submission_ES1.zip"

    def es1_from_frozen(self) -> Path:
        """TF3 with the queue column of frozen ES4. Byte-identical to the ES1 CSV."""
        target = self.out / "es1"
        target.mkdir()
        csv_path = target / "submission_ES1.csv"
        with zipfile.ZipFile(TF3_ZIP) as tf3, zipfile.ZipFile(ES4_ZIP) as es4:
            with tf3.open(tf3.namelist()[0]) as left, es4.open(es4.namelist()[0]) as right, csv_path.open("wb") as out:
                for line, other in zip(left, right):
                    fields = line.rstrip(b"\n").split(b",")
                    if fields[1] == b"queue":
                        fields[4] = other.rstrip(b"\n").split(b",")[4]
                    out.write(b",".join(fields) + b"\n")
        return archive_csv(csv_path)

    def check_parent(self, name: str, path: Path):
        entry = dict(zip_sha256=sha256(path), zip_matches_recorded=sha256(path) == EXPECTED[name]["zip"])
        if "csv" in EXPECTED[name]:
            entry["csv_sha256"] = member_sha256(path)
            entry["csv_matches_recorded"] = entry["csv_sha256"] == EXPECTED[name]["csv"]
        self.report["files"][name] = entry
        self.save()

    def recenter(self, stage: str, module: str, constant: str, parent: Path) -> Path:
        target = self.out / stage.lower()
        self.timed(stage.lower(), lambda: self.launch(dict(
            module=module, call="main", constants={constant: sha256(parent)},
            argv=["--out", str(target), "--base", str(parent), "--release-root", str(self.release)])))
        output = target / f"submission_{stage}.zip"
        if not output.is_file():
            raise SystemExit(f"{stage}: the stage's own gate wrote no file. See {target}")
        self.check_parent(stage, output)
        return output


def compare_with_frozen(candidate: Path, frozen: Path) -> dict:
    """Stream every row of the regenerated zip against the frozen zip."""
    with zipfile.ZipFile(candidate) as new_zip, zipfile.ZipFile(frozen) as old_zip:
        with new_zip.open(new_zip.namelist()[0]) as new, old_zip.open(old_zip.namelist()[0]) as old:
            header = new.readline()
            if header != old.readline():
                raise SystemExit("The regenerated ES4 has a different header")
            names = header.decode().strip().split(",")
            different = {name: 0 for name in names}
            largest = {name: 0.0 for name in names[2:]}
            rows = 0
            while True:
                left, right = new.readline(), old.readline()
                if not left or not right:
                    if left or right:
                        raise SystemExit("The regenerated ES4 has a different row count")
                    break
                rows += 1
                if left == right:
                    continue
                a = left.rstrip(b"\n").split(b",")
                b = right.rstrip(b"\n").split(b",")
                for index, name in enumerate(names):
                    if a[index] == b[index]:
                        continue
                    different[name] += 1
                    if index >= 2:
                        gap = abs(float(a[index] or "nan") - float(b[index] or "nan"))
                        largest[name] = max(largest[name], gap) if gap == gap else float("inf")
    identifiers = different["submission_id"] == 0 and different["task"] == 0
    passed = identifiers and largest["queue_pred"] == 0 and all(
        largest[name] <= 1e-8 for name in ("speed_kmh", "flow_vph", "path_flow"))
    return dict(rows=rows, cells_with_different_text=different, max_absolute_difference=largest,
                csv_byte_identical_to_frozen=member_sha256(candidate) == member_sha256(frozen),
                tolerance="queue exact; speed, flow and path flow within 1e-8", passed=bool(passed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate ES4 from the competition release.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cache-root", type=Path, help="Complete dense-cache directory to reuse")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--eq1-model", type=Path, help="Use this EQ1 model file instead of retraining")
    route.add_argument("--ongoing-from-frozen", action="store_true",
                       help="Copy ongoing labels from the frozen ES4 and regenerate speeds only")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    chain = Chain(args)
    chain.timed("caches", chain.caches)
    if args.ongoing_from_frozen:
        chain.report["ongoing_queue_route"] = "copied from the frozen ES4 archive; not an independent regeneration"
        es1 = chain.timed("es1", chain.es1_from_frozen)
    else:
        if args.eq1_model:
            model = args.eq1_model.resolve()
            chain.report["ongoing_queue_route"] = "supplied EQ1 model"
        else:
            model = chain.train_eq1()
            chain.report["ongoing_queue_route"] = "EQ1 retrained from the release"
        chain.report["eq1_model"] = dict(sha256=sha256(model), matches_recorded=sha256(model) == EQ1_SHA)
        es1 = chain.es1_with_model(model)
    chain.check_parent("ES1", es1)
    es2 = chain.recenter("ES2", "research_recenter_state", "ES1_SHA", es1)
    es3 = chain.recenter("ES3", "research_gap2_speed", "ES2_SHA", es2)
    es4 = chain.recenter("ES4", "research_gap_ladder", "ES3_SHA", es3)
    chain.report["comparison_with_frozen_ES4"] = chain.timed("comparison", lambda: compare_with_frozen(es4, ES4_ZIP))
    chain.save()
    passed = chain.report["comparison_with_frozen_ES4"]["passed"]
    print(json.dumps(chain.report["comparison_with_frozen_ES4"], indent=2), flush=True)
    print("ES4 regeneration", "PASSED" if passed else "FAILED", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
