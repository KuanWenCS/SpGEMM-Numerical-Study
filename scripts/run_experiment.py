import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def input_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Record a reproducible study run; unrecognized options are passed to the study executable.")
    parser.add_argument("--binary", default="build/study")
    parser.add_argument("--a", required=True)
    parser.add_argument("--p", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int)
    options, forwarded = parser.parse_known_args()
    out = Path(options.out).resolve()
    if out.exists():
        parser.error("Output directory already exists; choose a new path.")
    binary = Path(options.binary).resolve()
    a, p = Path(options.a).resolve(), Path(options.p).resolve()
    for path in (binary, a, p):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    out.mkdir(parents=True)
    environment = os.environ.copy()
    if options.gpu is not None:
        if options.gpu < 0:
            parser.error("GPU index must be nonnegative.")
        environment["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
    command = [str(binary), "--a", str(a), "--p", str(p), "--out", str(out / "artifacts"), *forwarded]
    manifest = {"started_utc": now(), "status": "preparing", "command": command,
                "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                "python": platform.python_version(),
                "git_commit": capture(["git", "rev-parse", "HEAD"]),
                "git_status": capture(["git", "status", "--porcelain"]),
                "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")}
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        start = time.perf_counter()
        manifest["inputs"] = {"A": input_info(a), "P": input_info(p)}
        sources = [ROOT / "Makefile"]
        for directory in ("include", "src", "scripts"):
            sources.extend(path for path in (ROOT / directory).rglob("*")
                           if path.suffix in {".cpp", ".cu", ".hpp", ".cuh", ".py"})
        manifest["source_sha256"] = {str(path.relative_to(ROOT)): input_info(path)["sha256"]
                                     for path in sorted(sources)}
        manifest["hash_seconds"] = time.perf_counter() - start
        if shutil.which("nvidia-smi"):
            manifest["gpu_snapshot"] = capture(["nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"])
        manifest["status"] = "running"
        save()
        print(f"Running; progress log: {out / 'stderr.log'}", flush=True)
        start = time.perf_counter()
        with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        manifest["process_wall_seconds"] = time.perf_counter() - start
        manifest["returncode"] = completed.returncode
        manifest["status"] = "complete" if completed.returncode == 0 else "failed"
        manifest["finished_utc"] = now()
        save()
        print(f"{manifest['status']}: {out}; process wall={manifest['process_wall_seconds']:.3f}s")
        return completed.returncode
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = now()
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
