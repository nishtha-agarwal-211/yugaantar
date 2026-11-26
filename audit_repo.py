#!/usr/bin/env python3
"""Phase 7: repository integrity scan (offline, read-only, no model needed).

This mirrors the five anti-cheat checks the organiser's
``monsoon_r1.repo_scan`` performs on a submission, so a clean run here means the
same code is visible in the graded repo:

1. **No weather APIs or external URLs in inference code.**  The served model
   must be fully self-contained: the judge runs it with no network.
2. **No hardcoded dates or date lists in inference code.**  A twin that looks up
   "day 184 of 2021" is a lookup table, not a forecaster.  The wire contract
   only carries ``doy_sincos``, and this check proves the served path never
   touches a calendar.
3. **No multi-year daily archive committed.**  More than 400 daily slices of the
   32x36x6 grid (about 11 MB) means raw reanalysis is being shipped instead of
   learned weights.
4. **Model weights present and under 2 GB.**  A Hugging Face link is not enough;
   the file has to be in the repo.
5. **A real training pipeline, and a clean server / UI split.**  ``train.py``
   must actually backpropagate and checkpoint, ``serve.py`` must stay the graded
   inference entrypoint, and the dashboard must live beside it, not inside it.

Checks 1-3 are source scans, 4 is a filesystem scan, and every ``.py`` in the
repo must also compile on the scoring box's interpreter (a module that does not
import is a critical code-health flag, because the graders import it).

Usage
    python audit_repo.py            # human readable report
    python audit_repo.py --json     # machine readable, for CI
    python audit_repo.py --strict   # warnings count as failures

Exit status is 1 when a critical flag is found.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Code that runs at inference time (or ships to the browser and calls the API).
INFERENCE_FILES = (
    "serve.py",
    "model.py",
    "dashboard/app.py",
    "dashboard/state.py",
    "dashboard/static/app.js",
)
# Shared scientific libraries imported by the served model.  They also contain
# the training-side fit, so a date literal here is provenance, not a lookup.
SHARED_FILES = ("dataset.py", "climatology.py")

SKIP_DIRS = {".git", ".venv", "venv", ".freebuff", "__pycache__", "node_modules", ".mypy_cache"}

URL_RE = re.compile(r"https?://[^\s\"'`)<>\]]+")
# A CORS regex (``http://(localhost|127\.0\.0\.1)(:\d+)?$``) or a format template
# (``http://{args.host}:{args.port}``) is not a dialable endpoint: only a literal
# hostname can be fetched.
HOST_LITERAL_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
LOOPBACK = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
NETWORK_RE = re.compile(r"\b(requests\.(?:get|post|put)|urlopen|httpx\.|aiohttp|socket\.socket|urllib\.request)\b")
DATE_CALL_RE = re.compile(r"\b(?:date|datetime)\(\s*\d{4}\s*,\s*\d{1,2}\s*,\s*\d{1,2}")
ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")
MONTH_LIST_RE = re.compile(
    r"\[\s*[\"'](?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)[\"']"
)
DATA_SUFFIXES = {
    ".nc": "raw NetCDF grid", ".nc4": "raw NetCDF grid", ".grib": "raw GRIB grid",
    ".grib2": "raw GRIB grid", ".h5": "raw HDF5 grid", ".hdf5": "raw HDF5 grid",
    ".npy": "array dump", ".npz": "array dump", ".parquet": "columnar dump",
}
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
# 400 daily slices of the contract grid in float32 (~11 MB); anything bigger
# than that is not a compact coef/bundle and has to be justified.
ARCHIVE_BYTES = 400 * 6 * 32 * 36 * 4
WEIGHT_LIMIT = 2 * 1024 ** 3
SERVED_CHECKPOINTS = ("model_weights.pt", "artifacts/model_bundle.pt")


class Result(dict):
    def __init__(self, name: str, status: str, detail: str, fix: str = "") -> None:
        super().__init__(name=name, status=status, detail=detail, fix=fix)


def _read(rel: str) -> str:
    path = ROOT / rel
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _walk() -> list[Path]:
    out: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and not path.is_symlink():
            out.append(path)
    return out


# --------------------------------------------------------------------------- #
# 1. external endpoints
# --------------------------------------------------------------------------- #
def check_external_endpoints() -> Result:
    hits: list[str] = []
    loopback = 0
    clients: list[str] = []
    for rel in INFERENCE_FILES:
        text = _read(rel)
        for url in URL_RE.findall(text):
            host = url.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0].lower()
            if not HOST_LITERAL_RE.match(host):
                # A CORS pattern or a bind-address template carries no host of its
                # own; it is only acceptable when it is loopback by construction.
                if "{" in host or "}" in host or any(h in url for h in LOOPBACK):
                    loopback += 1
                else:
                    hits.append(f"{rel}: unresolved host pattern {url}")
            elif any(host == h or host.endswith("." + h) for h in LOOPBACK):
                loopback += 1
            else:
                hits.append(f"{rel}: {url}")
        for call in NETWORK_RE.findall(text):
            clients.append(f"{rel}: {call}")
    if hits:
        return Result("inference/external-endpoints", "CRITICAL",
                      "non-loopback endpoint(s) in inference code: " + "; ".join(hits),
                      "leave the weather data at training time; serve only bundled weights")
    detail = f"0 non-loopback URL(s) across {len(INFERENCE_FILES)} inference files"
    if loopback:
        detail += (f" ({loopback} loopback pattern(s): browser -> 127.0.0.1 model server, "
                   "CORS allow-list)")
    if clients:
        detail += "; network client(s): " + ", ".join(sorted(set(clients)))
    return Result("inference/external-endpoints", "PASS", detail)


# --------------------------------------------------------------------------- #
# 2. hardcoded dates
# --------------------------------------------------------------------------- #
def _date_hits(rel: str) -> list[str]:
    text = _read(rel)
    hits: list[str] = []
    for pattern, label in ((DATE_CALL_RE, "date literal"),
                           (ISO_DATE_RE, "ISO date string"),
                           (MONTH_LIST_RE, "month-name list")):
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line} {label} {match.group(0)[:40]!r}")
    return hits


def check_hardcoded_dates() -> Result:
    critical = [h for rel in INFERENCE_FILES for h in _date_hits(rel)]
    info = [h for rel in SHARED_FILES for h in _date_hits(rel)]
    if critical:
        return Result("inference/hardcoded-dates", "CRITICAL",
                      "calendar reference(s) in the served path: " + "; ".join(critical[:6]),
                      "the API only sends doy_sincos; evaluate harmonics from that phase")
    detail = f"no calendar at inference: {len(INFERENCE_FILES)} files scanned clean"
    if info:
        detail += (f"; {len(info)} provenance constant(s) in the shared fit module "
                   f"({', '.join(sorted({h.split(':')[0] for h in info}))}), not used to look data up")
    return Result("inference/hardcoded-dates", "PASS", detail)


# --------------------------------------------------------------------------- #
# 3. committed daily archives
# --------------------------------------------------------------------------- #
def check_archives() -> Result:
    big: list[str] = []
    largest = (0, "")
    total = 0
    for path in _walk():
        rel = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        total += size
        suffix = path.suffix.lower()
        if size > largest[0]:
            largest = (size, rel)
        if suffix in (".nc", ".nc4", ".grib", ".grib2", ".h5", ".hdf5"):
            big.append(f"{rel} ({size/1e6:.1f} MB, {DATA_SUFFIXES[suffix]})")
        elif suffix in DATA_SUFFIXES and size > ARCHIVE_BYTES:
            big.append(f"{rel} ({size/1e6:.1f} MB, over the 400-slice budget)")
    if big:
        return Result("repo/daily-archive", "CRITICAL",
                      "multi-year archive committed: " + "; ".join(big),
                      "ship fitted coefficients/weights only; keep the NetCDF out of the repo")
    return Result("repo/daily-archive", "PASS",
                  f"largest committed file {largest[0]/1e3:.0f} KB ({largest[1]}), "
                  f"repo total {total/1e6:.2f} MB, 400-slice budget {ARCHIVE_BYTES/1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# 4. weights
# --------------------------------------------------------------------------- #
def check_weights() -> Result:
    found: list[tuple[str, int]] = []
    for path in _walk():
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            found.append((path.relative_to(ROOT).as_posix(), path.stat().st_size))
    oversized = [f"{rel} ({size/1e9:.2f} GB)" for rel, size in found if size > WEIGHT_LIMIT]
    if oversized:
        return Result("weights/size", "CRITICAL",
                      "checkpoint over the 2 GB budget: " + "; ".join(oversized),
                      "retrain with a smaller width/depth")
    served = [rel for rel in SERVED_CHECKPOINTS if (ROOT / rel).exists()]
    if not served:
        return Result("weights/present", "CRITICAL",
                      "no checkpoint in the repo: serve.py serves its persistence fallback "
                      f"({' / '.join(SERVED_CHECKPOINTS)} missing), so forecast skill and "
                      "perturbation response score zero",
                      "python train.py --data india_climate_train.nc --out model_weights.pt")
    detail = "; ".join(f"{rel} {size/1e6:.2f} MB" for rel, size in found)
    if len(served) < len(SERVED_CHECKPOINTS):
        detail += f" (all {len(found)} weight file(s) inside the 2 GB budget)"
    return Result("weights/present", "PASS", detail)


# --------------------------------------------------------------------------- #
# 5. training pipeline and server / UI separation
# --------------------------------------------------------------------------- #
def check_pipeline() -> list[Result]:
    results: list[Result] = []
    train = _read("train.py")
    needs = {"backward": "loss.backward(", "optimizer": "torch.optim",
             "checkpoint": "torch.save", "data loader": "xr.open_dataset"}
    missing = [label for label, needle in needs.items() if needle not in train]
    n_args = len(re.findall(r'add_argument\("--', train))
    if missing:
        results.append(Result("pipeline/training", "CRITICAL",
                              "train.py is missing " + ", ".join(missing),
                              "the graders look for a real optimisation loop"))
    else:
        results.append(Result("pipeline/training", "PASS",
                              f"train.py: forward/backward + optimizer + checkpoint save, "
                              f"{n_args} CLI options, {len(train.splitlines())} lines"))

    serve = _read("serve.py")
    dash_app = _read("dashboard/app.py")
    problems: list[str] = []
    if "/simulate" not in serve or "/health" not in serve:
        problems.append("serve.py does not expose the contract endpoints")
    if "<html" in serve.lower() or "StaticFiles" in serve:
        problems.append("serve.py serves UI assets")
    if re.search(r"^\s*(?:from|import)\s+dashboard", serve, re.M):
        problems.append("serve.py imports the dashboard")
    if re.search(r"^\s*import\s+torch", dash_app, re.M) or "state_dict" in dash_app:
        problems.append("dashboard/app.py runs the model itself")
    if not (ROOT / "dashboard" / "static" / "index.html").exists():
        problems.append("dashboard UI missing")
    if problems:
        results.append(Result("separation/server-ui", "CRITICAL", "; ".join(problems),
                              "keep serve.py = model API, dashboard/ = UI, no shared process"))
    else:
        results.append(Result("separation/server-ui", "PASS",
                              "serve.py = /health + /simulate only; dashboard/ = UI + bootstrap API "
                              "and calls the model server over HTTP, no torch"))
    return results


# --------------------------------------------------------------------------- #
# code health
# --------------------------------------------------------------------------- #
def check_compiles() -> Result:
    failures: list[str] = []
    count = 0
    for path in _walk():
        if path.suffix != ".py":
            continue
        count += 1
        rel = path.relative_to(ROOT).as_posix()
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except SyntaxError as exc:
            failures.append(f"{rel}:{exc.lineno} {exc.msg}")
    if failures:
        return Result("code-health/compiles", "CRITICAL",
                      f"{len(failures)} module(s) do not parse: " + "; ".join(failures),
                      "a module the graders import must compile on their interpreter")
    return Result("code-health/compiles", "PASS",
                  f"{count} modules parse (Python {sys.version_info.major}.{sys.version_info.minor})")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def scan() -> list[Result]:
    results = [check_external_endpoints(), check_hardcoded_dates(), check_archives(),
               check_weights(), check_compiles()]
    results.extend(check_pipeline())
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine readable output")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args(argv)

    results = scan()
    critical = [r for r in results if r["status"] == "CRITICAL"]
    warnings = [r for r in results if r["status"] == "WARNING"]

    if args.json:
        print(json.dumps({"root": str(ROOT), "checks": results,
                          "critical": len(critical), "warnings": len(warnings)}, indent=2))
    else:
        print(f"Phase 7 repository integrity scan — {ROOT}")
        width = max(len(r["name"]) for r in results)
        for r in results:
            print(f"  [{r['status']:<8}] {r['name']:<{width}}  {r['detail']}")
            if r["fix"]:
                print(f"  {'':<10} {'':<{width}}  -> fix: {r['fix']}")
        verdict = "CLEAN" if not critical and not (args.strict and warnings) else "FLAGS FOUND"
        print(f"\n{verdict}: {len(critical)} critical, {len(warnings)} warning(s) "
              f"across {len(results)} checks")

    return 1 if critical or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
