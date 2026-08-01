"""Batch intake: ZIP/folder extraction, manifest validation, bounded execution.

The brief requires that a single malformed file must not fail the batch, and
that the dashboard says which file failed and why. That guarantee is
implemented in two layers: :func:`preflight` catches structural problems before
any work starts, and :func:`run_batch` isolates every per-file failure.

Manifest parsing handles what real CSVs actually contain - a UTF-8 BOM, CRLF
line endings, quoted JSON containing commas, and an empty ``result_json`` for
an unlabeled hidden set.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

from autoace.config import Settings
from autoace.ingest import SUPPORTED_SUFFIXES
from autoace.predict import analyse_file

MANIFEST_NAMES = ("labels.csv", "manifest.csv")
MAX_ZIP_RATIO = 200          # guards against zip bombs
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3


class BatchError(Exception):
    """Fatal problem with the batch as a whole (not with one file)."""


def parse_manifest(data: bytes) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Parse ``name,result_json`` into truths plus a list of row-level problems.

    ``utf-8-sig`` strips a BOM if present; ``csv`` handles CRLF and quoted
    fields containing commas. An empty ``result_json`` is not an error - that is
    what an unlabeled hidden set looks like.
    """
    issues: list[str] = []
    truths: dict[str, dict[str, Any]] = {}
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        issues.append("manifest was not valid UTF-8; decoded as latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise BatchError("manifest is empty or has no header row")

    normalised = {(f or "").strip().lower(): f for f in reader.fieldnames}
    if "name" not in normalised:
        raise BatchError(
            f"manifest must have a 'name' column (found: {list(reader.fieldnames)})"
        )
    name_col = normalised["name"]
    json_col = normalised.get("result_json")

    seen: set[str] = set()
    for i, row in enumerate(reader, start=2):
        name = (row.get(name_col) or "").strip()
        if not name:
            issues.append(f"row {i}: empty name, skipped")
            continue
        if name in seen:
            issues.append(f"row {i}: duplicate name {name!r}, later row ignored")
            continue
        seen.add(name)
        raw = (row.get(json_col) or "").strip() if json_col else ""
        if not raw:
            truths[name] = {}          # present but unlabeled
            continue
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("not a JSON object")
            truths[name] = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            truths[name] = {}
            issues.append(f"row {i} ({name}): result_json unparseable ({exc}); "
                          f"treated as unlabeled")
    return truths, issues


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> list[Path]:
    """Extract a ZIP, refusing path traversal and absurd expansion ratios."""
    total = 0
    out: list[Path] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name          # flatten; ignore archive dirs
        if not name or name.startswith("."):
            continue
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise BatchError("archive expands beyond the size limit")
        if info.compress_size and info.file_size / max(info.compress_size, 1) > MAX_ZIP_RATIO:
            raise BatchError(f"archive entry {name!r} has a suspicious compression ratio")
        target = dest / name
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        out.append(target)
    return out


def stage_upload(files: list[tuple[str, bytes]], workdir: Path) -> list[Path]:
    """Write uploaded files into ``workdir``, expanding a ZIP if given one."""
    workdir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for filename, blob in files:
        base = Path(filename).name
        if base.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    staged.extend(_safe_extract(zf, workdir))
            except zipfile.BadZipFile as exc:
                raise BatchError(f"{base}: not a readable ZIP archive ({exc})") from exc
        else:
            target = workdir / base
            target.write_bytes(blob)
            staged.append(target)
    if not staged:
        raise BatchError("upload contained no files")
    return staged


def preflight(staged: list[Path]) -> dict[str, Any]:
    """Validate the batch before any processing starts.

    The evaluator sees this report and decides whether to run, which is the
    point: structural problems should surface in seconds, not after a paid
    batch has half-completed.
    """
    manifest_path = next(
        (p for p in staged if p.name.lower() in MANIFEST_NAMES),
        next((p for p in staged if p.suffix.lower() == ".csv"), None),
    )
    audio = [p for p in staged if p.suffix.lower() in SUPPORTED_SUFFIXES]
    other = [
        p for p in staged
        if p.suffix.lower() not in SUPPORTED_SUFFIXES and p != manifest_path
    ]

    truths: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    if manifest_path is not None:
        try:
            truths, issues = parse_manifest(manifest_path.read_bytes())
        except BatchError as exc:
            issues.append(str(exc))

    audio_names = {p.name for p in audio}
    manifest_names = set(truths)

    zero_byte = [p.name for p in audio if p.stat().st_size == 0]
    labelled = sum(1 for v in truths.values() if v)

    return {
        "manifest": manifest_path.name if manifest_path else None,
        "n_audio_files": len(audio),
        "n_manifest_rows": len(truths),
        "n_labelled_rows": labelled,
        "has_ground_truth": labelled > 0,
        "in_manifest_not_uploaded": sorted(manifest_names - audio_names),
        "uploaded_not_in_manifest": sorted(audio_names - manifest_names),
        "unsupported_files": sorted(p.name for p in other),
        "zero_byte_files": sorted(zero_byte),
        "manifest_issues": issues,
        "processable": sorted(audio_names),
        "blocking": (
            [] if audio else ["no supported audio files found in the upload"]
        ),
    }


async def run_batch(
    *,
    store: Any,
    batch_id: str,
    workdir: Path,
    truths: dict[str, dict[str, Any]],
    settings: Settings,
    on_progress: Callable[[], None] | None = None,
) -> None:
    """Process every audio file with bounded concurrency and per-file isolation."""
    store.mark_started(batch_id)
    audio = sorted(
        p for p in workdir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    sem = asyncio.Semaphore(max(1, settings.max_concurrency))

    async def one(path: Path) -> None:
        async with sem:
            try:
                # analyse_file never raises, but the executor boundary itself
                # can - so this is belt and braces around the whole hop.
                result = await asyncio.to_thread(analyse_file, path, settings=settings)
                payload = result.to_row()
                payload["evidence"] = result.evidence
                payload["warnings"] = result.warnings
                payload["repairs"] = result.repairs
                payload["tokens"] = result.tokens
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "name": path.name,
                    "status": "error",
                    "reason": f"worker failure: {type(exc).__name__}: {str(exc)[:200]}",
                }
            truth = truths.get(path.name) or None
            store.add_result(batch_id, path.name, payload, truth or None)
            if on_progress:
                on_progress()

    try:
        await asyncio.gather(*(one(p) for p in audio))
        store.mark_finished(batch_id, "complete")
    except Exception:  # noqa: BLE001
        store.mark_finished(batch_id, "failed")
        raise
    finally:
        # Privacy: audio is removed as soon as the batch finishes. The results
        # rows persist; the audio does not.
        shutil.rmtree(workdir, ignore_errors=True)
        store.mark_purged(batch_id)


def results_to_csv(rows: list[dict[str, Any]]) -> str:
    """Flatten results for download, preserving the original filename."""
    from autoace.schema import FIELD_ORDER

    cols = ["name", "status", "reason", *FIELD_ORDER,
            "duration_s", "latency_s", "cost_usd", "cost_per_audio_min", "model"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in cols})
    return buf.getvalue()
