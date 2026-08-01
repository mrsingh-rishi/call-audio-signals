"""FastAPI dashboard.

Single container: this app serves both the API and the static UI, so there is
one deploy, one URL and one thing to keep alive through the evaluation period.

Auth is a single shared credential in a signed cookie. The brief asks for login
credentials to hand over, not a user management system.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoace.config import (  # noqa: E402
    COST_CEILING_PER_AUDIO_MIN,
    REPO_ROOT,
    get_settings,
    redacted_summary,
)
from autoace.metrics import score_batch  # noqa: E402
from autoace.schema import FIELD_ORDER  # noqa: E402

from batch import BatchError, parse_manifest, preflight, results_to_csv, run_batch  # noqa: E402
from db import Store  # noqa: E402

SETTINGS = get_settings()
UPLOAD_ROOT = REPO_ROOT / "uploads"
WEB_DIR = REPO_ROOT / "web"
DB_PATH = REPO_ROOT / "autoace.db"
COOKIE = "autoace_session"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

store = Store(DB_PATH)
_secret = SETTINGS.session_secret or secrets.token_urlsafe(32)
_serializer = URLSafeTimedSerializer(_secret, salt="autoace-session")

app = FastAPI(title="AutoAce Voice Tone Analysis", docs_url=None, redoc_url=None)


# --- auth -----------------------------------------------------------------

def _issue(resp: Response) -> None:
    resp.set_cookie(
        COOKIE,
        _serializer.dumps({"u": SETTINGS.app_user, "t": time.time()}),
        httponly=True,
        samesite="lax",
        secure=False,  # platform terminates TLS; cookie must survive the hop
        max_age=60 * 60 * 12,
    )


def require_auth(request: Request) -> str:
    raw = request.cookies.get(COOKIE)
    if not raw:
        raise HTTPException(status_code=401, detail="not authenticated")
    try:
        data = _serializer.loads(raw, max_age=60 * 60 * 12)
    except BadSignature as exc:
        raise HTTPException(status_code=401, detail="invalid session") from exc
    return str(data.get("u", ""))


@app.post("/api/login")
async def login(payload: dict[str, str]) -> JSONResponse:
    user = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not SETTINGS.app_password:
        raise HTTPException(
            status_code=500,
            detail="APP_PASSWORD is not configured on the server",
        )
    ok = secrets.compare_digest(user, SETTINGS.app_user) and secrets.compare_digest(
        password, SETTINGS.app_password
    )
    if not ok:
        # Uniform delay so a wrong username and a wrong password are
        # indistinguishable by timing.
        await asyncio.sleep(0.4)
        raise HTTPException(status_code=401, detail="invalid credentials")
    resp = JSONResponse({"ok": True, "user": user})
    _issue(resp)
    return resp


@app.post("/api/logout")
async def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/api/me")
async def me(user: str = Depends(require_auth)) -> dict[str, Any]:
    return {"user": user, "config": redacted_summary(),
            "cost_ceiling_per_audio_min": COST_CEILING_PER_AUDIO_MIN}


# --- batches --------------------------------------------------------------

@app.post("/api/batches")
async def create_batch(
    files: list[UploadFile], user: str = Depends(require_auth)
) -> dict[str, Any]:
    """Upload a ZIP or a set of files. Runs pre-flight; does NOT start work."""
    blobs: list[tuple[str, bytes]] = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="upload exceeds 512 MB")
        blobs.append((f.filename or "unnamed", data))

    workdir = UPLOAD_ROOT / f"stage-{secrets.token_hex(6)}"
    try:
        from batch import stage_upload

        staged = stage_upload(blobs, workdir)
        report = preflight(staged)
    except BatchError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_id = store.create_batch(report, str(workdir), len(report["processable"]))
    return {"batch_id": batch_id, "preflight": report}


@app.post("/api/batches/{batch_id}/run")
async def start_batch(batch_id: str, user: str = Depends(require_auth)) -> dict[str, Any]:
    b = store.get_batch(batch_id)
    if not b:
        raise HTTPException(status_code=404, detail="unknown batch")
    if b["status"] != "pending":
        return {"ok": True, "status": b["status"]}
    if b["preflight"].get("blocking"):
        raise HTTPException(status_code=400, detail=b["preflight"]["blocking"])

    workdir = Path(b["workdir"])
    truths: dict[str, dict[str, Any]] = {}
    manifest_name = b["preflight"].get("manifest")
    if manifest_name and (workdir / manifest_name).exists():
        truths, _ = parse_manifest((workdir / manifest_name).read_bytes())

    asyncio.create_task(
        run_batch(store=store, batch_id=batch_id, workdir=workdir,
                  truths=truths, settings=SETTINGS)
    )
    return {"ok": True, "status": "running"}


def _live_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Score predictions against manifest labels when they are present.

    This is what turns validation from a claim in a document into something the
    evaluator can verify in the browser on their own data.
    """
    pairs = [
        (r["_truth"], r)
        for r in rows
        if r.get("status") == "ok" and r.get("_truth")
        and all(k in r["_truth"] for k in ("emotional_tone",))
    ]
    if not pairs:
        return None
    truths = [t for t, _ in pairs]
    preds = [{k: p.get(k) for k in FIELD_ORDER} for _, p in pairs]
    result = score_batch(truths, preds)
    result["caveat"] = (
        f"n={result['n']}. Fewer than ~30 labelled clips supports no statistical "
        f"claim; treat this as raw agreement, not a metric."
        if result["n"] < 30 else None
    )
    return result


@app.get("/api/batches/{batch_id}")
async def get_batch(batch_id: str, user: str = Depends(require_auth)) -> dict[str, Any]:
    b = store.get_batch(batch_id)
    if not b:
        raise HTTPException(status_code=404, detail="unknown batch")
    rows = store.get_results(batch_id)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    total_audio = sum(float(r.get("duration_s") or 0) for r in ok_rows)
    total_cost = sum(float(r.get("cost_usd") or 0) for r in ok_rows)
    return {
        "batch": {k: b[k] for k in
                  ("id", "status", "n_total", "n_done", "n_error",
                   "created_at", "started_at", "finished_at")},
        "preflight": b["preflight"],
        "results": rows,
        "metrics": _live_metrics(rows),
        "totals": {
            "audio_seconds": round(total_audio, 2),
            "cost_usd": round(total_cost, 6),
            "cost_per_audio_min": (
                round(total_cost / (total_audio / 60), 6) if total_audio else None
            ),
            "ceiling": COST_CEILING_PER_AUDIO_MIN,
            "mean_latency_s": (
                round(sum(float(r.get("latency_s") or 0) for r in ok_rows) / len(ok_rows), 2)
                if ok_rows else None
            ),
        },
    }


@app.get("/api/batches")
async def list_batches(user: str = Depends(require_auth)) -> dict[str, Any]:
    return {"batches": store.list_batches()}


@app.get("/api/batches/{batch_id}/results.json")
async def download_json(batch_id: str, user: str = Depends(require_auth)) -> Response:
    rows = store.get_results(batch_id)
    payload = {
        r["name"]: (
            {k: r.get(k) for k in FIELD_ORDER}
            if r.get("status") == "ok"
            else {"status": "error", "reason": r.get("reason", "")}
        )
        for r in rows
    }
    return Response(
        json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{batch_id}-results.json"'},
    )


@app.get("/api/batches/{batch_id}/results.csv")
async def download_csv(batch_id: str, user: str = Depends(require_auth)) -> Response:
    return Response(
        results_to_csv(store.get_results(batch_id)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{batch_id}-results.csv"'},
    )


# --- ops ------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "gemini_enabled": SETTINGS.gemini_enabled}


@app.on_event("startup")
async def _startup() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(_retention_sweep())


async def _retention_sweep() -> None:
    """Delete staged audio past the retention window.

    Batches normally purge themselves the moment they finish; this catches
    uploads that were never run, or a batch interrupted by a restart.
    """
    while True:
        try:
            cutoff_s = SETTINGS.upload_retention_hours * 3600
            for b in store.batches_to_purge(cutoff_s):
                shutil.rmtree(b["workdir"], ignore_errors=True)
                store.mark_purged(b["id"])
        except Exception:  # noqa: BLE001 - sweeper must never kill the app
            pass
        await asyncio.sleep(900)


@app.get("/")
async def index() -> Response:
    page = WEB_DIR / "index.html"
    if not page.exists():
        return PlainTextResponse("UI not built", status_code=500)
    return FileResponse(page)
