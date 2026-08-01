"""Deliverable 4: predictions for the three provided calls, in the required schema.

Also acts as the first honest end-to-end test. Agreement against the three
labels is reported as RAW AGREEMENT, never as a metric - n=3 supports no
statistical claim. The proxy set carries the actual validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import COST_CEILING_PER_AUDIO_MIN, REPO_ROOT, get_settings  # noqa: E402
from autoace.ingest import AudioIngestError, probe  # noqa: E402
from autoace.metrics import format_report, score_batch  # noqa: E402
from autoace.path_a_gemini import analyse_single  # noqa: E402


def read_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the manifest. Tolerates BOM and CRLF, which real CSVs have."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            raw = (row.get("result_json") or "").strip()
            if not name or not raw:
                continue
            try:
                out[name] = json.loads(raw)
            except json.JSONDecodeError:
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REPO_ROOT / "data" / "provided_calls"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "predictions_provided.json"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    if not s.gemini_enabled:
        print("ERROR: Gemini disabled (no key or LOCAL_ONLY=true)", file=sys.stderr)
        return 2

    src = Path(args.dir)
    labels = read_labels(src / "labels.csv")
    files = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() not in {".csv", ".md"}
    )

    model = args.model or s.gemini_model
    print(f"Predicting {len(files)} clips | model={model} | "
          f"thinking_budget={s.gemini_thinking_budget}\n")

    predictions: dict[str, Any] = {}
    truths, preds = [], []
    total_cost = total_audio_s = 0.0
    t_wall = time.perf_counter()

    for path in files:
        try:
            pr = probe(path)
            analysis, meta = analyse_single(path, settings=s, model=model)
        except (AudioIngestError, RuntimeError, Exception) as exc:  # noqa: BLE001
            reason = getattr(exc, "reason", None) or f"{type(exc).__name__}: {exc}"
            print(f"{path.name}: ERROR - {reason}\n")
            predictions[path.name] = {"status": "error", "reason": str(reason)[:300]}
            continue

        usage = meta["usage"]
        cost = usage.cost_usd(model)
        per_min = usage.cost_per_audio_min(pr.duration_s, model)
        total_cost += cost
        total_audio_s += pr.duration_s

        out = analysis.to_output_dict()
        predictions[path.name] = out

        print(f"{'=' * 78}\n{path.name}  ({pr.duration_s:.1f}s)\n{'=' * 78}")
        print(json.dumps(out, indent=2))
        print(f"\n  latency {usage.latency_s:.2f}s | in={usage.prompt_tokens} "
              f"out={usage.output_tokens} thoughts={usage.thought_tokens} "
              f"cached={usage.cached_tokens}")
        print(f"  cost ${cost:.6f}  =  ${per_min:.6f}/audio-min  "
              f"({'OK' if per_min < COST_CEILING_PER_AUDIO_MIN else 'OVER CEILING'})")
        if meta["repairs"]:
            print(f"  schema repairs: {meta['repairs']}")
        for k, v in meta["evidence"].items():
            if v:
                print(f"  {k}: {v}")

        if path.name in labels:
            truths.append(labels[path.name])
            preds.append(out)
            diffs = [
                f"{k}: truth={labels[path.name][k]!r} pred={out[k]!r}"
                for k in out
                if k in labels[path.name] and k != "confidence" and labels[path.name][k] != out[k]
            ]
            print(f"\n  vs ground truth: "
                  f"{'ALL MATCH' if not diffs else str(len(diffs)) + ' mismatch(es)'}")
            for d in diffs:
                print(f"    - {d}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(predictions, indent=2))

    wall = time.perf_counter() - t_wall
    print(f"{'=' * 78}")
    print(f"wrote {args.out}")
    if total_audio_s:
        print(f"\nCOST: ${total_cost:.6f} over {total_audio_s / 60:.2f} audio-min "
              f"= ${total_cost / (total_audio_s / 60):.6f}/audio-min "
              f"(ceiling ${COST_CEILING_PER_AUDIO_MIN})")
        print(f"LATENCY: {wall:.1f}s wall for {total_audio_s / 60:.2f} audio-min "
              f"= {wall / (total_audio_s / 60):.1f}s per audio-min")

    if truths:
        print()
        print(format_report(score_batch(truths, preds),
                            "RAW AGREEMENT vs the three provided labels"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
