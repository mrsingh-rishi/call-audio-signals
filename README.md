# AutoAce — Voice Tone & Background Noise Analysis

Classifies emotional tone and detects background noise / technical audio issues in
production call recordings, emitting the nine-field JSON contract defined in the trial brief.

**Hosted dashboard:** <https://autoace-voice-tone.onrender.com>
Sign in with the `APP_USER` / `APP_PASSWORD` credentials supplied with the submission
(not committed to this repository). See [Deploy](#deploy) to run your own instance.

---

## Quick start

Requires Python 3.11+ and `ffmpeg` on PATH.

```bash
git clone <repo> && cd autoace-voice-tone
uv venv --python 3.12 && uv pip install -e ".[server,gemini,overlap,dev]"
./scripts/fetch_model.sh  # 5.7 MB speaker-segmentation checkpoint for Path D
cp .env.example .env      # then fill in GEMINI_API_KEY, APP_PASSWORD, SESSION_SECRET
```

> `fetch_model.sh` is optional. Without it Path D reports itself unavailable and
> `speaker_overlap_present` falls back to Path B's weaker cue rather than failing.
> The Docker image runs the equivalent step at build time.

> **The Gemini key must be paid-tier.** Free-tier content is used to improve Google's
> products, which conflicts with the confidentiality constraint in the brief. See
> [COMPLIANCE.md](COMPLIANCE.md).

### Run the dashboard locally

```bash
uv run uvicorn main:app --app-dir server --port 8000
```

Open http://127.0.0.1:8000 and sign in with `APP_USER` / `APP_PASSWORD`.

### Command line

```bash
# Reproduce every forensic measurement quoted in the memo
uv run python -m autoace.forensics data/provided_calls

# Predictions for the three provided calls (deliverable 4)
uv run python scripts/predict_provided.py

# Role-labelled transcripts + customer-identification gate
uv run python scripts/probe_customer_id.py

# Tests, including deliberate-failure isolation cases
uv run pytest -q
```

---

## What it does

For each clip it emits exactly:

```json
{
  "emotional_tone": "neutral | satisfied | frustrated | upset | distressed",
  "emotional_intensity": "low | medium | high",
  "background_noise_present": true,
  "background_noise_type": "free text, empty when no noise",
  "background_noise_severity": "none | low | medium | high",
  "audio_quality": "clear | slightly_impaired | severely_impaired",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.82
}
```

The contract lives in one place — [`src/autoace/schema.py`](src/autoace/schema.py) — which
generates the Gemini `response_schema`, the validators and the label definitions used in
prompts. They cannot drift apart.

## Architecture

```
                 ┌─► path_a_gemini.py   tone evidence, noise NAMING, silence veto
                 │     (1 API call, evidence-ordered schema, thinking off)
                 ├─► path_b_acoustic.py noise presence/severity, quality, silence
audio ─► ingest ─┤     (numpy + ffmpeg, no weights)
                 ├─► path_c_prosody.py  emotional_tone, emotional_intensity
                 │     (26 prosodic features -> logistic regression, 8.8 KB JSON)
                 └─► path_d_overlap.py  speaker_overlap_present
                       (pyannote segmentation-3.0, ONNX, 5.7 MB, MIT)
                                  │
                                  ▼
                          fusion.py  per-field authority
                                  │
                                  ▼
                    9-field JSON + latency + measured cost
```

**Authority is assigned per field from measured capability**, not from a general
preference for one path. Gemini reads words rather than voice, so it does not own
tone; it does own naming the noise source, which spectral shape can only
approximate. Paths B, C and D never see a transcript, so they are structurally
incapable of the field conflation the brief warns about.

**Evidence-ordered schema.** The response schema declares `noise_evidence`,
`quality_evidence`, `customer_identification` and `emotion_evidence` *before* any label.
Generation is autoregressive, so ordering forces the model to commit to separate
observations per field group before choosing labels. This is the anti-conflation
mechanism the brief's scoring warning demands.

**Per-file failure isolation.** `predict.analyse_file()` never raises. Unreadable audio,
API errors and unparseable model output all become a result row with `status: "error"` and
a human-readable reason. A malformed file cannot fail a batch.

## Batch workflow

Upload a ZIP (or several files) containing audio at the root plus a `labels.csv` manifest
with `name` and `result_json` columns.

1. **Pre-flight** runs before any processing and reports: files in the manifest that were
   not uploaded, files uploaded that are not in the manifest, unsupported extensions,
   zero-byte files, and manifest parse problems. The manifest reader handles a UTF-8 BOM,
   CRLF line endings, quoted JSON containing commas, and an empty `result_json` (which is
   what an unlabeled hidden set looks like).
2. **Run** processes with bounded concurrency and live per-file progress.
3. **Results** show all nine fields plus per-file latency and measured cost, with failed
   files shown inline with their reason.
4. **Live validation** — if `result_json` is populated, the dashboard computes accuracy,
   macro-F1, per-class precision/recall/F1 and a confusion matrix *in the browser*, so the
   numbers can be verified rather than taken on trust.
5. **Download** as CSV or JSON, preserving original filenames.

## Privacy

- Uploaded audio is staged in a temporary directory and deleted the moment the batch
  finishes, or after `UPLOAD_RETENTION_HOURS` (default 24), whichever comes first.
- Audio content is never logged. ffmpeg error text is scrubbed of server paths before it
  reaches results or the UI.
- `LOCAL_ONLY=true` disables the Gemini path entirely for a zero-egress deployment.
- Secrets are reported only through `config.redacted_summary()`, which returns presence,
  never values.

## Deploy

Single Docker image — API, static UI and ffmpeg in one container. No node build step.

```bash
docker build -t autoace-vt .
docker run -p 8000:8000 --env-file .env autoace-vt
```

**Render:** [`render.yaml`](render.yaml) is a Blueprint. Set `GEMINI_API_KEY` and
`APP_PASSWORD` in the dashboard; `SESSION_SECRET` is generated.

Sizing is measured, not assumed: **peak RSS is 60 MB** processing a 5-file batch including
a 172 s clip, so the `starter` plan (512 MB) has roughly 8× headroom. Free tier is
deliberately *not* used — it spins down after 15 minutes with a ~60 s cold start, which
conflicts with the brief's requirement that the deployment stay available throughout the
evaluation period.

## Findings

Measured, reproducible results that shaped the design:

- [`outputs/validation/forensics_findings.md`](outputs/validation/forensics_findings.md) —
  channel layout, bandwidth, clipping and silence analysis of the provided calls.
- [`outputs/validation/t0_probe_findings.md`](outputs/validation/t0_probe_findings.md) —
  customer-identification gate, timestamp reliability, and evidence that the labels track
  vocal delivery rather than lexical content.

## Repository layout

| Path | Purpose |
|---|---|
| `src/autoace/schema.py` | The nine-field contract; single source of truth |
| `src/autoace/ingest.py` | Probe, decode, channel analysis, canonical normalisation |
| `src/autoace/forensics.py` | Reproducible measurement CLI |
| `src/autoace/path_a_gemini.py` | Gemini analysis, prompts, token/cost accounting |
| `src/autoace/path_b_acoustic.py` | Deterministic DSP: noise, quality, silence |
| `src/autoace/path_c_prosody.py` | 26 prosodic features + logistic regression for tone |
| `src/autoace/path_d_overlap.py` | pyannote segmentation-3.0 (ONNX) for speaker overlap |
| `src/autoace/fusion.py` | Per-field authority rules between the four paths |
| `src/autoace/predict.py` | Orchestrator; guarantees per-file isolation |
| `src/autoace/transcript.py` | Role-labelled turns, timestamp repair |
| `src/autoace/metrics.py` | Scoring: macro-F1, ordinal off-by-one, confusion matrices |
| `server/` | FastAPI app, batch intake, SQLite state |
| `web/index.html` | Self-contained dashboard |
| `tests/` | Schema contract and failure-isolation tests |

## Status

**Built and verified end-to-end:** forensics, all analysis paths, fusion, the batch
pipeline, the dashboard with live scoring and downloads, per-file failure isolation, and
cost/latency measurement. The 500-clip proxy validation set, the B0 majority baseline and
the technical memo are complete — see [MEMO.md](MEMO.md) and
[validation_report.md](outputs/validation/validation_report.md).

**Deliberately not built:** long-call windowing. It is designed (see the memo's
limitations section) but not shipped — cost per audio-minute is flat in duration, so it
buys nothing at current clip lengths.

**Known weak fields**, reported rather than hidden: `emotional_tone` carries an
acted-corpus optimism bias, and `audio_quality` is 0.472 exact on the proxy eval split.
Both are quantified in the validation report. Accuracy is measured on the proxy set, never
tuned against the three provided labels — thresholds that scored 3/3 on those three calls
scored 3/16 on the proxy set, which is what motivated building it.
