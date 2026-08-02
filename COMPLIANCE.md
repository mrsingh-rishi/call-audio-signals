# Data handling and external-API disclosure

Required by §11 of the trial brief: *"Any external paid API must be disclosed, including
model name, pricing assumptions, retention policy, and whether customer audio leaves
AutoAce-controlled infrastructure."*

---

## Third-party models and their licences

Licence is a hard filter here, not a footnote: this is a trial for a commercial
product, so a research-only checkpoint is unusable however good it is.

| Component | What it is | Licence | Runs where | Data leaves? |
|---|---|---|---|---|
| **Path A** — Gemini | audio LLM, see below | commercial API | Google | **yes** |
| **Path B** — acoustic DSP | numpy + ffmpeg, no weights | n/a | in-container | no |
| **Path C** — prosody classifier | 26 features + logistic regression, coefficients fitted by us | ours | in-container | no |
| **Path D** — `pyannote/segmentation-3.0` | overlapped-speech detection, ONNX, 5.7 MB | **MIT** (© 2022 CNRS) | in-container | no |

Path D's weights are baked into the image at build time from the k2-fsa ONNX
export, so nothing is fetched at boot and no Hugging Face token is needed. Only
Path A sends data anywhere.

**Models deliberately rejected on licence grounds**, recorded because the choice
cost measurable accuracy: audEERING `wav2vec2-large-robust-...-msp-dim` is the
de-facto standard for dimensional speech emotion but is CC-BY-NC-SA-4.0, and
`emotion2vec+ large` is restricted by its training data. Both are fine for a
demo and not for the product this is a trial for. See `RESEARCH.md` §4.

## External API in use

| | |
|---|---|
| **Provider** | Google — Gemini Developer API (`generativelanguage.googleapis.com`) |
| **Model** | `gemini-2.5-flash-lite` (set by `GEMINI_MODEL`; `gemini-3.1-flash-lite` is the forward path once 2.5 retires) |
| **Tier** | **Paid.** Required — see below |
| **SDK** | `google-genai` |
| **What is sent** | The audio file bytes, plus a text system instruction. No customer identifiers, no account data, no metadata beyond the audio itself |
| **Thinking budget** | 0 by default (`GEMINI_THINKING_BUDGET`) |

### Paid tier is mandatory, not a preference

Free-tier Gemini content is used to improve Google's products. The brief requires that
production call audio be treated as confidential and not uploaded to unapproved public
services. **Prototyping on the free tier — including the AI Studio web UI — would
therefore be a data-handling violation**, and would be one committed on the very first
action of the project.

The system refuses to send audio when no key is configured; it returns a per-file error
rather than silently degrading.

## Does customer audio leave AutoAce-controlled infrastructure?

**Yes, in the default configuration.** Audio is transmitted to Google's Gemini API over
TLS for inference. This is the trade-off that buys the accuracy and the cost profile
below, and it must be an explicit, informed choice.

**A zero-egress mode exists.** Setting `LOCAL_ONLY=true` disables the Gemini path
entirely. In that mode no audio leaves the container.

`LOCAL_ONLY=true` is a genuine alternative analysis mode, not merely a kill switch. The
deterministic acoustic path (Path B), the prosodic classifier (Path C) and the speaker-
overlap model (Path D) all run locally with no network access and nothing to download at
runtime, so the mode still produces all nine fields:
tone, intensity, noise presence/type/severity, quality, overlap and silence. What it loses
is Gemini's semantic customer identification and its naming of the noise source — Path B
falls back to naming the noise from its spectral character instead.

## Retention

| Where | Retention |
|---|---|
| Uploaded audio (server) | Deleted when the batch completes, or after `UPLOAD_RETENTION_HOURS` (default 24), whichever is first. A background sweep collects batches that were uploaded but never run |
| Prediction results (server) | Retained in SQLite. Contains labels and metrics only — no audio, no transcript |
| Transcripts (`data/transcripts/`) | Produced only by the T0 probe script. Contain customer speech, so the directory is gitignored and is not created by the server |
| Logs | Audio content is never logged. ffmpeg error output is scrubbed of server filesystem paths before it reaches results or the UI |
| Google | Paid-tier terms apply. Prompt data is not used to improve Google's products |

## Cost

Gemini bills audio at **32 tokens/second = 1,920 tokens per audio-minute**. This was
confirmed empirically against live `usageMetadata` (measured 32.4–34.2 tok/s across the
three provided calls; the rate converges on 32 as the fixed system-instruction overhead
amortises over longer clips).

Rates verified 2026-08-01, USD per 1M tokens:

| Model | Audio in | Text out |
|---|---|---|
| `gemini-2.5-flash-lite` | 0.30 | 0.40 |
| `gemini-3.1-flash-lite` | 0.50 | 1.50 |

**Measured end-to-end cost: $0.000875 per audio-minute** across 3.96 audio-minutes —
**3.4× under the $0.003 ceiling**. Cost is computed from live
`usageMetadata`, not estimated, and thinking tokens are billed at output rates explicitly
because an unbounded thinking budget is the usual way a per-minute ceiling gets breached
silently.

`gemini-2.5-flash-lite` retires **2026-10-16**. The model ID is configuration, not code:
migrating is a `GEMINI_MODEL` environment-variable change with no redeploy of source, and
`gemini-3.1-flash-lite` pricing is already in the table in `config.py` so the cost model
stays accurate across the switch. There is no automatic runtime failover between models —
the migration is a deliberate operator action, not something the service does silently.

## Secrets

No secret is ever written to a log, an API response, or the UI. `config.redacted_summary()`
is the only function permitted to describe configuration, and it reports presence
(`gemini_api_key_present: true`) rather than values. `.env` is gitignored.
