# Diagnosis: why every clip returned identical labels

Reproduce with:

```bash
uv run python scripts/diagnose_collapse.py
```

---

## 1. It is not a mock, and the audio does reach the model

Three independent proofs:

| Evidence | call_001 | call_002 | call_003 |
|---|---|---|---|
| SHA-256 of bytes sent | `50b5188d172f8954` | `b7c9e8355f80c91a` | `694ad688d9fcbbd4` |
| prompt tokens billed | 1,800 | 1,928 | 6,311 |
| model's `emotion_evidence` | *"'Are you a real person?' delivered with a slightly hesitant and questioning tone"* | *"tone is flat and monotonous… 'Español, mamahuevo' with no discernible change in vocal delivery"* | *"generally neutral and polite, with a slight hint of impatience"* |

Token counts scale with duration, and the evidence strings quote **different real
content from each file**. The model was hearing three different calls and then
emitting one answer. A grep for `mock|stub|fake|hardcoded|dummy` across `src/`,
`server/` and `scripts/` returns nothing.

## 2. Root cause: the system prompt, not the model

Same three clips, same model, same temperature — only the system instruction changed:

| Variant | System instruction | Fields that varied across the 3 clips |
|---|---|---|
| **A — shipped version** | 3,953 chars | **NONE — fully collapsed** |
| B — none at all | – | `emotional_tone` |
| C — brief's definitions only | 2,228 chars | `emotional_tone`, `emotional_intensity`, `audio_quality` |
| D — definitions + neutral prosody cue | 2,851 chars | `emotional_tone`, `emotional_intensity`, `audio_quality` |

**Removing my prompt entirely made the model *more* discriminative than my
carefully written one.** That is the whole finding.

### The specific defect

The T0 probe found that ground-truth labels track vocal delivery rather than
lexical sentiment. The fix I wrote encoded that as worked examples:

> A customer who swears in a flat voice is **neutral**. A customer who is told "no"
> repeatedly but stays pleasant and agreeable is **satisfied or neutral, not upset**.

Those two sentences describe call_002 and call_003 almost exactly. Pairing a
concrete scenario with a named label turned the prompt into an answer key: the
model stopped classifying and started reciting. Because the examples all pointed
at the low-arousal end, every clip collapsed to `neutral` / `low`.

**Rule adopted:** never pair a scenario with a label name in a prompt. State the
decision criterion, never the answer.

### The fix

`_DELIVERY_RULE` now states the criterion without examples and without naming any
label. After the change the outputs differentiate:

| | call_001 | call_002 | call_003 |
|---|---|---|---|
| before | neutral / low | neutral / low | neutral / low |
| after | neutral / low | **upset / high** | neutral / low |

Accuracy is still poor. That is expected and deliberate — the thresholds were not
tuned until the three labels matched, because tuning on n=3 is precisely what
produced the bug.

## 3. Separate, larger finding: Path A is blind on two fields

Four fields never varied under **any** prompt variant. A directive probe
(explicitly instructing the model to listen for non-speech sound and talk-over)
isolated what is happening:

| clip | `noise_evidence` returned | `background_noise_present` | ground truth |
|---|---|---|---|
| call_001 | *(empty)* | false | false ✅ |
| call_002 | *"a faint hiss throughout the recording"* | **false** ❌ | true (`TV`, medium) |
| call_003 | *"faint, intermittent hiss throughout"* | true (`hiss`, low) | true (`sharp static`, medium) |

- **call_002 hears the noise and then labels it absent.** The model applies the
  brief's "barely perceptible artifacts should not count" clause too aggressively
  and contradicts its own evidence. The prompt now requires labels to agree with
  the evidence fields.
- **`speaker_overlap_present` returned false on all three clips under every
  variant, including one that explicitly asked about interruptions and talk-over.**
  Ground truth is true for two of the three.

### Consequence for the architecture

The plan treated the deterministic Path B as a *guard rail* that cross-checks
Path A. The measurement says otherwise:

> **Path A cannot do `speaker_overlap_present` at all, and systematically
> under-calls `background_noise_severity`. For those fields Path B is not a
> cross-check — it is the primary source.**

This raises Path B from optional to load-bearing, and it is now the highest-value
remaining work for the 45% hidden-set bucket. Overlap on a summed mono mix is a
tractable signal-processing problem (spectral flux, harmonic-count and
pitch-track discontinuity within voiced regions); it is evidently not a tractable
prompting problem.

`audio_quality` also remains pinned at `slightly_impaired` across all clips while
ground truth is `clear` 3/3 — the residual cause is the model treating the
synthetic TTS agent voice as a defect. The prompt now says explicitly that
synthetic timbre is not a quality defect; whether that is sufficient must be
measured on the proxy set, not on n=3.

## 4. Container verification

Built and run locally, not just described:

| Check | Result |
|---|---|
| `docker build` | succeeded, **983 MB** (dominated by the ffmpeg apt dependency tree; `pip install` layer is 154 MB) |
| ffmpeg in image | 7.1.5 |
| health check | `{"ok":true,"gemini_enabled":true}`, container reports `healthy` |
| full batch through container | 5/5 completed, 2 malformed files isolated with clean reasons |
| **peak container memory** | **119.3 MiB** — inside a 512 MB instance with ~4× headroom |
| cost in container | $0.000849 / audio-minute |
| mean latency | 4.4 s |
