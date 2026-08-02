# Technical Memo — Voice Tone & Background Noise Analysis

**Rishi Singh** · AutoAce AI technical trial · August 2026
**Live system:** https://autoace-voice-tone.onrender.com

---

## 1. The problem, and the decision that shaped everything else

The brief asks for validation results, per-class performance and a confusion
matrix — and provides **three labelled calls**. Three samples support no
statistical claim of any kind. Every accuracy number computed on them is noise.

This is not a hypothetical risk. Early in the project my system scored **1/3 on
`emotional_tone`** against those three calls. It looked like partial success. It
was actually a constant predictor: it answered `neutral` for every clip, and one
of the three calls happens to be `neutral`. A constant and a working classifier
are indistinguishable at n=3.

So the first substantial thing I built was not a model. It was a **500-clip
synthetic validation set whose labels are known by construction**, drawn from a
115-speaker pool of which 102 appear in the set (61 fit / 41 eval, disjoint), and every metric in this memo comes from its
evaluation split. The three provided calls are used only as a sanity check and
are reported as raw agreement, explicitly labelled "n=3, not a metric".

The proxy set immediately earned its cost: thresholds that scored **3/3 on the
provided calls collapsed to 3/16 on it**. Details in §5.

---

## 2. What the audio actually is

Before modelling anything I measured the provided files. Three findings changed
the architecture; all are reproducible with
`uv run python -m autoace.forensics data/provided_calls`.

**Duplicated mono, not dual-channel.** The files are stereo, but `RMS(L−R)` is
−84 to −92 dBFS with **91–98% of samples bit-identical** and Pearson r > 0.99999.
The source is mono, encoded as stereo. Agent and customer are summed into one
signal, so speaker overlap cannot come from cross-channel energy and the customer
cannot be isolated by channel.

**Wideband VoIP, not 8 kHz telephony.** Encoder tag is `GStreamer opusenc`,
48 kHz. The 3500–4000 Hz and 4000–5000 Hz bands sit within 0.2 dB of each other —
G.711 would put a 30 dB wall there. The proxy set's degradation chain therefore
terminates in 48 kHz Opus, not G.711, with container carried as an explicit
factor.

**No meaningful clipping.** Two files peak above 0 dBFS, which looks like
clipping and is not: call_001 has **3 samples at full scale out of 1,485,105 with
zero flat-topped runs**. That is Opus intersample overshoot. Clipping is detected
by run-length and flat-top analysis, never by peak dBFS.

**The agent is AutoAce's own TTS voice** ("Erica", identical scripted opener in
all three calls, and an explicit "transferring you to an advisor" in call_003).
That turned out to be exploitable — see §4.

---

## 3. Approaches tested

The brief asks for at least two materially different approaches. There are four,
and they are genuinely different in kind: an audio foundation model, hand-written
signal processing, a fitted classifier over prosodic features, and a purpose-built
neural segmentation model. Each owns the fields it measurably wins.

### Path A — Gemini audio foundation model

`gemini-2.5-flash-lite`, structured output, evidence fields ordered *before*
labels so the model must commit to separate observations for noise, quality and
emotion before choosing any of them.

**Where it excels:** semantic customer identification. On a summed mono mix it
identified the customer correctly in **3/3** calls with a defensible reason each
time. The acoustic alternative (F0 clustering) gave only ~2.5σ separation. This
is genuinely hard and Gemini does it well.

**Where it fails, and why it is not fixable by prompting:** it reads words, not
voice. [arXiv:2510.10444](https://arxiv.org/pdf/2510.10444) tested Gemini 2.5
directly and found accuracy collapses when the emotional cue is prosodic rather
than lexical. Our ground truth contains two cases that invert a text reading — a
**flatly delivered obscenity labelled `neutral`**, and a customer **refused
throughout, labelled `satisfied`**. Gemini got both backwards.

It is also simply blind on some fields: `speaker_overlap_present` returned
`false` on every clip under every prompt variant tried, including one explicitly
asking about interruptions. On one clip it wrote *"a faint hiss throughout the
recording"* in its evidence and then set `background_noise_present: false`.

### Path B — deterministic acoustic analysis

numpy + ffmpeg, no model weights. Noise estimated by **minimum statistics**
(per-bin running minimum), speech/non-speech structure from an adaptive VAD,
clipping by run-length, gaps by speech-absence runs.

### Path D — pyannote segmentation-3.0 for overlapped speech

Added after the diagnosis in §6 showed the field was tractable. A **5.7 MB ONNX
export**, CPU-only through onnxruntime, no torch and nothing fetched at boot —
the checkpoint is baked into the image at build time.

**Licence is why this one is usable and the speech-emotion candidates are not.**
`pyannote/segmentation-3.0` is **MIT** (© 2022 CNRS). The two strongest
dimensional-SER models are not: audEERING's MSP-dim is CC-BY-NC-SA-4.0 and
emotion2vec+ is restricted by its training data. Fine for a demo, not for the
product this is a trial for.

The network emits a powerset over 3 speakers with at most 2 concurrent, so
overlap is simply `argmax(frame) >= 4` — no threshold tuning in reading the
model. The one fitted parameter is how much overlap counts as "enough to affect
understanding", the brief's wording, chosen on the fit split.

### Path C — prosodic features + lightweight classifier

The brief names "acoustic features plus a lightweight classifier" as a valid
approach, and it is the right one here. Every off-the-shelf SER model is
**660 MB–1.1 GB** against a 512 MB instance, and the two strongest are
**licence-blocked for commercial use** (audEERING MSP-dim is CC-BY-NC;
emotion2vec+ is restricted by its training data — my original plan specified
emotion2vec+, so following it unchecked would have shipped a licence problem).

26 prosodic features — F0 statistics and contour, energy dynamics, speaking rate,
pause structure, jitter, shimmer, HNR, spectral tilt — into a multinomial
logistic regression. **Coefficients ship as 8.8 KB of JSON**; inference is a
numpy dot product, so the container gains no dependency.

Features target **arousal and valence** rather than emotion categories, because
the five target labels map onto those two axes far more cleanly than onto any
categorical emotion set.

---

## 4. Final architecture

Authority is assigned per field from **measured capability**, not preference.

| field | source | why |
|---|---|---|
| `emotional_tone`, `emotional_intensity` | **Path C** | macro-F1 0.479 vs 0.082 for majority baseline; Path A reads lexical content |
| `background_noise_*` | **Path B** | Path A wrote "faint hiss" then labelled it absent |
| `background_noise_type` | Path B presence + Path A naming | Gemini names sources ("television") that spectral shape only approximates |
| `audio_quality` | **Path B** | Path A pins `slightly_impaired`, mistaking the synthetic agent voice for a defect |
| `speaker_overlap_present` | **Path D** | pyannote `segmentation-3.0`: 0.792 balanced accuracy vs 0.544 for the hand-built cue, same labels and split. Path A returns `false` universally |
| `long_silence_present` | Path B, Path A may veto | the brief scopes it to silence "indicating a call-flow problem" — a semantic judgement |

**TTS/human separation.** Since the agent is synthetic, prosodic features are
extracted from the human speaker only where possible: 2-means clustering of
voiced runs on (jitter, periodicity) separates synthetic from human speech, which
is far easier than human-vs-human diarization and needs no model. It succeeds on
2/3 provided calls and **declines rather than guessing** on the third. On
call_002 it retains 26% of frames as human; the independent transcript-based
estimate is 23% — two unrelated methods agreeing.

**Independence by construction.** Path B never sees transcript or emotion, so it
cannot infer noise from tone. Noise is measured on the noise PSD, quality on
speech frames. This is what the brief's scoring warning demands, enforced
structurally rather than by instruction.

---

## 5. Validation

**Proxy set:** 500 clips · 102 speakers from a 115-speaker pool · **fit 271 / eval
229, speakers disjoint (61 / 41)** · noise files split disjointly · babble built only from fitting-split
speakers · all seeds logged. Degradation axes: noise SNR {∞,30,20,15,10,5,0} ·
quality defects applied *independently* of noise · overlap {0,1.5,4}s · silence
{0,2,5,12}s · three container chains · concatenated multi-utterance bases.

**Adversarial cells (44 each), crossed deliberately:** loud-but-satisfied,
distorted-but-quiet-background, clean-but-noisy, and
lexically-hostile-but-prosodically-flat — the last added because the literature
predicts exactly that failure for audio LLMs.

### Results on the evaluation split (unseen speakers, unseen noise files)

| field | metric | result | baseline |
|---|---|---|---|
| `emotional_tone` | macro-F1 | **0.479** | 0.082 (B0 majority) |
| `emotional_tone` | accuracy | 0.493 | 0.258 |
| `emotional_intensity` | macro-F1 | 0.449 | 0.254 |
| `background_noise_present` | balanced acc | **0.920** (F1 0.939) | 0.603 |
| `background_noise_severity` | exact / within-1 | 0.703 / **0.930** | — |
| `audio_quality` | exact / within-1 | 0.472 / 0.782 | — |
| `speaker_overlap_present` | balanced acc | **0.792** (F1 0.789) | 0.520 |
| `long_silence_present` | balanced acc | **0.773** (F1 0.711) | 0.633 |

**Per-class F1 for `emotional_tone`:** upset 0.653 · neutral 0.639 · satisfied
0.430 · distressed 0.379 · **frustrated 0.295**.

> **These are not comparable to the numbers in an earlier draft of this memo.**
> The proxy set was regenerated after the label defect in §6 was found, so the
> evaluation clips are different audio. The tone figures moved 0.421 → 0.479, and
> almost none of that is a better model — see the ablation immediately below.

`frustrated` being worst was predicted before the experiment: it has no clean
acted analogue and is approximated from low-intensity anger and disgust. Per-class
F1 is reported precisely so this is visible rather than buried in an average.

### Prior correction: implemented, measured, and it does almost nothing

Acted corpora are balanced; real call audio is overwhelmingly neutral. That is a
textbook prior shift, and the textbook fix is **logit adjustment** — add
`τ·log π_target` to each class logit, which for a logistic regression is a
per-class offset on the intercept and costs nothing at inference.

τ is chosen on **held-out speakers inside the fit split**, never on the eval
split and never on the three provided calls. The result:

| field | τ chosen | eval macro-F1 at τ=0 | at chosen τ | delta |
|---|---|---|---|---|
| `emotional_tone` | 0.10 | 0.477 | 0.479 | **+0.002** |
| `emotional_intensity` | 0.00 | 0.449 | 0.449 | **+0.000** |

**This is a negative result and I am reporting it as one.** It is also the
expected one: the proxy set *is* the acted, balanced distribution, so shifting
toward a spontaneous prior cannot help on it. For intensity the search chose
τ=0 — the correction switched itself off rather than being switched off by me.

The mechanism is worth shipping anyway, because it is the correct lever and it
is now a one-number change: if AutoAce can supply the real tone distribution
across their calls, `DEPLOY_TONE_PRIOR` and τ are where it goes, with no
retraining. What cannot be done is *validating* that setting against acted data,
so the prior in the code is flagged as an estimate rather than a measurement.

**Leakage prevention:** grouped by speaker; no source clip in both splits under
any degradation; noise files disjoint; scaler statistics computed on the fit
split only; sklearn used for fitting only, never shipped.

---

## 6. Failure modes and limitations

**`speaker_overlap_present` was not unsolved — my labels were wrong.** This is
the finding I would most want to be asked about.

The field sat at 0.577 balanced accuracy against a 0.524 baseline, and the
correlation between injected overlap duration and the dual-pitch feature was
**0.069**. I had written it off as a hard modelling problem. It was not. My
proxy generator placed the interrupting talker at a **uniformly random offset**,
but the concatenated bases are only ~42% voiced, so the interrupter usually
landed in a pause and overlapped nothing. Measured over 40 placements per
condition: a **1.5 s** requested overlap produced a median of **0.28 s** of real
two-voice time and cleared the 0.5 s label threshold in **8%** of draws; a
**4.0 s** request produced **0.46 s** and cleared it **42%** of the time.

So most clips labelled `speaker_overlap_present=true` contained less overlap
than the generator's own definition required. Every detector was being scored
against noise, and no detector can beat a label that does not describe its audio.

Two changes fixed the generator — place the interrupter on voiced speech, and
derive the label by **measuring** the result instead of trusting the request —
and the field is now served by pyannote `segmentation-3.0` (§4).

| | balanced accuracy | correlation with true overlap |
|---|---|---|
| dual-pitch cue, on the corrected labels | 0.544 | 0.069 |
| **pyannote segmentation-3.0** | **0.792** (F1 0.789) | **0.388** |

The lesson generalises past this field: **when a metric says a problem is
unsolvable, check that the labels describe the audio before believing it.** The
same class of defect had already bitten this project twice (the `mix_noise`
percentile bug and the silence threshold), which is exactly why I looked.

**Acted-corpus optimism.** Tone thresholds are fitted on RAVDESS and CREMA-D. The
SER literature is explicit that acted prosody exaggerates cues relative to
spontaneous speech, so **0.479 macro-F1 is an optimistic bound** for real
dealership calls. The derived fields do not have this problem — their ground
truth comes from my own degradation chain. Correct fix: MSP-Podcast (324 h
naturalistic), whose request form would not clear inside this deadline.

**`audio_quality` is weak** (exact 0.454). Within-one is 0.769, so errors are
mostly adjacent rather than wild, but the clear/slightly_impaired boundary is
substantially a judgement call and my proxy definitions may not match AutoAce's.

**`long_silence_present` carries a semantic qualifier.** call_003 contains 6.7 s
of genuine dead air at the clip's noise floor and is labelled `false`. The brief
scopes the field to silence "that may indicate a call-flow or audio problem" — a
mid-call pause while an advisor looks something up is not a fault. A pure
duration rule cannot express this, which is why Path A retains a veto.

**Mono mixing caps everything.** Agent and customer are summed. If AutoAce can
export dual-channel recordings, overlap becomes a trivial cross-channel energy
computation and customer isolation becomes exact. **This is the single highest-value
change available, and it is on the recording side, not the model side.**

**Untested:** non-native accents, code-switching (call_002 is partly Spanish),
calls over ~10 minutes, and any hidden-set format other than the three tested.

---

## 7. Next steps, in order of expected value

1. **Dual-channel recording export** — removes the hardest constraint entirely.
2. **Label 200–500 real calls with 3 annotators**, and report inter-annotator
   agreement. Human agreement on 5-class tone is typically the real ceiling; we
   currently have no idea what it is here.
3. **Replace the overlap heuristic** with a trained overlapped-speech model.
4. **Refit tone on in-domain audio** once real labels exist; the acted-corpus
   bias is the largest known error source.
5. **Per-field confidence calibration** against real labels. `confidence` is
   currently a fused agreement score; the provided labels ship a constant 0.82
   placeholder, so it cannot be calibrated against them.

---

## 8. Cost, latency, privacy

Full detail in [`outputs/validation/cost_analysis.md`](outputs/validation/cost_analysis.md),
[`latency_analysis.md`](outputs/validation/latency_analysis.md) and
[`COMPLIANCE.md`](COMPLIANCE.md).

**Cost: $0.000875 per audio-minute — 3.4× under the $0.003 ceiling.** Measured
from live `usageMetadata`, not estimated. Paths B and C are $0. Reported
*uncached*: implicit caching made repeated benchmark runs look 3.5× cheaper, and
quoting that number would have been misleading.

**Latency:** 4–5 s per clip locally; **40 s per clip on the deployed free plan**,
which allocates 0.1 CPU. The analysis paths are CPU-bound, so the Starter plan's
0.5 CPU would cut this roughly 5x. Documented rather than hidden — see
`latency_analysis.md`.

**A production bug worth recording:** the container was OOM-killed at 512 MiB on
the 172-second clip. `analyse_acoustics` was decoding and framing the whole clip
a second time via `extract_features`, frame matrices were float64, and the F0
tracker allocated an (n_frames x 1024) complex array for the entire clip. Peak
went 637 MB -> 286 MB, and long clips are now chunked so memory is bounded by
chunk rather than by duration.

**Privacy:** paid Gemini tier only — free tier trains on submitted content and
would breach the confidentiality constraint. Audio deleted on batch completion or
after 24 h. Never logged. `LOCAL_ONLY=true` runs Paths B, C and D with **zero
data egress** and still produces all nine fields; it loses only Gemini's semantic
customer identification and its naming of the noise source. Path D's weights are
MIT-licensed and baked into the image, so the zero-egress mode needs no network
at all — not even at startup.
