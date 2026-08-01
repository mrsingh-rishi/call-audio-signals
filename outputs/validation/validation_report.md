# Validation Report

Deliverable 6: metric used, per-class performance, confusion matrix.

Reproduce with:

```bash
uv run python -m autoace.proxy.build --n 500   # regenerate the set (seeded)
uv run python scripts/fit_detectors.py         # objective fields
uv run python scripts/fit_tone.py              # tone + intensity
```

---

## 1. Why a proxy set exists at all

The brief provides **three labelled calls**. Three samples cannot support a
confusion matrix, per-class F1, cross-validation or calibration — any number
computed on them is noise.

The concrete demonstration: an early version of this system scored **1/3 on
`emotional_tone`** against the three calls. That looked like partial success. It
was a constant predictor answering `neutral` every time, and one of the three
calls is `neutral`. **At n=3 a constant predictor and a working classifier are
indistinguishable.**

So all metrics below come from a 500-clip synthetic set whose labels are known by
construction. The three real calls are reported separately as raw agreement and
explicitly labelled *"n=3, not a metric."*

---

## 2. Proxy set construction

| | |
|---|---|
| clips | 500 |
| speakers | 115 (RAVDESS 24 + CREMA-D 91) |
| **split** | **fit 271 / eval 229 — speakers disjoint** |
| tone balance | neutral 83 · satisfied 127 · frustrated 82 · upset 103 · distressed 105 |
| formats | Opus/ogg 164 · wav16 179 · G.711/mp3 157 |
| noise present | 278 | 
| overlap present | 240 |
| long silence | 208 |

**Degradation axes**, all seeded and recorded so a manifest row fully determines
the audio:

- noise SNR ∈ {∞, 30, 20, 15, 10, 5, 0} dB
- quality defects applied **independently of noise**: clipping (calibrated to
  produce genuine flat-topped runs), 20–200 ms packet loss, RIR echo, −20 dB
  level, band-limiting, low-bitrate codec
- overlap ∈ {0, 1.5, 4} s · silence ∈ {0, 2, 5, 12} s
- three container chains · concatenated multi-utterance bases (single 3.5 s
  corpus utterances make a 12 s injected silence meaningless)

**Adversarial cells, 44 each, placed deliberately rather than left to chance:**
`loud_but_satisfied` · `distorted_quiet_background` · `clean_but_noisy` ·
`lexical_prosody_conflict`.

### Leakage prevention

- **Grouped by speaker.** No speaker in both splits, so no threshold is fitted to
  a voice it is later scored on.
- **Noise files split disjointly** between fit and eval.
- **Babble built only from fitting-split speakers**, so an evaluation voice never
  appears inside another clip's noise.
- **Scaler statistics computed on the fit split only** — no evaluation data
  leaks through standardisation.
- All seeds logged in `data/proxy/build_info.json`.
- scikit-learn is used for fitting only and never ships; coefficients export to
  16 KB of JSON and inference is a numpy dot product.

---

## 3. Metric choice

**Macro-F1** is the headline for `emotional_tone`, as the brief names it and
because the classes are imbalanced — a micro average would let a dominant class
mask total failure on a rare one.

**Ordinal fields report exact *and* within-one.** Predicting `medium` when the
truth is `high` is not the same error as predicting `none`, and exact-match alone
hides that distinction.

---

## 4. Results — evaluation split (unseen speakers, unseen noise files)

### `emotional_tone` (Path C, prosodic)

| | accuracy | macro-F1 |
|---|---|---|
| **Path C — prosodic classifier** | **0.428** | **0.421** |
| B0 — majority baseline | 0.258 | 0.082 |

**5.1× the baseline on macro-F1.** Chance on five balanced classes is ~0.20.

**Per-class F1:**

| class | F1 | n |
|---|---|---|
| upset | **0.600** | 44 |
| neutral | 0.505 | 37 |
| satisfied | 0.393 | 59 |
| distressed | 0.358 | 53 |
| **frustrated** | **0.250** | 36 |

**Confusion matrix** (rows = truth, columns = predicted):

| | neutral | satisfied | frustrated | upset | distressed |
|---|---|---|---|---|---|
| **neutral** | **23** | 3 | 9 | 0 | 2 |
| **satisfied** | 12 | **21** | 5 | 8 | 13 |
| **frustrated** | 10 | 7 | **8** | 2 | 9 |
| **upset** | 0 | 5 | 2 | **27** | 10 |
| **distressed** | 9 | 12 | 4 | 9 | **19** |

Reading it: `upset` is cleanest (27/44, and never confused with `neutral` — high
arousal is the easiest thing to hear). `frustrated` is the weakest by a wide
margin, scattering across every other class.

**That was predicted before the experiment ran.** `frustrated` has no clean
analogue in acted corpora and is approximated from low-intensity anger and
disgust. Per-class F1 is reported precisely so this is visible rather than
averaged away.

### `emotional_intensity`

| | accuracy | macro-F1 |
|---|---|---|
| Path C | 0.629 | **0.434** |
| B0 majority | 0.616 | 0.254 |

Accuracy barely moves because the eval split is dominated by `high` (141/229),
but macro-F1 nearly doubles. `low` has F1 0.000 on **n=4** — RAVDESS provides
only two intensity levels, so genuine low-intensity examples are almost absent.
That is a proxy-set limitation, not a model result.

### Objective fields (Path B, deterministic)

| field | metric | result | baseline |
|---|---|---|---|
| `background_noise_present` | balanced acc / F1 | **0.916 / 0.935** | 0.603 |
| `background_noise_severity` | exact / within-1 | 0.699 / **0.930** | — |
| `audio_quality` | exact / within-1 | 0.454 / 0.769 | — |
| `speaker_overlap_present` | balanced acc | **0.559** | 0.524 |
| `long_silence_present` | balanced acc | 0.638 → improved, see §5 | 0.633 |

---

## 5. What the proxy set caught

**This is the section that justifies its cost.**

**Thresholds tuned on n=3 scored 3/16 on the proxy set — worse than chance.**
They had scored 3/3 on the provided calls.

Three specific defects, none of which were visible with three clips:

1. **Noise measured on VAD-inverted regions inverts at low SNR.** When noise is
   as loud as speech, the VAD marks noisy frames as *speech*, the "non-speech"
   region empties, and the loudest clips scored as having no noise. Replaced with
   **minimum-statistics** noise estimation, which needs no speech/non-speech
   decision. `background_noise_present` went to 0.916 balanced accuracy.

2. **The silence detector thresholded below the codec noise floor.** A fixed
   −35 dB below a −47 dBFS speech level puts the threshold at −82 dBFS, but
   injected digital silence lands at ~−70 dBFS after Opus/MP3. Injected 12-second
   gaps were never detected. Re-anchored to speech *absence* rather than signal
   absence: correlation between injected and detected silence went **0.477 →
   0.767**, with injected 12 s now measuring 10.93 s median.

3. **The generator itself was writing wrong labels.** `mix_noise` estimated
   speech level at percentile-60, but concatenated clips are >40% silent, so the
   cut landed inside the silence and noise was scaled far too quietly — "0 dB
   SNR" clips were actually 20–50 dB. The ground truth, not just the detector,
   was wrong.

---

## 6. The three provided calls — n=3, not a metric

Reported as raw agreement only. See `outputs/predictions_provided.json`.

The hybrid system reaches **5/7 scored fields correct** on these clips
(`background_noise_present`, `speaker_overlap_present`, `audio_quality`,
`long_silence_present` exact; `background_noise_severity` within-one; noise type
matching by token overlap — `television`/`TV`, `static`/`sharp static`).

No claim is made from this. It is a smoke test that the pipeline produces sane
output on real production audio, nothing more.

---

## 7. Confidence and ambiguity handling

`confidence` is a fused agreement score: it starts from Path A's self-reported
value, is penalised 0.08 per disagreement between paths, and 0.15 more when only
one path ran.

**It is deliberately not calibrated against the provided labels**, because those
labels carry a constant `0.82` on all three calls — and `0.82` is also the value
in the brief's own example output. It is a copied placeholder, not ground truth.
Fitting a calibrator to a constant would produce a constant.

Ambiguity is handled by the fusion rules being explicit and inspectable: every
field records which path produced it (`sources`) and every inter-path
disagreement is retained (`disagreements`), so a low-confidence result can be
traced to the specific conflict that caused it.
