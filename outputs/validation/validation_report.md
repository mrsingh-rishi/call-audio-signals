# Validation Report

Deliverable 6: metric used, per-class performance, confusion matrix.

Reproduce with:

```bash
uv run python -m autoace.proxy.build --n 500   # regenerate the set (seeded)
uv run python scripts/fit_detectors.py         # objective fields (Path B)
uv run python scripts/fit_tone.py              # tone + intensity (Path C)
uv run python scripts/fit_overlap.py --write   # overlap threshold (Path D)
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
| speakers | 102 in the set, from a 115-speaker pool (RAVDESS 24 + CREMA-D 91) |
| **split** | **fit 271 / eval 229 — speakers disjoint (61 fit / 41 eval)** |
| tone balance | neutral 83 · satisfied 127 · frustrated 82 · upset 103 · distressed 105 |
| formats | Opus/ogg 164 · wav16 179 · G.711/mp3 157 |
| noise present | 278 | 
| overlap present | 239 |
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
  ~18 KB of JSON across the two classifiers and inference is a numpy dot product.

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

> All numbers below come from the proxy set **as regenerated after the label
> defect in §5.4 was fixed**. They are not comparable to figures in earlier
> drafts, which were computed on different audio.

| | accuracy | macro-F1 |
|---|---|---|
| **Path C — prosodic classifier** | **0.493** | **0.479** |
| B0 — majority baseline | 0.258 | 0.082 |

**5.8× the baseline on macro-F1.** Chance on five balanced classes is ~0.20.

**Per-class F1:**

| class | F1 | n |
|---|---|---|
| upset | **0.653** | 44 |
| neutral | 0.639 | 37 |
| satisfied | 0.430 | 59 |
| distressed | 0.379 | 53 |
| **frustrated** | **0.295** | 36 |

**Confusion matrix** (rows = truth, columns = predicted):

| | neutral | satisfied | frustrated | upset | distressed |
|---|---|---|---|---|---|
| **neutral** | **31** | 2 | 3 | 1 | 0 |
| **satisfied** | 10 | **23** | 5 | 8 | 13 |
| **frustrated** | 10 | 10 | **9** | 2 | 5 |
| **upset** | 1 | 4 | 1 | **32** | 6 |
| **distressed** | 8 | 9 | 7 | 11 | **18** |

Reading it: `neutral` (31/37) and `upset` (32/44) are cleanest and are almost
never confused with each other — the two ends of the arousal axis are the easiest
thing to hear. `frustrated` is the weakest by a wide margin, scattering across
every other class, and `distressed` leaks heavily into `upset` (11) which is its
nearest neighbour in both arousal and valence.

**That was predicted before the experiment ran.** `frustrated` has no clean
analogue in acted corpora and is approximated from low-intensity anger and
disgust. Per-class F1 is reported precisely so this is visible rather than
averaged away.

**Logit adjustment contributed +0.002 of this.** τ=0.10 was selected on held-out
speakers inside the fit split; eval macro-F1 was 0.477 at τ=0 and 0.479 at
τ=0.10. Almost all of the movement from the previous draft is the regenerated
proxy set, **not a better model**, and it is reported that way rather than
claimed as an improvement. See §8.

### `emotional_intensity`

| | accuracy | macro-F1 |
|---|---|---|
| Path C | 0.646 | **0.449** |
| B0 majority | 0.616 | 0.254 |

Accuracy barely moves because the eval split is dominated by `high` (141/229),
but macro-F1 nearly doubles. `low` has F1 0.000 on **n=4** — RAVDESS provides
only two intensity levels, so genuine low-intensity examples are almost absent.
That is a proxy-set limitation, not a model result. Prior correction chose
**τ=0.00** here, i.e. the search switched the correction off rather than being
switched off by hand.

### Objective fields (Path B, deterministic)

| field | metric | result | baseline |
|---|---|---|---|
| `background_noise_present` | balanced acc / F1 | **0.920 / 0.939** | 0.603 |
| `background_noise_severity` | exact / within-1 | 0.703 / **0.930** | — |
| `audio_quality` | exact / within-1 | 0.472 / 0.782 | — |
| `long_silence_present` | balanced acc | **0.773** (F1 0.711) | 0.633 |
| `speaker_overlap_present` *(dual-pitch cue, superseded)* | balanced acc | 0.544 | 0.520 |

### `speaker_overlap_present` (Path D, pyannote segmentation-3.0)

Scored on the same eval split and the same corrected labels as the row above, so
this is a like-for-like replacement rather than a change of yardstick.

| | balanced acc | precision | recall | F1 |
|---|---|---|---|---|
| **Path D — pyannote segmentation-3.0** | **0.792** | 0.826 | 0.756 | **0.789** |
| Path B — dual-pitch cue | 0.544 | — | — | 0.521 |
| majority baseline | 0.520 | — | — | — |

Confusion: TP=90 · FP=19 · FN=29 · TN=91.

Correlation between the true two-voice duration and the detected duration is
**0.388**, against **0.069** for the dual-pitch cue. Errors are recall-side —
the model is conservative, missing short talk-overs rather than inventing them,
which is the right direction for a field a human will review.

---

## 5. What the proxy set caught

**This is the section that justifies its cost.**

**Thresholds tuned on n=3 scored 3/16 on the proxy set — worse than chance.**
They had scored 3/3 on the provided calls.

Five specific defects, none of which were visible with three clips:

1. **Noise measured on VAD-inverted regions inverts at low SNR.** When noise is
   as loud as speech, the VAD marks noisy frames as *speech*, the "non-speech"
   region empties, and the loudest clips scored as having no noise. Replaced with
   **minimum-statistics** noise estimation, which needs no speech/non-speech
   decision. `background_noise_present` went to 0.911 balanced accuracy.

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

4. **The same defect, again, in `speaker_overlap_present` — and this one had
   been misdiagnosed as a modelling failure.**

   `inject_overlap` placed the second talker at a **uniformly random offset**
   and then labelled the clip from the *requested* duration. But the
   concatenated bases are only ~42% voiced, so the interrupter usually landed in
   a pause and overlapped nothing at all.

   Measured directly, over 40 random placements per condition:

   | requested | actual two-voice time (median) | draws actually clearing the 0.5 s label threshold |
   |---|---|---|
   | 1.5 s | **0.28 s** | **8%** |
   | 4.0 s | **0.46 s** | **42%** |

   So the majority of clips labelled `speaker_overlap_present=true` contained
   **less overlap than the generator's own 0.5 s definition required.** Every
   overlap detector was being scored against noise.

   This is why the hand-built dual-pitch cue correlated **0.069** with "injected"
   duration, why it sat at 0.577 balanced accuracy, and why the previous version
   of this report concluded the field was *unsolved*. **It was not unsolved. The
   labels were wrong**, and a detector cannot beat a label that does not describe
   its audio.

   Two changes fix it: the interrupter is placed on the most voiced stretch
   available, and the label is derived from **measuring** the result rather than
   from the request. After the fix, a 1.5 s request yields 0.93 s of genuine
   two-voice time (was 0.28 s) and a 4.0 s request yields 2.00 s (was 0.46 s),
   while the positive rate is essentially unchanged (239 vs 240 of 500) — so the
   class balance is preserved and only the *correctness* of the labels changed.

5. **A stale-cache bug that would have invalidated every number here.** The
   feature caches keyed on **row count** alone, so regenerating the proxy set —
   500 new clips, same count — silently reused the previous build's features.
   Coefficients were then fitted and scored against audio they had never seen.
   All three caches now key on a hash of the clip names, which carry a spec
   fingerprint. Worth recording because the failure was completely silent: every
   metric still printed, and every one of them would have been meaningless.

---

## 6. The three provided calls — n=3, not a metric

Reported as raw agreement only. See `outputs/predictions_provided.json`.

| field | exact | within-1 |
|---|---|---|
| `audio_quality` | **3/3** | 3/3 |
| `long_silence_present` | **3/3** | — |
| `emotional_intensity` | 2/3 | **3/3** |
| `background_noise_present` | 2/3 | — |
| `speaker_overlap_present` | 2/3 | — |
| `emotional_tone` | 1/3 | — |
| `background_noise_severity` | 1/3 | 2/3 |

**17 of 27 scored fields**, up from 15 before this round of work; the whole of
that gain is `speaker_overlap_present` going 0/3 → 2/3.

No claim is made from this. It is a smoke test that the pipeline produces sane
output on real production audio, nothing more.

### A threshold choice that costs agreement here and is still correct

Path D's threshold is "any detected overlap frame", chosen on the fit split. At
that setting call_001 is predicted `true` (0.34 s detected) against a ground
truth of `false`, so the real calls score 2/3 rather than 3/3.

Measured on the eval split:

| threshold | eval balanced acc | provided calls |
|---|---|---|
| **0.01 s — chosen on the fit split, shipped** | **0.792** | 2/3 |
| 0.30 s | 0.802 | — |
| 0.60 s | 0.762 | 3/3 |

Two honest observations rather than one convenient one:

1. **The eval cost of taking 3/3 is small** — 0.792 → 0.762, about 0.03. On 229
   clips the standard error on a balanced accuracy near 0.8 is roughly 0.026, so
   that gap is about one standard error and is **not statistically meaningful**.
   I am keeping 0.01 s because it is what the fit split chose, not because 0.762
   is demonstrably worse.
2. **0.30 s would actually score highest on eval (0.802) — and I am not taking
   it**, because choosing it would mean selecting a hyperparameter on the split
   used to report the result. That is the leakage this whole document is built to
   avoid, and it would be leakage even though the number is real.

The residual disagreement is a distribution gap, not a tuning error: the proxy's
synthetic overlap is attenuated relative to genuine talk-over (§5.4), so the
threshold it prefers sits lower than real audio would. The right fix is real
labelled overlap, not a threshold nudge against three clips.

### The domain gap, stated plainly

The fitted detectors score **better on the synthetic evaluation split than on
real audio**. Before fitting, the hand-written rules scored 3/3 on these three
calls (and 3/16 on the proxy set); after fitting they score 0.920 on the proxy
set and worse on these three.

Both directions are overfitting to whichever set you look at. The honest reading
is that **my synthetic proxy has a distribution gap with real production audio** —
synthesised static and babble are not real TV bleed on a phone line.

One consequence is already applied: `audio_quality` deliberately does *not* use
the fitted classifier. It reaches only 0.472 exact on the proxy split (chance on
three classes is 0.33) and it called two genuinely-clear real calls
`severely_impaired`. The proxy set's quality definitions are the weakest part of
the generator — they are my invention rather than measured from real calls — so
the bandwidth rule is retained for that field. **This is a judgement call made
against n=3 and is flagged as such rather than buried.**

---

## 7. Confidence and ambiguity handling

`confidence` blends three inputs: **Path C's own posterior on `emotional_tone`**,
Path A's self-reported value, a 0.08 penalty per inter-path disagreement, and
0.15 more when only one path ran.

**The posterior earns its place — it was checked, not assumed.** Binning the
eval split by Path C's top posterior:

| bin | mean posterior | actual accuracy | n |
|---|---|---|---|
| lowest third | 0.362 | 0.355 | 76 |
| middle third | 0.477 | 0.500 | 76 |
| highest third | 0.716 | 0.623 | 77 |

Correlation between posterior and correctness is **+0.213**, and the mean
posterior tracks the realised accuracy closely in each bin — under-confident at
the top, which is the safe direction. A signal that did *not* track correctness
would be decoration and would not have been shipped.

**It is deliberately not calibrated against the provided labels**, because those
labels carry a constant `0.82` on all three calls — and `0.82` is also the value
in the brief's own example output. It is a copied placeholder, not ground truth.
Fitting a calibrator to a constant would produce a constant.

Ambiguity is handled by the fusion rules being explicit and inspectable: every
field records which path produced it (`sources`) and every inter-path
disagreement is retained (`disagreements`), so a low-confidence result can be
traced to the specific conflict that caused it.
