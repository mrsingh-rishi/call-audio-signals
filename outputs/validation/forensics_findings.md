# Forensics findings — the three provided calls

Reproduce with:

```bash
uv run python -m autoace.forensics data/provided_calls
```

All figures below are emitted by that command. Nothing here is hand-transcribed.

---

## 1. Container and channel layout

| | call_001 | call_002 | call_003 |
|---|---|---|---|
| codec / rate / channels | opus / 48 kHz / 2 | opus / 48 kHz / 2 | opus / 48 kHz / 2 |
| duration | 30.946 s | 34.967 s | 171.928 s |
| bitrate | ~130 kbps | ~130 kbps | ~130 kbps |
| encoder tag | `Encoded with GStreamer opusenc` | same | same |

Format is **`.ogg`/Opus**, not the `.wav`/`.mp3` shown in the brief's example
batch. The ingest layer accepts both plus `.m4a`/`.flac`/`.webm`.

### Duplicated mono, not dual-channel

| | call_001 | call_002 | call_003 |
|---|---|---|---|
| RMS(L−R) | −90.03 dBFS | −92.09 dBFS | −84.03 dBFS |
| max abs(L−R) | 3.13e−03 | 2.53e−03 | 4.22e−03 |
| bit-identical samples | 98.01% | 96.68% | 91.14% |
| Pearson r | 0.99999996 | 0.99999984 | 0.99999985 |

The residual is Opus mid/side quantisation applied to a mono source. **The
agent and customer are summed into one channel.** Consequences:

- `speaker_overlap_present` cannot come from cross-channel energy.
- The customer cannot be isolated by channel; role attribution has to come
  from content, not from the signal.
- Gemini merges multi-channel to mono anyway, so nothing is lost by this on
  the Path A side.

`analyse_channels()` re-tests this per file at runtime rather than assuming it,
because the hidden set may differ.

---

## 2. Wideband VoIP, not 8 kHz telephony

Mean PSD over speech-active frames, dB relative to peak:

| band (Hz) | call_001 | call_002 | call_003 |
|---|---|---|---|
| 3000–3500 | −30.71 | −37.44 | −25.92 |
| **3500–4000** | **−39.03** | **−45.31** | **−30.56** |
| **4000–5000** | **−39.20** | **−49.63** | **−30.91** |
| 6000–8000 | −51.37 | −59.12 | −42.49 |
| 10000–12000 | −57.52 | −67.95 | −47.38 |
| 12000–14000 | −88.18 | −76.96 | −51.38 |
| 16000–20000 | −93.57 | −79.00 | −58.73 |

G.711 telephony would put a 30 dB+ wall at 3.4–4 kHz. Instead the 3500–4000 and
4000–5000 bins sit within 0.2 dB of each other on call_001. Each file has its
own Opus adaptive cap (~12 kHz / ~12 kHz / 20 kHz).

**Consequence for the proxy set:** the degradation chain must terminate in
48 kHz Opus, not G.711. Container/bandwidth is carried as an explicit factor so
thresholds can be checked for portability rather than assumed.

---

## 3. No meaningful clipping in any file

| | call_001 | call_002 | call_003 |
|---|---|---|---|
| peak | +0.0354 dBFS | −4.1355 dBFS | +0.8816 dBFS |
| samples ≥ full scale | 3 / 1,485,105 | 0 | 196 / 8,252,221 |
| fraction | 0.00020% | 0% | 0.00244% |
| runs ≥3 samples @0.999 | 1 (longest 3) | 0 | 12 (longest 10 ≈ 0.2 ms) |
| **flat-top samples** | **0** | **0** | **1** |
| **true clipping** | **no** | **no** | **no** |

Peak dBFS above 0 is *not* evidence of clipping — lossy transform codecs decode
to intersample peaks above full scale routinely. Genuine hard clipping shows as
sustained flat-topped runs, and there are none.

**These files therefore say nothing about where the clipping→`audio_quality`
threshold sits.** That threshold is fitted on the proxy set alone.

### Bug found here (regression-tested)

`ffmpeg -ac 1` is **not** level-preserving: for stereo it weights each channel
by 1/√2, so identical L/R sum to **+3.01 dB**. Measured on call_001 that turned
3 full-scale samples into 2,498 and produced a false "true clipping" verdict on
call_003. Every level-dependent measurement in Path B would have been wrong.
`ingest._downmix_filter()` now uses equal weights summing to 1;
`test_downmix_preserves_level_on_duplicated_mono` guards it.

---

## 4. `long_silence_present` carries a semantic qualifier, not just a duration

Frame analysis of call_003 around 113–121 s:

| | value |
|---|---|
| gap region 113.8–120.5 s, mean level | **−68.22 dB** |
| clip noise floor (p05 of frame levels) | **−68.99 dB** |
| clip speech level (p90 of frame levels) | −13.76 dB |
| gap duration | **≈ 6.7 s** |

The gap sits *at* the clip's own noise floor — this is genuine dead air, not
low-level static, and it is bounded by speech at 113.6 s and 121.0 s.

**Ground truth for call_003 is `long_silence_present: false`.**

So a pure duration rule must have a threshold above 6.7 s, which is implausibly
high. The more likely reading is the brief's own wording:

> "an unusually long period of silence or dead air **that may indicate a
> call-flow or audio problem**"

A 6.7 s mid-conversation pause — an advisor looking up a record — is normal in a
dealership service call, not a fault. The field appears to encode *intent*, not
just duration.

**Design consequence.** The provisional plan value of ≥5.0 s would have produced
a false positive here. Instead:

1. The DSP detector is tuned **conservatively** (provisional ≥8 s at a gate
   near the clip's own noise floor, final value fitted on the proxy set's
   injected 0/2/5/12 s silences).
2. The DSP signal is treated as **evidence the LLM can override**, not as the
   authority — because "does this silence look like a problem?" is a semantic
   judgement the deterministic path cannot make.

**Limitation, stated honestly:** with n=3 we cannot distinguish "the threshold
is 8 s" from "the field is semantic". Both readings are recorded and the fusion
rule is fitted rather than asserted.

---

## 5. Label-ontology signals (n=3 — calibration, not metrics)

- **"sharp static" is `background_noise_type` while `audio_quality` stays
  `clear`** (call_003). Static/hiss is *noise*, not a quality defect — the
  inverse of the naive mapping. Encoded verbatim in `schema.ONTOLOGY_NOTES`.
- **`confidence` is 0.82 on all three, and 0.82 is also the value in the
  brief's own example output.** It is a copied placeholder, not ground truth.
  Calibration is reported for rigor, not treated as a hidden-set lever.
- **Base rates:** noise present 2/3 · overlap present 2/3 · quality `clear` 3/3
  · long silence 0/3 · intensity `medium` 2/3.
