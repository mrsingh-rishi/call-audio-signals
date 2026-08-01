# Research: what model should actually produce `emotional_tone`

Path B (deterministic DSP) fixed the four objective fields. `emotional_tone`
remains 0/3 and is the only field with no working solution. This is a survey of
what the literature and the open-model ecosystem offer, and what is actually
deployable for a commercial product.

---

## 1. The failure is documented, not specific to our prompt

**"Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs.
Acoustic Emotion Cues Reliance"** ([arXiv 2510.10444](https://arxiv.org/pdf/2510.10444))
tested **GPT-4o-audio, Gemini 2.5, Qwen2-Audio and Qwen3-Omni**. Its method is
the one that matters here: it separated lexical from acoustic cues by pairing
*matched words with different emotional prosody* against *different words with
matched prosody*.

Finding: these models score markedly higher when the emotion is carried in the
**transcribed words** than when it is carried by the **voice**. They behave
largely as transcribe-then-classify systems.

A second survey puts it directly: general-purpose audio LLMs *"consistently
prioritize speech semantics (transcription) over subtle paralinguistic cues…
their capacity for fine-grained emotion reasoning remains fundamentally limited."*

**This independently confirms what our own measurement showed** — Gemini reading
"Español, mamahuevo" and returning `upset` when the delivery is flat, and reading
a polite refusal and returning `neutral` when ground truth is `satisfied`. It is
a property of the model class, not a defect in our prompt. No amount of prompt
engineering closes it.

Corroborating scale: on speaker-trait tasks including SER, **human performance is
~87.6 while audio-LLM scores sit below 65**.

## 2. The fix the literature points to: dimensional SER

A/D/V — **arousal, dominance, valence** — is described as "the main avenue for
speech emotion recognition", and the MSP-Podcast challenge baselines are built on it.

This matters because **our five target classes map onto valence-arousal far more
naturally than onto any categorical emotion set.** The 9-class categorical
mapping in the original plan was lossy precisely because `frustrated` has no
acted analogue; in A/V space it is simply *low valence, moderate arousal*:

| target label | valence | arousal |
|---|---|---|
| `neutral` | mid | low |
| `satisfied` | **high** | low–mid |
| `frustrated` | low | mid |
| `upset` | low | **high** |
| `distressed` | **very low** | **very high** |

Two axes and a set of thresholds replace a lossy 9→5 category mapping. The
thresholds are fitted on the proxy set, and — unlike a categorical model — the
mapping is inspectable and adjustable per field.

## 3. Candidate models, with the licence column that decides it

AutoAce is a commercial product, so licence is a hard filter, not a footnote.

| Model | Output | Size | Licence | Commercial? |
|---|---|---|---|---|
| **audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim`** | A/D/V, 0–1 | 0.2 B params (pruned 24→12 layers) | **CC-BY-NC-SA-4.0** | ❌ **research only** |
| **audEERING `Wav2Small`** | A/D/V | **72 K params, 120 KB quantised ONNX** | audEERING (verify) | ⚠️ verify |
| **`emotion2vec+ large`** | 9 categories | ~300 M | custom "model-license" | ❌ **training data restricts commercial use** |
| **`MERaLiON-SER-v1`** | categorical | – | custom, **trained on data permitting commercial model building** | ✅ likely |
| **`SenseVoiceSmall`** (FunAudioLLM) | ASR + emotion + **audio event detection** | small, very low latency | FunASR model licence, **commercial permitted** | ✅ |

Notes that changed my recommendation:

- **The obvious pick is blocked.** audEERING's MSP-dim model is the de-facto
  standard for A/V/D and has an ONNX export
  ([doi:10.5281/zenodo.6221127](https://doi.org/10.5281/zenodo.6221127)), but
  CC-BY-NC-SA-4.0 makes it unusable in a commercial deployment. Fine for a trial
  demo; not fine for the product it is a trial *for*. That distinction belongs in
  the memo.
- **emotion2vec+ — the plan's original choice — is also commercially
  restricted**, because of the datasets it was trained on. The original plan
  would have shipped a licence problem.
- **`MERaLiON-SER-v1` reports the best numbers** among open encoders, beating
  emotion2vec-seed by +4.9 / +2.3 / +7.1 / +4.3 UAR across its configurations,
  and beating multimodal LLMs.
- **`SenseVoiceSmall` is the pragmatic pick**: commercial-permitted, very low
  latency, and it performs **audio event detection as well as emotion** — which
  would also strengthen `background_noise_type`, currently produced by spectral
  heuristics.
- **`Wav2Small` at 120 KB quantised** is remarkable for a 512 MB container if its
  licence permits — it would add essentially nothing to the image.

## 4. Consequence for the proxy set

The literature is blunt about the plan's corpus choice:

> *"Acted prosody exaggerates cues, limiting generalization to spontaneous speech."*

and, on RAVDESS/CREMA-D specifically, that such corpora *"do not adequately
address … background noise, speech interruptions, speaker overlap, and diverse
recording environments"* — i.e. exactly the conditions in the AutoAce data.

**This weakens the planned CREMA-D + RAVDESS proxy set for the emotion axis.** It
remains fine for the *derived* fields (noise, overlap, quality, silence), where
ground truth comes from our own degradation chain and the emotional content is
irrelevant. But tone thresholds fitted on acted speech will be optimistic.

**MSP-Podcast** (324 h naturalistic conversational speech, the Interspeech 2025 /
Odyssey challenge corpus) is the right corpus for the tone axis. It needs a
request form, which the plan already flagged as unlikely to land inside the
deadline — so the honest position for the memo is: *tone thresholds are fitted on
acted speech, this is a known optimism bias, and the fix with more time is
MSP-Podcast.*

## 5. Experiment run: does chain-of-thought help?

The literature reports that CoT prompting *"significantly enhances the zero-shot
SER performance of LALMs"*, with Qwen2-Audio approaching specialist models. Our
`thinking_budget` was fixed at 0 and never tested, so this was worth checking.

| thinking budget | tone correct | thinking tokens | cost / audio-min | latency (172 s clip) |
|---|---|---|---|---|
| 0 | **0/3** | 0 | $0.000849 | 4.4 s |
| 512 | **0/3** | ~385 | $0.000980 | 5.9 s |
| 2048 | **0/3** | 887–1673 | $0.001264 | 12.9 s |

**No accuracy benefit at n=3, and n=3 cannot detect one.** Predictions do move
(`neutral`→`frustrated` at 512), and intensity edges toward the truth at 2048.
Cost stays comfortably under the ceiling at every setting (2.4× headroom even at
2048), so cost is not the reason to keep it at 0 — latency is (3× slower).

**Decision: keep `thinking_budget=0` for now, re-run this A/B on the proxy set.**
Recorded so the choice is evidence-backed rather than assumed.

## 6. Recommendation

1. **Add a dimensional SER model as a third path for `emotional_tone` only.**
   Gemini keeps semantic customer identification — which it does well, 3/3 — and
   the SER model supplies arousal/valence from the customer's audio. This mirrors
   the Path B decision: use the LLM where it is strong, a specialist where it is
   measurably blind.
2. **Evaluate `SenseVoiceSmall` first** — commercially permitted, low latency,
   and its audio-event detection also feeds `background_noise_type`. Then
   `MERaLiON-SER-v1` for accuracy comparison.
3. **Use audEERING's MSP-dim model as the research-only reference ceiling** to
   quantify how much accuracy the licence constraint costs. That number is a good
   memo line: it tells AutoAce what a commercial licence would buy.
4. **State the acted-corpus optimism bias explicitly** in the limitations section
   rather than letting a proxy-set F1 imply more than it should.
