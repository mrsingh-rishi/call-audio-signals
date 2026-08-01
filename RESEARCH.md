# Research Notes — Voice Tone & Background Noise Analysis

**Rishi Singh** · AutoAce AI technical trial · August 2026

This is my working research log for the trial. It records what I read, what I
measured, what changed my mind, and the decisions that came out of it. I have
kept the dead ends in, because two of them changed the architecture more than the
successes did.

Every claim I make about a model, a price or a licence has a link in §8. Where I
only read an abstract or a model card rather than the full paper, I say so.

---

## 1. Why I went looking

I had the system working end-to-end — hosted, batching, per-file error isolation,
cost and latency measured — but the accuracy was bad in a specific and revealing
way. All three provided calls came back with **identical labels on every field**.

My first assumption was a bug in my code. It wasn't. I verified the audio was
genuinely reaching the model (distinct SHA-256 per file, prompt token counts
scaling with duration — 1,800 / 1,928 / 6,311 — and evidence strings quoting
different real content from each call). So the model was hearing three different
calls and returning one answer.

Ablating my own system prompt fixed part of it: I had written worked examples
into the prompt that happened to describe the three calls, and the model started
reciting instead of classifying. That was mine to fix and I fixed it.

But even with the prompt neutralised, four fields stayed pinned:
`background_noise_present`, `background_noise_severity`,
`speaker_overlap_present` and `audio_quality`. And `emotional_tone` stayed wrong
in a way that looked systematic rather than noisy — the model kept reading the
*words* and ignoring the *voice*.

That last observation is what sent me to the literature.

---

## 2. The finding that reframed the whole problem

The paper that matters here is **"Do Audio LLMs Really LISTEN, or Just
Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance"**
([arXiv:2510.10444](https://arxiv.org/pdf/2510.10444)). I read this one in full.

Their method is exactly the experiment I would have wanted to run and couldn't
with three clips. They separate lexical from acoustic cues by constructing paired
conditions:

- the same words spoken with different emotional prosody
- different words spoken with matched prosody
- neutral words delivered with angry prosody, and angry words delivered neutrally

They ran this against **GPT-4o-audio, Gemini 2.5, Qwen2-Audio and Qwen3-Omni** —
Gemini 2.5 being the family I am using.

Their result: these models score significantly higher when the emotional signal
is carried in the **transcribed words** than when it is carried by the **voice**.
They behave, functionally, as transcribe-then-classify systems.

A broader survey states the same thing without hedging: general-purpose audio
LLMs *"consistently prioritize speech semantics (transcription) over subtle
paralinguistic cues… their capacity for fine-grained emotion reasoning remains
fundamentally limited."* Benchmark work puts the scale of it plainly — human
performance on speaker-trait tasks including SER sits around **87.6**, while
model scores are generally **below 65**
([AudioBench, arXiv:2406.16020](https://arxiv.org/pdf/2406.16020)).

**Why this mattered to me:** it turned an accuracy problem into an architecture
problem. My own data said the same thing — the labeller marked an explicit
Spanish obscenity as `neutral` (flat delivery) and marked a customer who was
refused throughout as `satisfied` (calm, agreeable delivery). Gemini got both
backwards, reading the words. That is not a prompt I can rewrite my way out of.
It is a property of the model class.

So I stopped trying to prompt-engineer the emotion field and started looking for
a specialist model.

---

## 3. Dimensional SER, and why it fits this schema better than categories

The consistent recommendation in the SER literature is **A/D/V — arousal,
dominance, valence** — rather than categorical emotion classes. It is described
as the main direction for the field, and it is what the MSP-Podcast challenge
baselines are built on ([Odyssey 2024 1st-place
solution](https://arxiv.org/pdf/2405.20064); [Interspeech 2025 SER
Challenge](https://lab-msp.com/MSP-Podcast_Competition/IS2025/)).

This solved a problem I already had. My original plan mapped a 9-class
categorical model onto the trial's 5 labels, and that mapping was lossy in an
obvious place: **`frustrated` has no clean acted analogue**, so it was always
going to be the weakest class.

In valence-arousal space it is not a missing category, it is just a region:

| target label | valence | arousal |
|---|---|---|
| `neutral` | mid | low |
| `satisfied` | **high** | low–mid |
| `frustrated` | low | mid |
| `upset` | low | **high** |
| `distressed` | **very low** | **very high** |

Two continuous axes plus fitted thresholds, instead of a lossy 9→5 category
collapse. It is also inspectable — if `frustrated` vs `upset` is the boundary
that hurts me, I can see exactly which threshold to move, which I cannot do with
a categorical softmax.

---

## 4. Model survey — and the licence column that decided it

I evaluated candidates on accuracy, size, and licence. **Licence turned out to be
the binding constraint**, and it eliminated the two models I would otherwise have
picked. AutoAce is a commercial product, so "research only" is disqualifying, not
a caveat.

| Model | Output | Size | Licence | Commercial use |
|---|---|---|---|---|
| [audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) | A/D/V, 0–1 | 0.2 B params (pruned 24→12 layers), 16 kHz | **CC-BY-NC-SA-4.0** | ❌ **research only** |
| [`Wav2Small`](https://arxiv.org/html/2408.13920v1) (audEERING) | A/D/V | **72 K params, 120 KB quantised ONNX** | audEERING — needs verifying | ⚠️ unverified |
| [`emotion2vec+ large`](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | 9 categories | ~300 M, 42.5 k h training | custom "model-license" | ❌ **training data restricts commercial use** |
| [`MERaLiON-SER-v1`](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) | categorical | – | custom; trained on data permitting commercial model building | ✅ likely |
| [`SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | ASR + emotion + **audio event detection** | small, very low latency | FunASR model licence | ✅ commercial permitted |

Three things came out of this table that I did not expect:

**The standard model is unusable.** audEERING's MSP-dim model is the de-facto
reference for A/V/D, has a published ONNX export
([doi:10.5281/zenodo.6221127](https://doi.org/10.5281/zenodo.6221127)), and would
have been my default. CC-BY-NC-SA-4.0 rules it out of a commercial deployment.
It is fine for a trial demo — but this is a trial *for a product*, and shipping a
research-only weight into a dealership product is exactly the kind of thing that
should be caught before it ships, not after.

**My own original plan had a licence bug in it.** I had specified `emotion2vec+`
as the open-source emotion model. It is commercially restricted because of its
training data. If I had followed my own plan without checking, I would have built
on a model AutoAce could not legally ship.

**`SenseVoiceSmall` is the pragmatic pick.** Commercially permitted, very low
inference latency, and it does **audio event detection alongside emotion** —
which also feeds `background_noise_type`, a field I am currently producing with
spectral heuristics. One model, two fields.

`MERaLiON-SER-v1` reports the strongest numbers of the open encoders I found,
beating `emotion2vec-seed` by **+4.9 / +2.3 / +7.1 / +4.3 UAR** across its
configurations and beating multimodal LLMs
([arXiv:2511.04914](https://arxiv.org/pdf/2511.04914)) — I read the abstract and
results summary, not the full paper.

**Plan:** ship SenseVoiceSmall, benchmark MERaLiON-SER-v1 against it, and use the
audEERING model *offline only* as a research-ceiling reference to quantify what
the licence constraint actually costs in accuracy. That number is worth knowing —
it tells AutoAce what a commercial audEERING licence would buy.

---

## 5. What the literature did to my validation plan

I had planned a synthetic proxy validation set built on **CREMA-D + RAVDESS**,
because the trial provides only three labelled calls and n=3 supports no
statistical claim whatsoever.

The literature is direct about the weakness of that choice:

> *"Acted prosody exaggerates cues, limiting generalization to spontaneous speech."*

and, on RAVDESS/IEMOCAP-style corpora specifically, that they *"do not adequately
address the complicated characteristics of spontaneous and authentic speech,
including background noise, speech interruptions, speaker overlap, and diverse
recording environments"* — which is a fairly precise description of the AutoAce
production audio I was handed.

I am keeping the proxy set, with a split verdict:

- **For the derived fields** (`background_noise_*`, `speaker_overlap_present`,
  `audio_quality`, `long_silence_present`) it is still sound. Ground truth there
  comes from *my own* degradation chain — I know the injected SNR, the injected
  overlap percentage, the injected silence duration — so the emotional content of
  the base clip is irrelevant.
- **For `emotional_tone` it is optimistic.** Thresholds fitted on acted speech
  will overstate real-world performance.

The right corpus for the tone axis is **MSP-Podcast** — 324 h of naturalistic
conversational speech, the corpus behind the Odyssey and Interspeech 2025
challenges ([corpus paper](https://arxiv.org/html/2509.09791v1); [challenge
baseline code](https://github.com/msplabresearch/MSP-Podcast_Challenge_IS2025)).
It requires a request form that will not clear inside this deadline.

So the honest position, which goes in the memo rather than being quietly omitted:
**tone thresholds are fitted on acted speech, this is a known optimism bias, and
the fix given more time is MSP-Podcast.**

---

## 6. Experiments I ran off the back of the reading

### 6.1 Chain-of-thought / thinking budget

Several papers report that CoT prompting substantially improves zero-shot SER for
audio LLMs, with Qwen2-Audio approaching specialist models on MELD and IEMOCAP
(see [OmniVox, arXiv:2503.21480](https://arxiv.org/pdf/2503.21480) and
[EMO-RL, arXiv:2509.15654](https://arxiv.org/pdf/2509.15654)).

I had fixed `thinking_budget=0` for cost reasons and never tested it, so I ran
the A/B:

| thinking budget | tone correct | thinking tokens | cost / audio-min | latency (172 s clip) |
|---|---|---|---|---|
| 0 | 0/3 | 0 | $0.000849 | 4.4 s |
| 512 | 0/3 | ~385 | $0.000980 | 5.9 s |
| 2048 | 0/3 | 887–1,673 | $0.001264 | 12.9 s |

No measurable benefit — but **n=3 could not detect one even if it existed**, so I
am not claiming the literature is wrong. What I did learn is that my stated reason
for setting it to 0 was wrong: cost is not the constraint (2.4× ceiling headroom
even at 2048 tokens), **latency is** — roughly 3× slower.

Decision: keep 0, re-run this on the proxy set where the result would mean
something.

### 6.2 Measuring whether the labels are recoverable at all

Before building a deterministic detector I wanted to know whether the noise and
overlap labels were physically present in the signal, or whether I was chasing
something unrecoverable.

| metric | call_001 (no noise) | call_002 (TV) | call_003 (static) |
|---|---|---|---|
| SNR | 45.5 dB | 26.0 dB | **44.9 dB** |
| **non-speech spectral flatness** | **0.060** | **0.126** | **0.217** |
| non-speech centroid | 608 Hz | 816 Hz | 1386 Hz |
| **dual-pitch frame fraction** | **0.352** | **0.516** | **0.406** |

**SNR — the obvious measure — is useless here.** The clip with audible static
sits 0.6 dB from the clip with no noise at all, because quiet static is still
static.

**Spectral flatness separates all three cleanly**, and it does so for a physical
reason rather than a fitted one: flatness is the ratio of geometric to arithmetic
mean of the power spectrum, so broadband hiss/static scores high (0.217),
structured sources like television score mid (0.126), and a near-silent line
carrying only low-frequency room tone scores low (0.060).

For overlap I used the fact that two simultaneous talkers on a summed mono mix
leave **two independent harmonic series**: autocorrelation finds the dominant
period, then I mask that lag together with its harmonics and sub-harmonics (so a
single voice's own octave is not miscounted) and re-search for a second
periodicity.

Building the deterministic path on those two measures took
`background_noise_present`, `speaker_overlap_present` and `audio_quality` from
0.33 / 0.33 / 0.00 to **1.00 / 1.00 / 1.00**, and noise severity to within-one on
all three. That is the single largest accuracy movement in the project, and it
came from measurement rather than from a bigger model.

---

## 7. Decisions

1. **The LLM is not the right tool for `emotional_tone` on its own.** This is a
   documented property of audio LLMs, confirmed on my own data. Add a specialist
   dimensional SER model.
2. **Keep Gemini for what it is measurably good at.** Semantic customer
   identification on a summed mono mix succeeded 3/3 — that is genuinely hard and
   the acoustic route (F0 clustering) gave me only ~2.5σ separation. Use the LLM
   there, use specialists elsewhere.
3. **Deterministic DSP owns the objective fields.** Not as a guard rail, as the
   primary source — Gemini returned `false` for speaker overlap on every clip
   under every prompt variant I tried, including one that explicitly asked about
   interruptions.
4. **Licence is an architecture constraint.** It eliminated my two default model
   choices, including the one written into my own original plan.
5. **Say the acted-corpus bias out loud** rather than letting a proxy-set F1 imply
   more than it should.

---

## 8. Sources

### Read in full

- **Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs.
  Acoustic Emotion Cues Reliance** — https://arxiv.org/pdf/2510.10444
- **Gemini API — audio understanding** (32 tokens/s tokenisation, 16 kbps
  downsampling, multi-channel merged to mono) —
  https://ai.google.dev/gemini-api/docs/audio
- **Gemini API — pricing** — https://ai.google.dev/gemini-api/docs/pricing
- **Gemini API — context caching** (minimum-token thresholds, implicit vs
  explicit) — https://ai.google.dev/gemini-api/docs/caching
- **audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` model card** —
  https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
- **TorchAudio pipelines / SQUIM** —
  https://docs.pytorch.org/audio/stable/pipelines.html

### Read abstract, results or model card

- **MERaLiON-SER: Robust Speech Emotion Recognition for English and SEA
  Languages** — https://arxiv.org/pdf/2511.04914 ·
  https://huggingface.co/MERaLiON/MERaLiON-SER-v1
- **Wav2Small: Distilling Wav2Vec2 to 72K parameters for Low-Resource SER** —
  https://arxiv.org/html/2408.13920v1 ·
  https://www.audeering.com/publications/distilling-wav2vec2-to-72k-paramters/
- **1st Place Solution to Odyssey Emotion Recognition Challenge Task 1: Tackling
  Class Imbalance** — https://arxiv.org/pdf/2405.20064
- **The Interspeech 2025 Challenge on SER in Naturalistic Conditions** —
  https://lab-msp.com/MSP-Podcast_Competition/IS2025/ ·
  https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html
- **The MSP-Podcast Corpus** — https://arxiv.org/html/2509.09791v1 ·
  https://github.com/msplabresearch/MSP-Podcast_Challenge_IS2025
- **AudioBench: A Universal Benchmark for Audio Large Language Models** —
  https://arxiv.org/pdf/2406.16020
- **OmniVox: Zero-Shot Emotion Recognition with Omni-LLMs** —
  https://arxiv.org/pdf/2503.21480
- **EMO-RL: Emotion-Rule-Based RL Enhanced Audio-Language Model for Generalized
  SER** — https://arxiv.org/pdf/2509.15654
- **Multimodal LLMs Meet Multimodal Emotion Recognition and Reasoning: A Survey**
  — https://arxiv.org/pdf/2509.24322
- **Kimi-Audio Technical Report** — https://arxiv.org/pdf/2504.18425 ·
  https://github.com/MoonshotAI/Kimi-Audio
- **emotion2vec (ACL 2024)** — https://github.com/ddlBoJack/emotion2vec ·
  https://huggingface.co/emotion2vec/emotion2vec_plus_large
- **SenseVoice** — https://huggingface.co/FunAudioLLM/SenseVoiceSmall ·
  https://github.com/QwenAudio/SenseVoice
- **pyannote `speaker-diarization-community-1`** —
  https://huggingface.co/pyannote/speaker-diarization-community-1 ·
  https://www.pyannote.ai/blog/community-1
- **Enhancing SER with Graph-Based Multimodal Fusion and Prosodic Features
  (Interspeech 2025)** — https://arxiv.org/abs/2506.02088
- **EmoNet-Voice: A Large-Scale Synthetic Benchmark for Fine-Grained Speech
  Emotion** — https://arxiv.org/html/2506.09827
- **A SER Model Combining WavLM Pre-Trained Features and Attention Mechanism
  (2026)** — https://www.mdpi.com/2079-9292/15/13/2855
- **Update on TorchAudio's future** (maintenance phase, I/O moved to TorchCodec) —
  https://github.com/pytorch/audio/issues/3902
- **Render pricing** (plan tiers, spin-down behaviour) — https://render.com/pricing

### Referenced, not retrieved

- **Kim & Stern, "Robust Signal-to-Noise Ratio Estimation Based on Waveform
  Amplitude Distribution Analysis" (WADA-SNR), Interspeech 2008** — the standard
  reference for the SNR estimator; I ended up not needing it, since spectral
  flatness outperformed SNR on this data.
- **Wagner et al., "Dawn of the Transformer Era in Speech Emotion Recognition"**,
  arXiv:2203.07378 — the paper behind the audEERING dimensional model card.
