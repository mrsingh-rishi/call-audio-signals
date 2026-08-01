# Cost Analysis

Deliverable 7. Every figure is computed from **live `usageMetadata`** returned by
the API, not from an estimate. Reproduce with `uv run python scripts/predict_provided.py`.

---

## 1. Headline

| | |
|---|---|
| **Measured cost** | **$0.000849 per audio-minute** |
| Ceiling | $0.003 per audio-minute |
| **Headroom** | **3.5×** |

Measured across the three provided calls (3.96 audio-minutes total, $0.003366).

---

## 2. Why the quoted number is the *uncached* one

The first version of this analysis reported **$0.000243/audio-minute**. That
number was real but misleading, and finding out why changed how cost is reported.

Gemini's **implicit caching** was hitting: 1,603 of 1,898 prompt tokens on
call_001, 5,639 of 6,409 on call_003 — 84–88% cached, billed at $0.03/M instead
of $0.30/M. The cause was my own testing: re-running the *same three files*
repeatedly caches the audio itself.

In production over distinct files, only the ~900-token system instruction is a
stable prefix. The audio never repeats. So the cache-assisted figure would not
survive contact with a real batch.

`CallUsage.cost_uncached_usd()` computes cost as if nothing were cached, and the
ceiling assertion in `predict.py` tests **the worse of the two**. Quoting the
cached number would have overstated the margin by 3.5×.

| clip | cached | uncached | cache hit |
|---|---|---|---|
| call_001 | $0.000224 | $0.000657 | 1,603 / 1,898 |
| call_002 | $0.000251 | $0.000697 | 1,653 / 2,026 |
| call_003 | $0.000490 | $0.002013 | 5,639 / 6,409 |
| **per audio-min** | $0.000243 | **$0.000849** | |

---

## 3. Assumptions, stated explicitly

1. **Audio tokenises at 32 tokens/second = 1,920 tokens per audio-minute.**
   Confirmed empirically, not taken from documentation: measured 34.2 / 33.9 /
   32.4 tok/s across the three calls, converging on 32 as the fixed ~180-token
   system-instruction overhead amortises over longer clips.
2. **One Gemini call per clip.** The architecture is single-pass; the three-head
   variant was rejected on cost (below).
3. **`gemini-2.5-flash-lite`**, audio input $0.30/M, text output $0.40/M, rates
   verified 2026-08-01.
4. **Thinking disabled** (`thinking_budget=0`). Thinking tokens bill at *output*
   rates and are the usual way a per-minute ceiling is breached silently; they
   are logged separately and counted in the cost function.
5. **Paths B and C cost $0** — pure numpy/ffmpeg, no API, no model weights.
6. Output is ~220 tokens/clip measured, against a 350-token budget used for
   planning.

---

## 4. Why single-pass, with the arithmetic

The brief's anti-conflation warning argues for separate heads per field group.
The cost model ruled it out on the forward-path model:

| variant | audio in | output | total /audio-min | vs ceiling |
|---|---|---|---|---|
| 2.5-flash-lite, 1 pass | $0.000576 | $0.000060 | **$0.000636** | ✅ 4.7× |
| 2.5-flash-lite, 3 heads | $0.001728 | $0.000180 | $0.001908 | ✅ 1.57× |
| 3.1-flash-lite, 1 pass | $0.000960 | $0.000225 | $0.001185 | ✅ 2.5× |
| **3.1-flash-lite, 3 heads** | $0.002880 | $0.000675 | **$0.003555** | ❌ **breach** |
| 3.1-flash-lite, 3 heads, batch | $0.001440 | $0.000338 | $0.001778 | ✅ |

Three heads fit *today* on 2.5-flash-lite, which **retires 2026-10-16**, and
breach on its successor at interactive latency. Rather than build an architecture
with a known expiry date, field independence was moved out of the LLM entirely:
Paths B and C are structurally incapable of conflating fields because they never
see transcript or emotion. That is stronger than a prompt instruction and it is
free.

## 5. Context caching — measured, not assumed

Documented minimums conflict across Google's own sources (2.5 Flash quoted as
both 1,024 and 2,048 tokens; Vertex docs say 2,048 for everything), and there is
a developer report of Flash-Lite returning `cached_content_token_count = 0`
despite meeting the documented threshold.

At 32 tok/s, a 2,048-token minimum is **64 seconds of audio** — two of the three
provided calls are shorter than that. So caching cannot be relied on for typical
call lengths. It is measured per call and reported, never assumed.

---

## 6. At dealership scale

10,000 calls/month at a 6-minute average = 60,000 audio-minutes:

| | monthly |
|---|---|
| Gemini (uncached, interactive) | **$50.94** |
| Gemini via Batch API (−50%) | $25.47 |
| Paths B + C | $0.00 |
| Hosting (Render starter) | $7.00 |
| **Total, interactive** | **≈ $58/month** |

For comparison the ceiling permits $180/month at that volume.

**`LOCAL_ONLY=true` runs Paths B and C alone at $0 marginal inference cost** with
zero data egress. It loses `emotional_tone` quality — which is Path C, so
actually it does not: the local mode retains tone, intensity, noise, quality,
overlap and silence. It loses only Gemini's semantic customer identification and
its noise *naming*.
