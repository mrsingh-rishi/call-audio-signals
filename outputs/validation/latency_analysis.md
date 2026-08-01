# Latency Analysis

Deliverable 8. All figures measured, not modelled.

---

## 1. Headline

| environment | mean per clip | notes |
|---|---|---|
| Local, direct | **4.1–5.3 s** | 8-core M3, concurrency 4 |
| Local Docker container | **4.4 s** | same host |
| **Deployed (Render free, 0.1 CPU)** | **40.5 s** | concurrency 2 — see below |

Measured on the three provided calls (31 s, 35 s, 172 s): 32.9 s, 33.0 s, 55.5 s.

### Why the deployed figure is 7x the local one

Two changes made to stop the container being OOM-killed, compounded by the free
plan's CPU allocation:

1. **Concurrency lowered 4 -> 2** for memory headroom, halving throughput.
2. **Paths B and C now do real CPU work on every file** — chunked feature
   extraction re-invokes ffmpeg per 60-second chunk, and the prosody path runs a
   batched FFT over every frame. Locally that is ~0.5–1.5 s; the analysis is
   CPU-bound, so it scales directly with available CPU.
3. **Render's free plan allocates 0.1 CPU.** Starter allocates 0.5 CPU — **5x** —
   so the upgrade that fixes cold start also cuts this figure substantially. That
   is now the primary argument for it, ahead of spin-down.

This is an honest regression: the earlier 5.9 s figure was measured when Path C
did not exist and Path B was doing far less work. Correctness came first; the
remedy for the latency is a plan change rather than a code change.

---

## 2. Where the time goes

| stage | time | notes |
|---|---|---|
| ffprobe + decode | ~0.1–0.4 s | scales with duration |
| Path B (acoustic) | **~0.5 s** | 19 features, pure numpy |
| Path C (prosody) | ~0.3–1.0 s | 26 features, vectorised F0 |
| Path A (Gemini round-trip) | **3–6 s** | dominant; network-bound |
| fusion | <1 ms | |

**Latency is dominated by the Gemini round-trip.** Paths B and C together add
under a second and are CPU-bound, so they parallelise across cores while Path A
is waiting on the network.

### A note on Path B's cost

An earlier version of the overlap feature computed dual-pitch autocorrelation
over sliding 2-second windows. That took **~12 s per clip** — three times the
Gemini round-trip — for a feature whose correlation with injected overlap was
0.069. It was removed. Feature extraction is now 0.51 s/clip.

The F0 tracker in Path C was written FFT-vectorised from the start rather than
as a per-frame Python loop, which is roughly 20× faster and is why the prosody
path is affordable in the hot path at all.

---

## 3. Batch throughput

Concurrency is bounded by an asyncio semaphore (`MAX_CONCURRENCY`, default 4).
Since the workload is I/O-bound on the API rather than CPU-bound, concurrency
scales close to linearly until rate limits bind.

Measured: a 5-file batch (238 audio-seconds, including two deliberately
malformed files) completed in **~26 s wall** on the deployed instance.

Projected at dealership scale: 10,000 calls/month at 6 minutes average ≈ 60,000
audio-minutes. At concurrency 4 and ~1.5 s per audio-minute that is roughly
**6.3 hours of wall time per month**, or ~13 minutes per working day — trivially
absorbed by an overnight batch window.

For bulk scoring the **Batch API** halves cost at the price of latency
(up to 24 h turnaround), which is the right trade for retrospective analysis and
the wrong one for anything interactive. The recommendation is two-tier: batch for
bulk, single-pass interactive for anything a human is waiting on.

---

## 4. Cold start — the number that matters for the reviewer

The deployed service is currently on Render's **free** plan, which spins down
after 15 minutes of inactivity.

| condition | time to first byte |
|---|---|
| warm | **0.44 s** |
| **cold** | **31.9 s** |

**This is a submission risk, not a technical one.** The brief requires the
deployment to "remain available through the evaluation period", and a reviewer
clicking a cold link waits half a minute. The fix is one line — `plan: free` →
`plan: starter` in `render.yaml`, $7/month — and it requires a payment method on
the Render account, which the API rejected with HTTP 402 at deploy time.

Starter also raises the CPU allocation from **0.1 to 0.5 CPU**, which addresses
the per-clip latency above at the same time.

Container memory **was** the binding constraint and is now controlled. The
deployed service was OOM-killed at the 512 MiB limit while processing the
172-second clip; the failure reproduced locally at **637 MB peak**. Three fixes —
deriving all acoustic features from a single pass instead of decoding the clip
three times, float32 frame matrices, and batching the F0 FFT — brought that to
**286 MB**, and chunked extraction now bounds memory by chunk rather than by clip
length, which is what makes a 45-minute recording survivable.

The architecture avoiding the 660 MB–1.1 GB speech-emotion models is what leaves
any headroom at all; the fitted classifiers ship as **16 KB of JSON**.

---

## 5. Long calls

Production dealership calls run 3–15 minutes against the 31 s–172 s provided
clips. Cost per audio-minute is flat in duration (audio tokens are linear), but
**latency is not a concern either**: Gemini processes a 172 s clip in ~6 s, so a
10-minute call should land around 15–25 s.

Windowing for long calls is designed but not shipped — see the limitations
section of the memo. The measured design constraint is that a compact per-window
schema is required: 350 output tokens per window drops ceiling headroom to 1.40×,
while a 60-token schema keeps it at 1.67×.
