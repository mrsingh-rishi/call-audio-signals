# Latency Analysis

Deliverable 8. All figures measured, not modelled.

---

## 1. Headline

| environment | mean per clip | per audio-minute |
|---|---|---|
| Local Docker container | **4.4 s** | ~1.1 s |
| Deployed (Render free, 0.1 CPU) | **5.9 s** | ~1.5 s |
| Local, direct | 4.1–5.3 s | ~1.0 s |

Measured on the three provided calls (31 s, 35 s, 172 s) at concurrency 4.

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
`plan: starter` in `render.yaml`, $7/month, same 512 MB — and it requires a
payment method on the Render account, which the API rejected with HTTP 402 at
deploy time.

Container memory is not the constraint: **peak RSS is 119 MiB** processing a
5-file batch including a 172-second clip, against a 512 MB limit — roughly 4×
headroom. That is only true because the architecture avoided the 660 MB–1.1 GB
speech-emotion models; the fitted classifiers ship as **16 KB of JSON**.

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
