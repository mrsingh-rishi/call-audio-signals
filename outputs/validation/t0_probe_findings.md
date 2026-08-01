# T0 gate — customer-identification probe

Reproduce with:

```bash
uv run python scripts/probe_customer_id.py
```

Model `gemini-2.5-flash-lite`, `thinking_budget=0`, `temperature=0`.
Total probe cost for all three calls: **$0.0034**.

---

## 1. Gate result: PASS, 3/3

| | call_001 | call_002 | call_003 |
|---|---|---|---|
| speakers detected | 2 | 2 | 2 |
| first speaker | agent | agent | agent |
| customer correctly identified | ✅ | ✅ | ✅ |
| customer speech fraction | 0.50 | 0.23 | 0.70 |
| latency | 4.9 s | 3.7 s | 9.4 s |

**The semantic-identification bet holds.** On a summed mono mix where F0
clustering gave only ~2.5σ separation, the model identifies the customer from
content in all three calls, and gives a defensible reason each time ("the person
calling to schedule a car service appointment", "questioning of the agent's
authenticity").

**The agent is AutoAce's own AI voice employee** — "Erica" from Toyota of
Braintree / Lexington Toyota, with an identical scripted opener in all three
calls, and in call_003 an explicit "transferring you to an advisor now". So the
agent side is synthetic speech and the customer side is a real human.

**The scripted-opener fallback is corroborated:** the agent speaks first in 3/3,
so `first_speaker == agent` is a sound tie-break if semantic ID ever fails.

---

## 2. Timestamps are unreliable — do not slice audio with them

Two distinct failures, one after the other:

**(a) Minutes.seconds encoding.** With the schema declaring `NUMBER` seconds and
the instruction saying "in seconds", the model emitted `2.50` meaning *2 min 50 s*:

| | duration | raw max `end_s` | as M.SS | ratio |
|---|---|---|---|---|
| call_001 | 30.95 s | 0.30 | 30 s | 0.97 |
| call_002 | 34.97 s | 0.26 | 26 s | 0.74 |
| call_003 | 171.93 s | 2.50 | 170 s | 0.99 |

Read literally, a 171.9 s call appears to end at 2.5 s. This would have silently
broken windowing and customer-turn extraction.

**(b) Overshoot after the fix.** Adding an explicit duration anchor to the prompt
plus a "total elapsed seconds, not minutes.seconds" instruction fixed call_001
and call_002, but call_003 then ran to **250.7 s for a 171.9 s clip — 46% over**.

`transcript.repair_timestamps()` detects both cases against the known duration
and repairs or clamps, always surfacing a note so a silent repair never goes
unnoticed.

**Architectural consequence.** The plan floated re-analysing customer-only audio
by slicing on pass-1 timestamps. **That is not viable at this timestamp quality.**
Instead:

1. Pass the whole clip and let the model attribute tone semantically — which it
   demonstrably can do (3/3 above).
2. If customer-only scoring is wanted later, derive boundaries from Path B's VAD
   (acoustically accurate) and use the transcript only for turn *order* and
   *role*, aligning the k-th speech segment to the k-th turn. Grounded in real
   acoustics rather than model-reported times.

---

## 3. Audio tokenisation confirmed empirically

| | duration | prompt tokens | implied tok/s |
|---|---|---|---|
| call_001 | 30.9 s | 1,237 | 34.2 |
| call_002 | 35.0 s | 1,365 | 33.9 |
| call_003 | 171.9 s | 5,750 | 32.4 |

(Implied rate subtracts ~180 tokens of system instruction; it converges on 32
as the clip lengthens and the fixed overhead amortises.)

**The documented 32 tokens/second = 1,920 tokens per audio-minute is confirmed
against live `usageMetadata`**, so the cost model rests on measurement rather
than documentation.

---

## 4. The most important finding: labels track delivery, not content

Cross-referencing the transcripts against ground truth:

| call | what the customer says | ground-truth tone |
|---|---|---|
| call_001 | "Are you a real person?", then "Hello. Hello. Hello. Hello, hello." | **upset / high** |
| call_002 | "Spanish please." → *"Español, mamahuevo."* (a vulgar Spanish insult) | **neutral / medium** |
| call_003 | Repeatedly told the dealership is closed and cannot take the car; never gets what they asked for | **satisfied / medium** |

Two of these invert what a text-sentiment reading would produce:

- **call_002** contains an explicit obscenity aimed at the agent, and is labelled
  **neutral**.
- **call_003** is a sustained sequence of denials with the customer's request
  refused, and is labelled **satisfied**.

The consistent explanation is that **the labels track vocal delivery — prosody,
pace, volume contour — rather than lexical sentiment or conversational outcome.**
The call_003 customer is agreeable and calm throughout ("Okay, yes, I do. Yeah.")
even while being refused; the call_002 customer apparently delivers the insult
flatly.

### Consequences

1. **B1 (transcript-only) should perform poorly on tone.** That makes the
   baseline genuinely informative rather than a formality: if the audio path
   beats B1 by a wide margin, the whole audio architecture is justified by a
   measured number.
2. **Prompting must emphasise delivery over content.** An instruction to weigh
   *how* something is said above *what* is said, with an explicit warning not to
   read tone off profanity or off whether the customer got what they wanted.
3. **These are exactly the adversarial cases the brief warns about**, arriving
   from the opposite direction to the one anticipated: not "loud but satisfied",
   but *hostile words delivered neutrally* and *a bad outcome received calmly*.
   The proxy set's adversarial cells should include a lexical-vs-prosodic
   conflict axis, which the plan did not have.

**Caveat, stated plainly:** this is an inference from n=3, and one alternative
reading is simply that the labeller did not parse the Spanish in call_002. The
prosody-over-content hypothesis is recorded as a hypothesis and tested on the
proxy set, not assumed.
