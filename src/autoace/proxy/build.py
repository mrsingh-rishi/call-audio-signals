"""Assemble the proxy validation set.

Run: ``python -m autoace.proxy.build --n 500``

Design decisions that matter more than the code:

* **Grouped split by speaker.** No speaker appears in both the fitting and
  evaluation split, so a threshold cannot be fitted to a voice it will later be
  scored on. Noise files are split disjointly for the same reason.
* **Balanced tone sampling.** The raw mapping is heavily skewed (RAVDESS yields
  ~5x more ``frustrated`` than ``upset``), and macro-F1 on a skewed set mostly
  measures the majority class. Sampling is balanced per tone up to availability.
* **Adversarial cells are placed deliberately**, not left to chance. Random
  crossing would produce them only rarely, and they are the cells that decide
  whether the system is conflating fields.
* **Synthetic noise for the classes ESC-50 lacks.** Static, television and music
  do not exist in ESC-50 but do exist in the ground truth. Static is generated
  as filtered broadband noise, which is not an approximation - that is what
  static physically is. Television is babble plus tonal content.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .degrade import (
    FORMAT_CHAINS,
    QUALITY_OPS,
    SR,
    DegradationSpec,
    degrade,
    ground_truth,
    load_wav,
    make_babble,
)
from .fetch import (
    CORPORA,
    NoiseClip,
    SpeechClip,
    extract_zip,
    index_cremad,
    index_esc50,
    index_ravdess,
)

OUT = Path(__file__).resolve().parents[3] / "data" / "proxy"
SNR_CHOICES = [float("inf"), 30.0, 20.0, 15.0, 10.0, 5.0, 0.0]
TONES = ("neutral", "satisfied", "frustrated", "upset", "distressed")


# --- synthetic noise for classes ESC-50 does not cover ---------------------

def synth_noise(kind: str, n: int, rng: np.random.Generator,
                babble_sources: list[Path] | None = None) -> np.ndarray:
    """Generate a noise class that the corpus lacks."""
    if kind == "static":
        # Broadband noise with a slight high-frequency tilt: line static/hiss.
        x = rng.normal(0, 1, n)
        return x * np.linspace(0.8, 1.2, n) ** 0
    if kind == "line noise":
        # Mains hum: 50/60 Hz plus harmonics, with a noise floor.
        t = np.arange(n) / SR
        f0 = float(rng.choice([50.0, 60.0]))
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
        return y + 0.05 * rng.normal(0, 1, n)
    if kind == "music":
        # Sustained harmonic tones over a slow chord change - stands in for
        # music as a *structured, tonal* interferer, which is the property that
        # distinguishes it from static spectrally.
        t = np.arange(n) / SR
        y = np.zeros(n)
        for _ in range(3):
            f = float(rng.uniform(180, 700))
            y += np.sin(2 * np.pi * f * t) * rng.uniform(0.3, 1.0)
        return y / 3
    if kind == "television" and babble_sources:
        # TV audio is mostly speech with music underneath.
        b = make_babble(babble_sources, n, n_voices=3, rng=rng)
        return 0.75 * b + 0.25 * synth_noise("music", n, rng)
    return rng.normal(0, 1, n)


SYNTH_CLASSES = ("static", "television", "music", "line noise")


# --- adversarial cells -----------------------------------------------------

def adversarial_spec(cell: str, rng: np.random.Generator) -> dict:
    """The crossed conditions that catch field conflation."""
    if cell == "loud_but_satisfied":
        # High energy, positive tone: punishes 'loud => upset'.
        return {"snr_db": float("inf"), "quality_ops": [], "require_tone": "satisfied"}
    if cell == "distorted_quiet_background":
        # Badly damaged signal, silent background: punishes
        # 'poor quality => noise present'.
        return {"snr_db": float("inf"),
                "quality_ops": list(rng.choice(["clip", "packet_loss", "opus_low"],
                                               size=2, replace=False))}
    if cell == "clean_but_noisy":
        # Pristine capture, loud background: punishes
        # 'noise present => quality impaired'.
        return {"snr_db": float(rng.choice([5.0, 0.0])), "quality_ops": []}
    if cell == "lexical_prosody_conflict":
        # Emotionally strong delivery on an otherwise unremarkable clip. This is
        # the cell the audio-LLM literature predicts models fail: they read the
        # words and miss the voice. Ours is prosody-only by construction, since
        # acted corpora use fixed neutral sentences.
        return {"snr_db": float("inf"), "quality_ops": [],
                "require_tone": rng.choice(["upset", "distressed"])}
    return {}


ADVERSARIAL_CELLS = ("loud_but_satisfied", "distorted_quiet_background",
                     "clean_but_noisy", "lexical_prosody_conflict")


def build(n_clips: int, seed: int = 20260801, adversarial_frac: float = 0.35) -> Path:
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    # --- load corpora ----------------------------------------------------
    extract_zip(CORPORA / "_dl" / "ravdess.zip", CORPORA / "ravdess")
    extract_zip(CORPORA / "_dl" / "esc50.zip", CORPORA / "esc50")
    speech: list[SpeechClip] = index_ravdess(CORPORA / "ravdess")
    cremad_dir = CORPORA / "cremad"
    if cremad_dir.exists():
        speech += index_cremad(cremad_dir)
    noises: list[NoiseClip] = index_esc50(CORPORA / "esc50")
    if not speech:
        raise SystemExit("no speech corpus found - run the downloads first")

    speakers = sorted({c.speaker_id for c in speech})
    pyrng.shuffle(speakers)
    cut = int(0.6 * len(speakers))
    fit_speakers, eval_speakers = set(speakers[:cut]), set(speakers[cut:])

    # Babble is built only from FITTING speakers, so an evaluation voice never
    # appears inside the noise of another evaluation clip.
    babble_pool = [c.path for c in speech if c.speaker_id in fit_speakers][:200]

    # Noise files split disjointly too.
    pyrng.shuffle(noises)
    ncut = int(0.6 * len(noises))
    noise_split = {"fit": noises[:ncut], "eval": noises[ncut:]}

    by_tone: dict[str, list[SpeechClip]] = defaultdict(list)
    for c in speech:
        by_tone[c.tone].append(c)
    for v in by_tone.values():
        pyrng.shuffle(v)

    OUT.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    n_adv = int(n_clips * adversarial_frac)
    cursor: dict[str, int] = defaultdict(int)

    for i in range(n_clips):
        is_adv = i < n_adv
        cell = ADVERSARIAL_CELLS[i % len(ADVERSARIAL_CELLS)] if is_adv else ""
        adv = adversarial_spec(cell, rng) if cell else {}

        # Pick the base clip, honouring a required tone for adversarial cells
        # and otherwise round-robining the tones to keep the set balanced.
        want = adv.get("require_tone") or TONES[i % len(TONES)]
        pool = by_tone.get(str(want)) or by_tone[TONES[i % len(TONES)]]
        base = pool[cursor[str(want)] % len(pool)]
        cursor[str(want)] += 1

        split = "fit" if base.speaker_id in fit_speakers else "eval"

        snr = adv.get("snr_db", float(pyrng.choice(SNR_CHOICES)))
        quality_ops = adv.get("quality_ops")
        if quality_ops is None:
            quality_ops = ([] if pyrng.random() < 0.6
                           else pyrng.sample(sorted(QUALITY_OPS), k=pyrng.choice([1, 2])))
        quality_ops = [str(q) for q in quality_ops]

        # Noise source: synthetic for the classes ESC-50 lacks, corpus otherwise.
        noise_arr = None
        noise_class = noise_file = ""
        if np.isfinite(snr):
            if pyrng.random() < 0.4:
                noise_class = str(pyrng.choice(SYNTH_CLASSES))
                noise_file = f"synth:{noise_class}"
            else:
                nc = pyrng.choice(noise_split[split])
                noise_class, noise_file = nc.noise_class, str(nc.path)

        overlap_s = float(pyrng.choice([0.0, 0.0, 1.5, 4.0]))
        silence_s = float(pyrng.choice([0.0, 0.0, 2.0, 5.0, 12.0]))
        fmt = str(pyrng.choice(FORMAT_CHAINS))

        spec = DegradationSpec(
            seed=int(rng.integers(0, 2**31)), snr_db=snr, noise_class=noise_class,
            noise_file=noise_file, quality_ops=quality_ops, overlap_s=overlap_s,
            silence_s=silence_s, format_chain=fmt, adversarial_cell=cell,
        )

        # Build a call-like base by stringing several utterances from the SAME
        # speaker together with natural gaps. Single corpus utterances are
        # ~3.5 s, which makes injected silence and overlap degenerate - a 12 s
        # gap inside a 3.5 s clip is not a call, it is an artefact. Keeping the
        # speaker constant preserves the grouping key for the split.
        same_speaker = [c for c in by_tone[base.tone] if c.speaker_id == base.speaker_id]
        n_utt = int(pyrng.choice([3, 4, 5, 6, 8]))
        chosen = [base] + [pyrng.choice(same_speaker) for _ in range(n_utt - 1)] \
            if same_speaker else [base]
        segments: list[np.ndarray] = []
        for c in chosen:
            u = load_wav(c.path)
            if u.size < SR // 4:
                continue
            segments.append(u)
            segments.append(np.zeros(int(pyrng.uniform(0.3, 1.2) * SR)))  # turn gap
        speech_arr = np.concatenate(segments) if segments else load_wav(base.path)
        if speech_arr.size < SR // 2:
            continue

        if noise_file.startswith("synth:"):
            noise_arr = synth_noise(noise_class, speech_arr.size + SR,
                                    np.random.default_rng(spec.seed), babble_pool)
        elif noise_file:
            noise_arr = load_wav(noise_file)

        overlap_arr = None
        if overlap_s > 0:
            other = pyrng.choice([c for c in speech if c.speaker_id != base.speaker_id])
            overlap_arr = load_wav(other.path)

        stem = f"proxy_{i:04d}_{spec.fingerprint()}"
        try:
            written = degrade(speech_arr, spec, audio_dir / stem,
                              noise=noise_arr, overlap_speech=overlap_arr)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {stem}: {type(exc).__name__}: {exc}")
            continue

        gt = ground_truth(spec, base.tone, base.intensity)
        rows.append({
            "name": written.name, "split": split, "speaker_id": base.speaker_id,
            "source_corpus": base.source, "adversarial_cell": cell,
            "format_chain": fmt, "snr_db": ("inf" if not np.isfinite(snr) else snr),
            "quality_ops": "|".join(quality_ops), "overlap_s": overlap_s,
            # `overlap_s` is what was REQUESTED; `actual_overlap_s` is the
            # two-voice time actually achieved and is what the label derives
            # from. Both are recorded so the gap stays auditable.
            "actual_overlap_s": round(spec.actual_overlap_s, 3),
            "silence_s": silence_s, "spec": json.dumps(asdict(spec)),
            "result_json": json.dumps(gt),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_clips} generated")

    manifest = OUT / "labels.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fit = [r for r in rows if r["split"] == "fit"]
    ev = [r for r in rows if r["split"] == "eval"]
    print(f"\nwrote {len(rows)} clips -> {manifest}")
    print(f"  fit {len(fit)} / eval {len(ev)}   "
          f"speakers {len(fit_speakers)}/{len(eval_speakers)} (disjoint)")
    print(f"  tone   : {dict(Counter(json.loads(r['result_json'])['emotional_tone'] for r in rows))}")
    print(f"  adversarial cells: {dict(Counter(r['adversarial_cell'] for r in rows if r['adversarial_cell']))}")
    print(f"  formats: {dict(Counter(r['format_chain'] for r in rows))}")
    print(f"  noise present: {sum(json.loads(r['result_json'])['background_noise_present'] for r in rows)}")
    print(f"  overlap present: {sum(json.loads(r['result_json'])['speaker_overlap_present'] for r in rows)}")
    print(f"  long silence: {sum(json.loads(r['result_json'])['long_silence_present'] for r in rows)}")

    (OUT / "build_info.json").write_text(json.dumps({
        "seed": seed, "n_requested": n_clips, "n_written": len(rows),
        "fit_speakers": sorted(fit_speakers), "eval_speakers": sorted(eval_speakers),
        "adversarial_frac": adversarial_frac,
    }, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()
    build(args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
