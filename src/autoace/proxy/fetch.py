"""Corpus acquisition and label mapping.

Three corpora, chosen to fit ~2 GB of disk rather than for being ideal:

* **RAVDESS** - 24 actors, 8 emotions, two explicit intensity levels. The
  intensity labels are the reason it is here: they are the only direct source of
  ``emotional_intensity`` ground truth available.
* **CREMA-D** - 91 actors, which is what makes an actor-grouped split
  meaningful. RAVDESS alone gives only 24 groups.
* **ESC-50** - 50 environmental noise classes for the noise axis.

Emotion mapping is lossy and the mapping table below is the honest record of
how. ``frustrated`` in particular has no clean acted analogue - it is
approximated from low-intensity anger and disgust - so per-class F1 is reported
rather than a single accuracy that would hide it.
"""

from __future__ import annotations

import csv
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

CORPORA = Path(__file__).resolve().parents[3] / "data" / "corpora"

# --- Emotion mapping -------------------------------------------------------
# Acted corpora label the emotion the actor was asked to portray. The trial's
# five labels describe how a customer sounds on a service call, so the mapping
# is an interpretation, not an identity.

RAVDESS_EMOTION = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
CREMA_EMOTION = {
    "NEU": "neutral", "HAP": "happy", "ANG": "angry",
    "SAD": "sad", "FEA": "fearful", "DIS": "disgust",
}


def map_tone(source_emotion: str, strong: bool) -> str | None:
    """Map a corpus emotion onto the trial's five tones.

    Returns None for emotions with no defensible mapping (``surprised``), rather
    than forcing them into a class and polluting the labels.
    """
    e = source_emotion
    if e in ("neutral", "calm"):
        return "neutral"
    if e == "happy":
        return "satisfied"
    if e == "angry":
        return "upset" if strong else "frustrated"
    if e == "disgust":
        return "frustrated"
    if e in ("sad", "fearful"):
        return "distressed" if strong else "frustrated"
    return None      # surprised: neither positive nor negative here


@dataclass(frozen=True)
class SpeechClip:
    path: Path
    speaker_id: str          # the grouping key for leakage-free splits
    tone: str
    intensity: str
    source: str


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if marker.exists():
        return dest
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.touch()
    return dest


def index_ravdess(root: Path) -> list[SpeechClip]:
    """RAVDESS filename: modality-channel-emotion-intensity-statement-rep-actor."""
    clips: list[SpeechClip] = []
    for p in sorted(root.rglob("*.wav")):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        emotion = RAVDESS_EMOTION.get(parts[2])
        if emotion is None:
            continue
        strong = parts[3] == "02"
        tone = map_tone(emotion, strong)
        if tone is None:
            continue
        clips.append(SpeechClip(
            path=p,
            speaker_id=f"ravdess_{parts[6]}",
            tone=tone,
            # RAVDESS has only two levels; 'normal' is treated as medium
            # because these are performed emotions, not conversational asides.
            intensity="high" if strong else "medium",
            source="ravdess",
        ))
    return clips


def index_cremad(root: Path) -> list[SpeechClip]:
    """CREMA-D filename: ActorID_Sentence_Emotion_Level.wav"""
    clips: list[SpeechClip] = []
    level_map = {"LO": "low", "MD": "medium", "HI": "high", "XX": "medium"}
    for p in sorted(root.rglob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) != 4:
            continue
        actor, _sentence, emo_code, level = parts
        emotion = CREMA_EMOTION.get(emo_code)
        if emotion is None:
            continue
        intensity = level_map.get(level, "medium")
        tone = map_tone(emotion, strong=(intensity == "high"))
        if tone is None:
            continue
        clips.append(SpeechClip(
            path=p, speaker_id=f"cremad_{actor}",
            tone=tone, intensity=intensity, source="cremad",
        ))
    return clips


# --- Noise -----------------------------------------------------------------
# ESC-50 class -> the vocabulary the trial actually uses. Classes with no
# plausible analogue on a dealership phone call are dropped.
ESC50_TO_NOISE = {
    "vacuum_cleaner": "mechanical noise", "washing_machine": "mechanical noise",
    "engine": "road noise", "car_horn": "road noise", "train": "road noise",
    "airplane": "mechanical noise", "helicopter": "mechanical noise",
    "chainsaw": "mechanical noise", "keyboard_typing": "keyboard typing",
    "mouse_click": "keyboard typing", "wind": "wind", "rain": "wind",
    "clock_tick": "office chatter", "clapping": "office chatter",
    "laughing": "office chatter", "footsteps": "office chatter",
    "crying_baby": "background chatter", "dog": "background chatter",
    "siren": "road noise",
}


@dataclass(frozen=True)
class NoiseClip:
    path: Path
    noise_class: str
    source_class: str


def index_esc50(root: Path) -> list[NoiseClip]:
    meta = next(root.rglob("esc50.csv"), None)
    if meta is None:
        return []
    audio_dir = meta.parent.parent / "audio"
    out: list[NoiseClip] = []
    with meta.open(newline="") as fh:
        for row in csv.DictReader(fh):
            mapped = ESC50_TO_NOISE.get(row["category"])
            if not mapped:
                continue
            p = audio_dir / row["filename"]
            if p.exists():
                out.append(NoiseClip(p, mapped, row["category"]))
    return out


def download_cremad_subset(dest: Path, clips_per_actor: int = 12,
                           jobs: int = 16) -> Path:
    """Fetch a per-actor sample of CREMA-D over git-lfs media URLs.

    The full corpus is 7,442 individually LFS-hosted files. A stratified subset
    keeps every actor represented - which is what the grouped split needs -
    without downloading half a gigabyte one file at a time.
    """
    dest.mkdir(parents=True, exist_ok=True)
    listing = dest / "_filelist.txt"
    if not listing.exists() or len(listing.read_text().splitlines()) < 5000:
        # The contents API caps at 1000 entries and CREMA-D has 7,442 files, so
        # a single call silently returns only the first ~13 actors. The git tree
        # API returns the whole tree in one request.
        import json as _json
        api = ("https://api.github.com/repos/CheyneyComputerScience/"
               "CREMA-D/git/trees/master?recursive=1")
        out = subprocess.run(
            ["curl", "-sL", "--max-time", "180", api], capture_output=True, text=True
        ).stdout
        names: list[str] = []
        try:
            tree = _json.loads(out).get("tree", [])
            names = [Path(e["path"]).name for e in tree
                     if e.get("path", "").startswith("AudioWAV/")
                     and e.get("path", "").endswith(".wav")]
        except Exception:
            names = []
        if names:
            listing.write_text("\n".join(names))
    names = [n for n in listing.read_text().splitlines() if n]

    by_actor: dict[str, list[str]] = {}
    for n in names:
        by_actor.setdefault(n.split("_")[0], []).append(n)
    wanted = [n for v in by_actor.values() for n in sorted(v)[:clips_per_actor]]

    todo = [n for n in wanted if not (dest / n).exists()]
    if todo:
        base = ("https://media.githubusercontent.com/media/"
                "CheyneyComputerScience/CREMA-D/master/AudioWAV/")
        cmd = (f"printf '%s\\n' {' '.join(todo)} | "
               f"xargs -P {jobs} -I{{}} curl -sL --max-time 60 -o {dest}/{{}} {base}{{}}")
        subprocess.run(["bash", "-c", cmd], capture_output=True)
    return dest
