"""
Helpers shared by the scripts: events <-> music, saving MIDI and rendering audio.

One musical "event" is a triple [soprano, bass, duration]:
    [67, 48, 1.0]  -> melody note 67 (G4) over bass note 48 (C3), held for 1 beat
    [0, 0, 0.5]    -> a rest of half a beat
(MIDI numbers: 60 is middle C.)
"""
import wave

import numpy as np
from music21 import chord, note, stream, tempo


def tokens_to_events(tokens, bpm=90):
    """List of {start, dur, pitches} in seconds. Used for the audio and for the piano roll."""
    seconds_per_beat = 60.0 / bpm
    events, time = [], 0.0
    for soprano, bass, beats in tokens:
        dur = beats * seconds_per_beat
        pitches = [] if soprano == 0 else sorted({int(bass), int(soprano)})
        events.append({"start": round(time, 3), "dur": round(dur, 3), "pitches": pitches})
        time += dur
    return events


def save_midi(tokens, path, bpm=90):
    """Convert events into a music21 stream and write a .mid file."""
    music = stream.Stream()
    music.append(tempo.MetronomeMark(number=bpm))
    for soprano, bass, beats in tokens:
        if soprano == 0:
            element = note.Rest()
        elif soprano == bass:
            element = note.Note(int(soprano))
        else:
            element = chord.Chord([int(bass), int(soprano)])
        element.quarterLength = beats
        music.append(element)
    music.write("midi", fp=str(path))


def render_wav(tokens, path, bpm=90, sample_rate=22050):
    """A tiny synthesizer: adds sine waves for every note and saves a .wav file."""
    events = tokens_to_events(tokens, bpm)
    total = events[-1]["start"] + events[-1]["dur"] if events else 0.0
    audio = np.zeros(int((total + 0.5) * sample_rate), dtype=np.float32)

    for event in events:
        if not event["pitches"]:
            continue
        length = int((event["dur"] + 0.12) * sample_rate)       # a short tail so notes ring out
        t = np.arange(length) / sample_rate
        envelope = np.minimum(t / 0.015, 1.0)                   # quick attack
        envelope *= 0.65 + 0.35 * np.exp(-t * 6.0)              # decay to a steady level
        envelope *= np.clip((event["dur"] + 0.12 - t) / 0.12, 0.0, 1.0)  # release
        start = int(event["start"] * sample_rate)
        gain = 0.3 / np.sqrt(len(event["pitches"]))
        for p in event["pitches"]:
            freq = 440.0 * 2 ** ((p - 69) / 12.0)
            tone = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(4 * np.pi * freq * t) + 0.15 * np.sin(6 * np.pi * freq * t)
            end = min(start + length, len(audio))
            audio[start:end] += (gain * envelope * tone)[: end - start]

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())
