"""
Step 1 - collect music and turn it into note sequences (music21).

Every chord is reduced to its highest note (the melody) and its lowest note (the bass), plus its duration.

By default this uses the Bach chorales that come with music21, so no download is needed.
To use your own music instead (jazz, classical...), put .mid files in a folder and run:
    python preprocess.py --midi-dir my_midi_files
"""
import argparse
import json
from pathlib import Path

from music21 import converter, corpus, interval, pitch

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "sequences.json"


def quantize(beats):
    """Round durations to quarter-beats between 0.25 and 4 so we get fewer different tokens."""
    return min(max(round(float(beats) * 4) / 4, 0.25), 4.0)


def piece_to_tokens(score):
    """Turn one piece into a list of [soprano, bass, duration] events."""
    try:  # move every piece to C major / A minor so the model sees one consistent key
        key = score.analyze("key")
        target = pitch.Pitch("C" if key.mode == "major" else "A")
        score = score.transpose(interval.Interval(key.tonic, target))
    except Exception:
        pass
    tokens = []
    for element in score.chordify().recurse().notesAndRests:   # chordify merges all voices into chords
        duration = quantize(element.quarterLength)
        if element.isRest:
            tokens.append([0, 0, duration])
        else:
            pitches = [p.midi for p in element.pitches]
            tokens.append([max(pitches), min(pitches), duration])   # highest note = melody, lowest = bass
    return tokens


def load_scores(max_files, midi_dir=None):
    if midi_dir:
        files = sorted(p for p in Path(midi_dir).rglob("*") if p.suffix.lower() in (".mid", ".midi"))[:max_files]
        source = f"MIDI files in {midi_dir}"
        for path in files:
            try:
                yield source, converter.parse(str(path))
            except Exception as error:
                print("  skipped", path.name, "-", error)
    else:
        paths = [p for p in corpus.getComposer("bach") if str(p).endswith((".mxl", ".xml"))][:max_files]
        for path in paths:
            yield "Bach chorales (music21 corpus)", corpus.parse(path)


def build_dataset(max_files=400, midi_dir=None):
    pieces, source = [], ""
    for i, (source, score) in enumerate(load_scores(max_files, midi_dir), start=1):
        tokens = piece_to_tokens(score)
        if len(tokens) > 40:
            pieces.append(tokens)
        if i % 10 == 0:
            print(f"  processed {i} pieces...")
    if not pieces:
        raise SystemExit("No music found. Check the MIDI folder.")
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"source": source, "pieces": pieces}, f)
    total = sum(len(p) for p in pieces)
    pitches = len({n for p in pieces for e in p for n in e[:2]})
    print(f"Saved {len(pieces)} pieces, {total} events, {pitches} different pitches -> {DATA_FILE}")
    print("Example:", pieces[0][:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect music and make note sequences")
    parser.add_argument("--max-files", type=int, default=400, help="how many pieces to use")
    parser.add_argument("--midi-dir", default=None, help="folder with your own .mid files")
    args = parser.parse_args()
    build_dataset(args.max_files, args.midi_dir)
