"""
Step 3 - generate new music with the trained model.

Run:  python generate.py                        (saves generated/song.mid and generated/song.wav)
      python generate.py --length 200 --temperature 1.2 --tempo 100 --name mysong
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
from pathlib import Path

import keras
import numpy as np

from music_utils import render_wav, save_midi

MODEL_DIR = Path(__file__).parent / "model"
OUTPUT_DIR = Path(__file__).parent / "generated"


class Composer:
    def __init__(self):
        model_file = MODEL_DIR / "music_lstm.keras"
        if not model_file.exists():
            raise SystemExit("No trained model found. Run 'python train.py' first.")
        self.model = keras.models.load_model(model_file)
        with open(MODEL_DIR / "vocab.json", encoding="utf-8") as f:
            data = json.load(f)
        self.pitches = data["pitches"]
        self.durations = data["durations"]
        self.seeds = data["seeds"]
        info_file = MODEL_DIR / "info.json"
        self.info = json.load(open(info_file, encoding="utf-8")) if info_file.exists() else {}

    @staticmethod
    def _sample(probs, temperature):
        """Pick the next value. Low temperature = safe and repetitive, high = surprising."""
        logits = np.log(probs.astype("float64") + 1e-9) / max(temperature, 0.05)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return int(np.random.choice(len(weights), p=weights))

    def compose(self, length=128, temperature=1.0):
        """Start from a real 32-event snippet, then keep predicting the next event."""
        soprano, bass, duration = [list(w) for w in random.choice(self.seeds)]
        result = []
        for _ in range(length):
            out = self.model([np.array([soprano]), np.array([bass]), np.array([duration])], training=False)
            if isinstance(out, dict):
                out = [out["soprano"], out["bass"], out["duration"]]
            s_id = self._sample(out[0].numpy()[0], temperature)
            b_id = self._sample(out[1].numpy()[0], temperature)
            d_id = self._sample(out[2].numpy()[0], temperature)
            result.append([self.pitches[s_id], self.pitches[b_id], self.durations[d_id]])
            soprano = soprano[1:] + [s_id]
            bass = bass[1:] + [b_id]
            duration = duration[1:] + [d_id]
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate music")
    parser.add_argument("--length", type=int, default=128, help="number of notes/chords")
    parser.add_argument("--temperature", type=float, default=1.0, help="0.5 safe ... 1.5 adventurous")
    parser.add_argument("--tempo", type=int, default=90, help="beats per minute")
    parser.add_argument("--name", default="song")
    args = parser.parse_args()

    composer = Composer()
    tokens = composer.compose(args.length, args.temperature)
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_midi(tokens, OUTPUT_DIR / f"{args.name}.mid", args.tempo)
    render_wav(tokens, OUTPUT_DIR / f"{args.name}.wav", args.tempo)
    print("First events [melody, bass, beats]:", tokens[:4])
    print(f"Saved {OUTPUT_DIR / (args.name + '.mid')} and {OUTPUT_DIR / (args.name + '.wav')}")


if __name__ == "__main__":
    main()
