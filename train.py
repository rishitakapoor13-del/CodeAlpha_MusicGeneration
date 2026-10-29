"""
Step 2 - build and train the LSTM model.

The model reads the last 32 events (melody note, bass note, duration) and predicts the next
melody note, the next bass note and the next duration at the same time:
    Embedding -> LSTM -> Dropout -> LSTM -> Dropout -> three Dense (softmax) outputs
Run:  python train.py            (about 5-15 minutes on a normal laptop)
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's very long startup messages

import argparse
import json
import random
from pathlib import Path

import keras
import numpy as np
from keras import layers

from preprocess import DATA_FILE, build_dataset

MODEL_DIR = Path(__file__).parent / "model"
SEQ_LEN = 32


def build_model(n_pitches, n_durations):
    soprano_in = layers.Input(shape=(SEQ_LEN,), name="soprano_in")
    bass_in = layers.Input(shape=(SEQ_LEN,), name="bass_in")
    duration_in = layers.Input(shape=(SEQ_LEN,), name="duration_in")

    pitch_embedding = layers.Embedding(n_pitches, 48)         # shared: a pitch means the same in both voices
    duration_embedding = layers.Embedding(n_durations, 16)
    x = layers.Concatenate()([pitch_embedding(soprano_in), pitch_embedding(bass_in), duration_embedding(duration_in)])

    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(256)(x)
    x = layers.Dropout(0.3)(x)

    outputs = {
        "soprano": layers.Dense(n_pitches, activation="softmax", name="soprano")(x),
        "bass": layers.Dense(n_pitches, activation="softmax", name="bass")(x),
        "duration": layers.Dense(n_durations, activation="softmax", name="duration")(x),
    }
    model = keras.Model([soprano_in, bass_in, duration_in], outputs)
    model.compile(
        optimizer="adam",
        loss={name: "sparse_categorical_crossentropy" for name in outputs},
        metrics={name: ["accuracy"] for name in outputs},
    )
    return model


def make_windows(pieces, pitch_id, duration_id):
    """Slide a window over every piece: 32 events in, the next event out."""
    S, B, D, yS, yB, yD = [], [], [], [], [], []
    for piece in pieces:
        s = [pitch_id[e[0]] for e in piece]
        b = [pitch_id[e[1]] for e in piece]
        d = [duration_id[e[2]] for e in piece]
        for i in range(len(piece) - SEQ_LEN):
            S.append(s[i:i + SEQ_LEN]); B.append(b[i:i + SEQ_LEN]); D.append(d[i:i + SEQ_LEN])
            yS.append(s[i + SEQ_LEN]); yB.append(b[i + SEQ_LEN]); yD.append(d[i + SEQ_LEN])
    return [np.array(a) for a in (S, B, D, yS, yB, yD)]


def main():
    parser = argparse.ArgumentParser(description="Train the music generation model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--midi-dir", default=None)
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print("No dataset yet, running the preprocessing step first...")
        build_dataset(args.max_files, args.midi_dir)
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    pieces = data["pieces"]

    pitches = sorted({n for piece in pieces for e in piece for n in e[:2]})
    durations = sorted({e[2] for piece in pieces for e in piece})
    pitch_id = {p: i for i, p in enumerate(pitches)}
    duration_id = {d: i for i, d in enumerate(durations)}
    S, B, D, yS, yB, yD = make_windows(pieces, pitch_id, duration_id)
    print(f"{len(pieces)} pieces -> {len(S)} training examples, {len(pitches)} pitches, {len(durations)} durations")

    model = build_model(len(pitches), len(durations))
    model.summary()
    history = model.fit(
        {"soprano_in": S, "bass_in": B, "duration_in": D},
        {"soprano": yS, "bass": yB, "duration": yD},
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, shuffle=True, verbose=2,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "music_lstm.keras")
    random.seed(1)
    picks = random.sample(range(len(S)), min(200, len(S)))
    seeds = [[S[i].tolist(), B[i].tolist(), D[i].tolist()] for i in picks]
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"pitches": pitches, "durations": durations, "seeds": seeds, "seq_len": SEQ_LEN}, f)
    last = {k: v[-1] for k, v in history.history.items()}
    info = {
        "source": data["source"],
        "pieces": len(pieces),
        "events": sum(len(p) for p in pieces),
        "examples": int(len(S)),
        "parameters": int(model.count_params()),
        "epochs": args.epochs,
        "melody_accuracy": round(float(last["soprano_accuracy"]) * 100, 1),
        "bass_accuracy": round(float(last["bass_accuracy"]) * 100, 1),
        "duration_accuracy": round(float(last["duration_accuracy"]) * 100, 1),
    }
    with open(MODEL_DIR / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("\nTraining finished. Model saved in the 'model' folder.")
    print("Now run:  python generate.py   or   python app.py")


if __name__ == "__main__":
    main()
