"""
Web page for the music generator.
Run:  python app.py     then open  http://127.0.0.1:5000
(Train the model first with:  python train.py)
"""
import time
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from generate import OUTPUT_DIR, Composer
from music_utils import render_wav, save_midi, tokens_to_events

BASE = Path(__file__).parent
PUBLIC_FILES = {"style.css", "script.js"}

app = Flask(__name__)
composer = Composer()
OUTPUT_DIR.mkdir(exist_ok=True)


@app.route("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/<name>")
def public_file(name):
    if name not in PUBLIC_FILES:
        abort(404)
    return send_from_directory(BASE, name)


@app.route("/generated/<name>")
def generated_file(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route("/api/info")
def info():
    return jsonify(composer.info)


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    length = max(16, min(int(data.get("length", 96)), 300))
    temperature = max(0.3, min(float(data.get("temperature", 1.0)), 2.0))
    bpm = max(40, min(int(data.get("tempo", 90)), 180))

    tokens = composer.compose(length, temperature)
    name = f"song_{int(time.time() * 1000)}"
    save_midi(tokens, OUTPUT_DIR / f"{name}.mid", bpm)
    render_wav(tokens, OUTPUT_DIR / f"{name}.wav", bpm)

    old_files = sorted(OUTPUT_DIR.glob("song_*"), key=lambda p: p.stat().st_mtime)[:-20]   # keep the folder tidy
    for path in old_files:
        path.unlink(missing_ok=True)

    events = tokens_to_events(tokens, bpm)
    return jsonify(
        wav=f"/generated/{name}.wav",
        midi=f"/generated/{name}.mid",
        events=events,
        seconds=round(events[-1]["start"] + events[-1]["dur"], 1),
    )


if __name__ == "__main__":
    app.run(debug=False)
