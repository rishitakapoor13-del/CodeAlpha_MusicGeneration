# CodeAlpha_MusicGeneration

AI music generation built for the **CodeAlpha Artificial Intelligence Internship (Task 3)**.

An **LSTM neural network** learns from Bach chorales and then composes new music. You can generate music in a web page, play it, see it as a piano roll, and download it as MIDI or WAV.

## How it works
1. **Collect data** - the Bach chorales that come with the `music21` library (about 390 pieces). You can also use your own `.mid` files (classical, jazz...).
2. **Preprocess** (`preprocess.py`) - with `music21`, every piece is moved to one key, its voices are merged into chords, and each chord becomes an event: **[melody note, bass note, duration]**.
3. **Build the model** (`train.py`) - a Keras LSTM: Embedding, LSTM, Dropout, LSTM, Dropout, then three softmax outputs that predict the next melody note, bass note and duration.
4. **Train** - the model reads 32 events and learns to predict the next one.
5. **Generate** (`generate.py`) - start from a real 32-event snippet, predict the next event, add it, and repeat. A "creativity" (temperature) setting controls how adventurous the choices are.
6. **Convert to MIDI and audio** (`music_utils.py`) - events become a MIDI file with `music21`, and a small built-in synthesizer renders a `.wav` file you can play anywhere.

## Features
- LSTM model with three outputs (melody, bass, duration)
- Adjustable length, creativity and tempo
- Web page with a live piano roll and audio player
- Download as MIDI and WAV
- Command-line generation too

## Project files
| File | Purpose |
|------|---------|
| `preprocess.py` | Collect music and make note sequences |
| `train.py` | Build and train the LSTM |
| `generate.py` | Compose new music (command line) |
| `music_utils.py` | Events to MIDI and WAV |
| `app.py` | Flask web server |
| `index.html`, `style.css`, `script.js` | The web page |
| `requirements.txt` | Python libraries |

## How to run
1. Install Python 3.9 - 3.12 (check with `python --version`)
2. Install the libraries (TensorFlow is a large download):
   ```
   pip install -r requirements.txt
   ```
3. Train the model (about 5-15 minutes, saves to the `model` folder):
   ```
   python train.py
   ```
   For a quick test: `python train.py --epochs 3`
4. Start the web page:
   ```
   python app.py
   ```
   and open http://127.0.0.1:5000
5. Or generate from the command line: `python generate.py`

## Using your own music
Put `.mid` files in a folder, then run:
```
python preprocess.py --midi-dir my_midi_files
python train.py
```

## Notes
- Output is short, simple music in the style of the training data. It is a learning project, not professional composition.
- `data/`, `model/` and `generated/` are created when you run the scripts, so they are not uploaded to GitHub.

## Author
Rishita Kapoor - CodeAlpha AI Intern
