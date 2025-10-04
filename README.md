# Whisper Sinhala Streamlit

Simple Streamlit app that loads the `kaan84/whisper-small-sinhala-proto` Whisper model from Hugging Face and transcribes uploaded audio files.

Files added:

- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies

Usage

1. Create and activate a Python environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Notes

- If you need MP3/M4A support, ensure `ffmpeg` is installed on your system (e.g., `sudo apt install ffmpeg`).
- The first model load downloads weights from Hugging Face and can take time.
- If you have a GPU and compatible PyTorch, the app will use it automatically.

Troubleshooting

- If audio can't be read, try converting to WAV with 16kHz sample rate.
- If model loading fails due to auth, ensure you have `huggingface-cli login` with access to the repo if it's private.
