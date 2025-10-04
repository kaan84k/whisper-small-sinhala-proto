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

Installing torchaudio

The `torchaudio` package is optional but recommended. It has platform- and
CUDA-dependent wheels, so installing it via `pip` may require selecting the
correct index. If you already installed `torch` using the instructions at
https://pytorch.org/get-started/locally/, re-run the recommended `pip` command
from that page to install a matching `torchaudio`. For a simple CPU-only
example on Linux you can do:

```bash
pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you don't install `torchaudio`, the app will fall back to `librosa` for
loading audio files (slower but easier to install).

Troubleshooting

- If audio can't be read, try converting to WAV with 16kHz sample rate.
- If model loading fails due to auth, ensure you have `huggingface-cli login` with access to the repo if it's private.
