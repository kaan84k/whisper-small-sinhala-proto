import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import librosa
from io import BytesIO
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    av = None


MODEL_DEFAULT = "kaan84/whisper-small-sinhala-proto"


@st.cache_resource
def load_model_and_processor(repo_id: str):
    """Load processor and model from Hugging Face and move model to the available device.

    Returns: (processor, model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(repo_id)
    model = WhisperForConditionalGeneration.from_pretrained(repo_id)
    model.to(device)
    return processor, model, device


def read_audio_bytes(audio_bytes: bytes, target_sr: int = 16000, filename: str = None):
    """Read audio from raw bytes and return 1d float32 numpy array at target sampling rate.

    This tries to read using soundfile (libsndfile) first. If that fails (commonly for mp3/m4a),
    it writes the bytes to a temporary file and uses librosa.load which can leverage ffmpeg/audioread
    backends to decode formats not supported by libsndfile.
    """
    f = BytesIO(audio_bytes)
    try:
        data, sr = sf.read(f, dtype="float32")
    except Exception:
        # Fallback: write to a temporary file and use librosa (which can use ffmpeg/audioread)
        import tempfile
        import os

        suffix = os.path.splitext(filename)[1] if filename else ".tmp"
        # Ensure a reasonable suffix
        if not suffix:
            suffix = ".tmp"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # librosa.load will resample if `sr` is provided. We ask librosa to return mono float32.
            data, sr = librosa.load(tmp_path, sr=target_sr, mono=True, dtype="float32")
            # librosa already resampled when sr=target_sr, so keep sr as target
            sr = target_sr
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    # If stereo, make mono (libsndfile may return multi-channel)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # Ensure float32
    return data.astype("float32"), target_sr


def transcribe_audio(audio_np: np.ndarray, processor: WhisperProcessor, model: WhisperForConditionalGeneration, device: str):
    """Run the model to transcribe the provided audio numpy array.

    Returns the transcribed string.
    """
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # Generate tokens
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def main():
    st.set_page_config(page_title="Whisper Sinhala Streamlit", layout="wide")
    st.title("Whisper (Sinhala) — Streamlit Transcription")

    st.markdown(
        "Upload an audio file and the app will transcribe it using a Whisper model hosted on Hugging Face. "
        "Supported file types depend on the soundfile/ffmpeg backends (wav, flac, ogg, mp3 with ffmpeg installed)."
    )

    repo_id = st.text_input("Model repo id", value=MODEL_DEFAULT)
    col1, col2 = st.columns([2, 1])

    mode = col2.selectbox("Input mode", options=["Upload file", "Microphone (real-time)"])

    with col1:
        if mode == "Upload file":
            audio_file = st.file_uploader("Upload audio", type=["wav", "flac", "mp3", "m4a", "ogg"])
        else:
            audio_file = None

        language_hint = st.selectbox("Language hint (optional)", options=["auto", "si (Sinhala)", "en (English)"], index=0)

    with col2:
        if st.button("Load model"):
            st.session_state["loaded"] = False

    # Load model once and cache
    with st.spinner("Loading model (this may take a while the first time)..."):
        try:
            processor, model, device = load_model_and_processor(repo_id)
            st.success(f"Model loaded on {device}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if mode == "Upload file":
        if audio_file is None:
            st.info("Please upload an audio file to transcribe.")
            return

        audio_bytes = audio_file.read()

        try:
            filename = getattr(audio_file, "name", None)
            audio_np, sr = read_audio_bytes(audio_bytes, target_sr=16000, filename=filename)
        except Exception as e:
            st.error(f"Couldn't read audio file: {e}\nIf the file is mp3/m4a you may need ffmpeg installed on your system.")
            return

        st.audio(audio_bytes)

        if st.button("Transcribe"):
            with st.spinner("Transcribing — this can take a while depending on CPU/GPU"):
                try:
                    text = transcribe_audio(audio_np, processor, model, device)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

            st.subheader("Transcription")
            st.text_area("Result", value=text, height=200)
            st.download_button("Download transcript", data=text, file_name="transcript.txt")

    else:  # Microphone mode
        if webrtc_streamer is None:
            st.error("streamlit-webrtc is not installed. Install dependencies from requirements.txt and restart.")
            return

        st.write("Allow microphone access in your browser. Click 'Start' to begin streaming audio from your microphone.")

        class AudioRecorder:
            def __init__(self):
                self.frames = []  # list of (np.ndarray, sample_rate)

            def recv_audio(self, frame):
                try:
                    arr = frame.to_ndarray()  # shape: (channels, samples)
                    # convert to mono
                    if arr.ndim > 1:
                        arr = np.mean(arr, axis=0)
                    # ensure float32
                    if np.issubdtype(arr.dtype, np.integer):
                        arr = arr.astype("float32") / np.iinfo(arr.dtype).max
                    else:
                        arr = arr.astype("float32")
                    self.frames.append((arr, frame.sample_rate))
                except Exception:
                    pass
                return frame

        # Start webrtc streamer in sendonly mode to capture mic audio
        ctx = webrtc_streamer(key="mic", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioRecorder)

        if ctx and ctx.audio_processor:
            if st.button("Stop & Transcribe"):
                frames = ctx.audio_processor.frames
                if not frames:
                    st.warning("No audio captured yet. Speak into the microphone and click Stop & Transcribe again.")
                else:
                    # Resample and concatenate
                    parts = []
                    for arr, sr in frames:
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        parts.append(arr)
                    audio_np = np.concatenate(parts)
                    with st.spinner("Transcribing microphone audio..."):
                        try:
                            text = transcribe_audio(audio_np, processor, model, device)
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                            return

                    st.subheader("Transcription")
                    st.text_area("Result", value=text, height=200)
                    st.download_button("Download transcript", data=text, file_name="transcript.txt")


if __name__ == "__main__":
    main()
