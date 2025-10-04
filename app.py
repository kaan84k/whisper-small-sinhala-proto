import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Your Hugging Face model repo
MODEL_ID = "kaan84/whisper-small-sinhala-proto"

# Optional imports for microphone mode
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    av = None


# ----------------------------
# Load Whisper model
# ----------------------------
@st.cache_resource
def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


processor, model, device = load_model()


# ----------------------------
# Transcription helper
# ----------------------------
def transcribe_audio_np(audio_np: np.ndarray, sr: int = 16000):
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)

    input_features = processor.feature_extractor(
        audio_np.astype("float32"),
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features.to(device)

    with st.spinner("Transcribing..."):
        predicted_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
    return transcription


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sinhala ASR Demo", page_icon="🎤")
st.title("Sinhala Speech-to-Text (Whisper)")

mode = st.radio("Input mode", ["Upload file", "Microphone (real-time)"])


# ----------------------------
# Upload mode
# ----------------------------
if mode == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV/FLAC/OGG recommended)",
        type=["wav", "flac", "ogg"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name

        try:
            # Load with soundfile (safe on Streamlit Cloud)
            audio_np, sr = sf.read(audio_path, dtype="float32")

            # Convert stereo → mono
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)

            # Resample if needed
            if sr != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
                sr = 16000

            transcription = transcribe_audio_np(audio_np, sr=sr)

            st.audio(audio_path)
            st.success("✅ Transcription:")
            st.write(transcription)

        except Exception as e:
            st.error(f"Could not read audio file: {e}")
            st.info("Tip: Please upload WAV or FLAC for best results.")


# ----------------------------
# Microphone mode
# ----------------------------
else:
    if webrtc_streamer is None:
        st.error("Microphone mode requires `streamlit-webrtc` and `av`. Install them and redeploy.")
    else:
        st.write("Allow microphone access, click Start, then Stop & Transcribe.")

        class AudioAccumulator:
            def __init__(self):
                self.frames = []

            def recv_audio(self, frame):
                try:
                    arr = frame.to_ndarray()
                    if arr.ndim > 1:
                        arr = np.mean(arr, axis=0)
                    if np.issubdtype(arr.dtype, np.integer):
                        arr = arr.astype("float32") / np.iinfo(arr.dtype).max
                    else:
                        arr = arr.astype("float32")
                    self.frames.append((arr, frame.sample_rate))
                except Exception:
                    pass
                return frame

        ctx = webrtc_streamer(
            key="mic",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioAccumulator,
        )

        if ctx and ctx.audio_processor:
            if st.button("Stop & Transcribe"):
                frames = ctx.audio_processor.frames
                if not frames:
                    st.warning("No audio captured yet. Speak into the microphone and try again.")
                else:
                    # Concatenate + resample
                    parts = []
                    for arr, sr in frames:
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        parts.append(arr)
                    audio_np = np.concatenate(parts)
                    transcription = transcribe_audio_np(audio_np, sr=16000)

                    st.success("✅ Transcription:")
                    st.write(transcription)
