import streamlit as st
import tempfile
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Your model repo
MODEL_ID = "kaan84/whisper-small-sinhala-proto"

# Optional imports (for mic mode)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    av = None


# ----------------------------
# Load model (cached)
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
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sinhala ASR Demo", page_icon="🎤")
st.title("Sinhala Speech-to-Text (Whisper)")

mode = st.radio("Input mode", ["Upload file", "Microphone (real-time)"])


# ----------------------------
# Transcription function
# ----------------------------
def transcribe_audio_np(audio_np: np.ndarray, sr: int = 16000):
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)

    # Convert to features
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
# Upload mode
# ----------------------------
if mode == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name

        # Load + resample with librosa
        audio_np, sr = librosa.load(audio_path, sr=16000)
        transcription = transcribe_audio_np(audio_np, sr=16000)

        st.audio(audio_path)
        st.success("✅ Transcription:")
        st.write(transcription)


# ----------------------------
# Microphone mode
# ----------------------------
else:
    if webrtc_streamer is None:
        st.error("Microphone mode requires extra packages. Install `streamlit-webrtc` and `av`.")
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
                    # Concatenate and resample
                    parts = []
                    for arr, sr in frames:
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        parts.append(arr)
                    audio_np = np.concatenate(parts)
                    transcription = transcribe_audio_np(audio_np, sr=16000)

                    st.success("✅ Transcription:")
                    st.write(transcription)
