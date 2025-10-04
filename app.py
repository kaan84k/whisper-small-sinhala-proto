import streamlit as st
import tempfile
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "kaan84/whisper-small-sinhala-proto"

try:
    # streamlit-webrtc is optional; we guard its import so upload-only still works
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    av = None


@st.cache_resource
def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu" if hasattr(__import__('torch'), 'cuda') else "cpu"
    try:
        import torch
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    return processor, model


processor, model = load_model()

st.set_page_config(page_title="Sinhala ASR Demo", page_icon="🎤")
st.title("Sinhala Speech-to-Text (Whisper)")

mode = st.radio("Input mode", ["Upload file", "Microphone (real-time)"])


def transcribe_audio_np(audio_np: np.ndarray):
    # Ensure 16k
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)
    # resample if needed - assume caller passes 16000 where possible
    # Prepare input features
    input_features = processor.feature_extractor(
        audio_np.astype("float32"), sampling_rate=16000, return_tensors="pt"
    ).input_features
    with st.spinner("Transcribing..."):
        predicted_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


if mode == "Upload file":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name

        speech_array, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech_array = resampler(speech_array)

        audio_np = speech_array.squeeze().numpy()
        transcription = transcribe_audio_np(audio_np)

        st.audio(audio_path)
        st.success("✅ Transcription:")
        st.write(transcription)

else:
    # Microphone mode
    if webrtc_streamer is None:
        st.error("Microphone mode requires additional packages. Install streamlit-webrtc and av (PyAV) and restart the app.")
    else:
        st.write("Allow microphone access in your browser, then click Start. When finished, click Stop & Transcribe.")

        class AudioAccumulator:
            def __init__(self):
                self.frames = []

            def recv_audio(self, frame):
                try:
                    arr = frame.to_ndarray()
                    # arr shape usually (channels, samples)
                    if arr.ndim > 1:
                        arr = np.mean(arr, axis=0)
                    # convert int -> float32 if necessary
                    if np.issubdtype(arr.dtype, np.integer):
                        arr = arr.astype("float32") / np.iinfo(arr.dtype).max
                    else:
                        arr = arr.astype("float32")
                    self.frames.append((arr, frame.sample_rate))
                except Exception:
                    pass
                return frame

        ctx = webrtc_streamer(key="mic", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioAccumulator)

        if ctx and ctx.audio_processor:
            if st.button("Stop & Transcribe"):
                frames = ctx.audio_processor.frames
                if not frames:
                    st.warning("No audio captured yet. Speak into the microphone and try again.")
                else:
                    parts = []
                    for arr, sr in frames:
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        parts.append(arr)
                    audio_np = np.concatenate(parts)
                    transcription = transcribe_audio_np(audio_np)
                    st.success("Transcription:")
                    st.write(transcription)
