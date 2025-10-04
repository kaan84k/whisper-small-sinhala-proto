import streamlit as st
import torchaudio
import tempfile
import subprocess
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Hugging Face model repo
MODEL_ID = "kaan84/whisper-small-sinhala-proto"

@st.cache_resource
def load_model():
    """Load processor and model once, cached for faster reloads."""
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    return processor, model

# Load model and processor
processor, model = load_model()

# Streamlit UI
st.set_page_config(page_title="Sinhala ASR Demo", page_icon="🎤")
st.title("🎤 Sinhala Speech-to-Text (Whisper Fine-tuned)")

# Input option: Upload or Record
option = st.radio("Choose input method:", ["Upload File", "Record Audio"])

audio_path = None

# -------- Upload --------
if option == "Upload File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name

# -------- Record --------
elif option == "Record Audio":
    st.info("Click Start to record from your microphone.")
    frames = []

    def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().mean(axis=1)  # mono
        frames.append(audio)
        return frame

    ctx = webrtc_streamer(
        key="speech-recorder",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=None,
        in_audio_frame_callback=audio_callback,
    )

    if ctx.state.playing and st.button("Stop & Save Recording"):
        if frames:
            wav_path = tempfile.mktemp(suffix=".wav")
            audio_np = np.concatenate(frames).astype(np.float32)
            sf.write(wav_path, audio_np, 16000)  # save as wav 16kHz
            audio_path = wav_path
            st.success("Recording saved!")

# -------- Transcription --------
if audio_path is not None:
    try:
        # Ensure WAV 16kHz using ffmpeg
        wav_path = audio_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Load audio
        speech_array, sr = torchaudio.load(wav_path)

        # Extract features
        input_features = processor.feature_extractor(
            speech_array.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Run model inference
        with st.spinner("⏳ Transcribing..."):
            predicted_ids = model.generate(input_features)
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        # Show results
        st.audio(wav_path, format="audio/wav")
        st.success("✅ Transcription:")
        st.write(transcription)

    except Exception as e:
        st.error(f"❌ Could not process audio file: {e}")
