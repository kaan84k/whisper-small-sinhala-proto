import streamlit as st
import torchaudio
import tempfile
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(uploaded_file.read())
        audio_path = tmpfile.name

    try:
        # Convert everything to WAV (16kHz) using ffmpeg
        wav_path = audio_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Load the converted file
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
