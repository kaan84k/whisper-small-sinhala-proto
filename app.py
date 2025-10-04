import streamlit as st
import torchaudio
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "kaan84/whisper-small-sinhala-proto"

@st.cache_resource
def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="Sinhala ASR Demo", page_icon="🎤")
st.title("🎤 Sinhala Speech-to-Text (Whisper)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(uploaded_file.read())
        audio_path = tmpfile.name

    speech_array, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)

    input_features = processor.feature_extractor(
        speech_array.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    with st.spinner("Transcribing..."):
        predicted_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    st.audio(audio_path, format="audio/wav")
    st.success("✅ Transcription:")
    st.write(transcription)
