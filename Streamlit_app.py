import streamlit as st
import whisper
import librosa
import tempfile
import numpy as np
import os
import noisereduce as nr
from scipy.io.wavfile import write

st.title("Real-Time Speech-to-Text Transcriber")
st.markdown("Upload an audio file to get the transcription with timestamps.")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

def transcribe_audio(file):
    audio, sr = librosa.load(file, sr=None)

    audio = nr.reduce_noise(y=audio, sr=sr)

    max_duration = 30
    audio = audio[:sr * max_duration]

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        write(tmp.name, sr, (audio * 32767).astype(np.int16))
        result = model.transcribe(tmp.name, fp16=False)
    os.remove(tmp.name)
    return result, audio, sr

uploaded_file = st.file_uploader("Upload Audio (.wav/.mp3)", type=["wav", "mp3"])

if uploaded_file:
    st.markdown("""
    **Note:** This app may take 1–2 minutes to generate a transcript after uploading a file.
    """, unsafe_allow_html=True)

    st.markdown("""
    **Why is it slow?** The Whisper model is a powerful Transformer-based model. Due to limited compute on Hugging Face Spaces (CPU-only), inference takes longer than on a local GPU. Please be patient.
    """, unsafe_allow_html=True)
    
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("Transcribing... please wait "):
        result, audio, sr = transcribe_audio(uploaded_file)

    st.success("Transcription Complete! ")
    st.subheader("Transcript")
    st.write(result["text"])

    st.subheader("Timestamps")
    for seg in result["segments"]:
        st.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")


