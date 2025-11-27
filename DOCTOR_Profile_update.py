import streamlit as st
from pyannote.audio import Inference
import torchaudio
import soundfile as sf
import numpy as np
import io
import os

# Streamlit app configuration
st.set_page_config(page_title="Audio Recorder with Embedding", page_icon="🎙️")

# App title and instructions
st.title("Audio Recorder with Speaker Embedding")
st.markdown("""
Record audio or upload an audio file.  
The audio will be saved as a WAV file, and a speaker embedding will be generated and saved as a `.npy` file.
""")

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    try:
        return Inference("pyannote/embedding", window="whole")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

inference = load_embedding_model()

# Generate embedding function
def get_embedding_from_bytes(audio_bytes):
    temp_wav = "temp.wav"
    try:
        with io.BytesIO(audio_bytes) as buffer:
            audio_data, sr = sf.read(buffer)
            sf.write(temp_wav, audio_data, sr)

        waveform, sr = torchaudio.load(temp_wav)
        emb = inference({'waveform': waveform, 'sample_rate': sr})
        return emb
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


# -------------------------
# 1. RECORD AUDIO SECTION
# -------------------------
st.subheader("🎤 Record Audio")

recorded = st.audio_input("Click to start recording")

if recorded:
    st.audio(recorded, format="audio/wav")

    # Save audio
    wav_file = "recorded_audio.wav"
    with io.BytesIO(recorded.getbuffer()) as buffer:
        audio_data, sr = sf.read(buffer)
        sf.write(wav_file, audio_data, sr)

    st.success("Recorded audio saved.")

    # Generate embedding
    with st.spinner("Generating embedding..."):
        emb = get_embedding_from_bytes(recorded.getbuffer())

    if emb is not None:
        npy_file = "recorded_embedding.npy"
        np.save(npy_file, emb)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download WAV", open(wav_file, "rb"), wav_file)
        with col2:
            st.download_button("Download Embedding (.npy)", open(npy_file, "rb"), npy_file)


# -------------------------
# 2. UPLOAD AUDIO SECTION
# -------------------------
st.subheader("📁 Upload Audio File")

uploaded = st.file_uploader("Upload a WAV, MP3, M4A file", type=["wav", "mp3", "m4a"])

if uploaded:
    st.audio(uploaded, format="audio/wav")

    # Save uploaded audio to WAV
    wav_file = "uploaded_audio.wav"
    try:
        with io.BytesIO(uploaded.read()) as buffer:
            audio_data, sr = sf.read(buffer)
            sf.write(wav_file, audio_data, sr)

        st.success("Uploaded audio saved as WAV.")
    except Exception as e:
        st.error(f"Invalid audio format: {e}")
        st.stop()

    # Generate embedding
    with st.spinner("Generating embedding..."):
        with open(wav_file, "rb") as f:
            emb = get_embedding_from_bytes(f.read())

    if emb is not None:
        npy_file = "uploaded_embedding.npy"
        np.save(npy_file, emb)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download WAV", open(wav_file, "rb"), wav_file)
        with col2:
            st.download_button("Download Embedding (.npy)", open(npy_file, "rb"), npy_file)

