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
Record audio using your microphone. The audio will be saved as a WAV file, and a speaker embedding will be generated and saved as a `.npy` file. Use the download buttons to retrieve the files.
""")

# Initialize pyannote.audio embedding model
@st.cache_resource
def load_embedding_model():
    try:
        return Inference("pyannote/embedding", window="whole")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}. Please check your Hugging Face token and internet connection.")
        st.stop()

inference = load_embedding_model()

# Function to generate embedding from audio bytes
def get_embedding(audio_bytes):
    temp_wav = "temp_audio.wav"
    try:
        # Read audio bytes and save as temporary WAV
        with io.BytesIO(audio_bytes) as audio_buffer:
            audio_data, sample_rate = sf.read(audio_buffer)
            sf.write(temp_wav, audio_data, sample_rate, format="WAV")
        
        # Load waveform and generate embedding
        waveform, sample_rate = torchaudio.load(temp_wav)
        embedding = inference({'waveform': waveform, 'sample_rate': sample_rate})
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

# Audio input widget
audio = st.audio_input("Record audio (click to start/stop)", disabled=False)

if audio:
    # Display the recorded audio
    st.audio(audio, format="audio/wav")
    
    # Save audio as WAV
    output_audio_file = "recorded_audio.wav"
    try:
        with io.BytesIO(audio.getbuffer()) as audio_buffer:
            audio_data, sample_rate = sf.read(audio_buffer)
            sf.write(output_audio_file, audio_data, sample_rate, format="WAV")
        st.success(f"Audio saved as '{output_audio_file}'")
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        st.stop()
    
    # Generate and save embedding
    with st.spinner("Generating speaker embedding..."):
        embedding = get_embedding(audio.getbuffer())
    
    if embedding is not None:
        output_embedding_file = "recorded_embedding.npy"
        try:
            np.save(output_embedding_file, embedding)
            st.success(f"Embedding saved as '{output_embedding_file}'")
            
            # Provide download buttons
            col1, col2 = st.columns(2)
            with col1:
                with open(output_audio_file, "rb") as f:
                    st.download_button(
                        label="Download WAV file",
                        data=f,
                        file_name=output_audio_file,
                        mime="audio/wav",
                        key="download_wav"
                    )
            with col2:
                with open(output_embedding_file, "rb") as f:
                    st.download_button(
                        label="Download embedding (.npy)",
                        data=f,
                        file_name=output_embedding_file,
                        mime="application/octet-stream",
                        key="download_npy"
                    )
        except Exception as e:
            st.error(f"Error saving embedding: {e}")
    else:
        st.warning("Embedding generation failed. You can still download the audio.")
        
        # Provide download button for audio only
        with open(output_audio_file, "rb") as f:
            st.download_button(
                label="Download WAV file",
                data=f,
                file_name=output_audio_file,
                mime="audio/wav",
                key="download_wav_fallback"
            )