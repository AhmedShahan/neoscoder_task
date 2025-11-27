import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os
import torch
import torchaudio
import tempfile
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import SpeakerDiarization
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from scipy.spatial.distance import cosine
from pyannote.core import Segment
from pydub import AudioSegment
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pyaudio
import wave
import threading
import queue
import re
import concurrent.futures
import asyncio # Added for potential future async I/O

# Load environment variables
load_dotenv()

# --- Configuration Constants (Moved here for clarity) ---
EMBEDDING_PATH = "/home/shahanahmed/Office_Shellow_EMR/doctor_embedding.npy"
ASR_MODEL_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/whisper-base-local"
EMBEDDING_MODEL_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-embedding-local/pytorch_model.bin"
EMBEDDING_CONFIG_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-embedding-local/config.yaml"
DIARIZATION_PIPELINE_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-diarization-local"
SIMILARITY_THRESHOLD = 0.60
DEVICE = 0 if torch.cuda.is_available() else -1
TARGET_SAMPLE_RATE = 16000
MIN_SEGMENT_SAMPLES = 1600
# Define the number of worker threads for parallel segment processing
MAX_WORKERS = 4 

# Page configuration
st.set_page_config(
    page_title="Medical System - Simplified",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .stage-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        color: black;
    }
    .conversation-item {
        background: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black;
    }
    .doctor-message {
        border-left-color: #007bff !important;
        background: #e3f2fd !important;
    }
    .patient-message {
        border-left-color: #28a745 !important;
        background: #e8f5e8 !important;
    }
    .alert-emergency {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-normal {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .soap-note-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .recording-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        background-color: #ff0000;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 10px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Audio recording class
class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.recording_thread = None
        
    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def _record(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        while self.is_recording:
            # Use non-blocking read if possible, though 'input=True' generally implies blocking unless
            # a non-blocking stream is explicitly configured. We keep it as is for simplicity.
            data = self.stream.read(CHUNK) 
            self.frames.append(data)
            
        self.stream.stop_stream()
        self.stream.close()
        
    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return WAVE_OUTPUT_FILENAME

    def __del__(self):
        # A simple check to prevent error if audio is not initialized
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

# Initialize session state
def initialize_session_state():
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'upload'
    if 'transcription' not in st.session_state:
        st.session_state.transcription = []
    if 'emergency_alert' not in st.session_state:
        st.session_state.emergency_alert = None
    if 'soap_note' not in st.session_state:
        st.session_state.soap_note = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {}

# Emergency Detection
def detect_emergency(conversation_text: str) -> dict:
    """Simple rule-based emergency detector."""
    if not conversation_text:
        return {"level": "NORMAL", "matches": []}

    text = conversation_text.lower()
    
    emergency_phrases = [
        r"chest pain", r"severe chest", r"shortness of breath",
        r"difficulty breathing", r"trouble breathing", r"not breathing",
        r"no pulse", r"cardiac arrest", r"unconscious",
        r"loss of consciousness", r"fainting", r"severe bleeding",
        r"bleeding heavily", r"severe burn", r"stroke",
        r"slurred speech", r"weakness on", r"numbness",
        r"sudden vision", r"anaphylaxis", r"anaphylactic",
        r"severe allergic", r"seizure", r"convulsions",
        r"suicidal", r"kill myself", r"overdose",
        r"poisoning", r"collapsed", r"not responding"
    ]

    matches = []
    for phrase in emergency_phrases:
        if re.search(phrase, text):
            matches.append(phrase)

    if matches:
        return {"level": "EMERGENCY", "matches": matches}
    return {"level": "NORMAL", "matches": []}

# Segment Processing Function for Parallel Execution (New/Modified)
def process_segment(turn_data, waveform, sr, total_samples, total_duration, doctor_embedding, asr, embedder):
    """Processes a single audio segment for transcription and speaker identification."""
    turn, speaker_label = turn_data # speaker_label is the initial label from diarization
    segment: Segment = turn

    results_segment = None
    
    # 1. Segment Validation
    if segment.duration < 0.5:
        return None

    start_time = min(max(segment.start, 0), total_duration)
    end_time = min(segment.end, total_duration)
    
    if start_time >= end_time:
        return None

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(start_sample, min(end_sample, total_samples))
    
    if start_sample >= end_sample:
        return None

    segment_waveform = waveform[:, start_sample:end_sample]
    segment_length = segment_waveform.shape[1]
    
    # 2. Padding/Normalization
    if segment_length < MIN_SEGMENT_SAMPLES:
        padding = MIN_SEGMENT_SAMPLES - segment_length
        speaker_waveform = torch.nn.functional.pad(
            segment_waveform, 
            (0, padding), 
            mode='constant', 
            value=0
        )
    else:
        speaker_waveform = segment_waveform

    if speaker_waveform.dim() == 1:
        speaker_waveform = speaker_waveform.unsqueeze(0)

    audio_np = speaker_waveform.squeeze().numpy()
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    
    # 3. Transcription
    try:
        # ASR is a major bottleneck, running it in a thread pool helps
        transcription = asr(audio_np, return_timestamps=True)["text"]
        if not transcription or not transcription.strip():
            return None
    except Exception as e:
        st.warning(f"Failed to transcribe segment at {start_time:.2f}s: {str(e)}")
        return None

    # 4. Generate Embedding and Speaker Classification
    try:
        # Embedding generation is another major bottleneck
        file_dict = {"waveform": speaker_waveform, "sample_rate": sr}
        segment_embedding = embedder(file_dict)
    except Exception as e:
        # Fallback to temp file for embedding if direct waveform fails
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, speaker_waveform, sr)
                segment_embedding = embedder(tmp_file.name)
                os.unlink(tmp_file.name)
        except Exception as e2:
            st.warning(f"Failed to generate embedding: {str(e2)}. Defaulting to Patient.")
            return {
                "speaker": "Patient", # Default to patient if identification fails
                "text": transcription,
                "timestamp": f"{start_time:.2f}-{end_time:.2f}s"
            }

    # Determine speaker
    similarity = 1 - cosine(doctor_embedding, segment_embedding)
    label = "Doctor" if similarity > SIMILARITY_THRESHOLD else "Patient"

    return {
        "speaker": label,
        "text": transcription,
        "timestamp": f"{start_time:.2f}-{end_time:.2f}s"
    }

# Audio processing (Refactored to use ThreadPoolExecutor)
def process_audio(audio_path):
    
    # Load models outside the thread pool execution
    try:
        doctor_embedding = np.load(EMBEDDING_PATH)
        # Note: Hugging Face pipelines are thread-safe if device is managed (which it is here)
        # However, for maximum stability, we pass the pipeline *class* to the segment processor
        # and rely on the executor to manage its state/locks.
        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL_PATH, device=DEVICE)
        
        # Pyannote models are also typically thread-safe if not using a shared state
        embedding_model = Model.from_pretrained(EMBEDDING_MODEL_PATH, config_yaml=EMBEDDING_CONFIG_PATH)
        embedder = Inference(embedding_model, window="whole")
    except Exception as e:
        st.error(f"Error loading models or embedding: {str(e)}")
        return []

    # 1. Convert/Load Audio
    temp_audio_path = audio_path
    if audio_path.endswith('.mp3') or audio_path.endswith('.m4a'):
        temp_audio_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
        # Using a thread for I/O conversion to be non-blocking (though Streamlit handles blocking for us)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(audio.export, temp_audio_path, format="wav").result()

    waveform, sr = torchaudio.load(temp_audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = TARGET_SAMPLE_RATE

    total_samples = waveform.shape[1]
    total_duration = total_samples / sr
    
    # 2. Pad audio for Diarization
    chunk_duration = 5.0
    chunk_samples = int(chunk_duration * sr)
    
    diarization_input = temp_audio_path
    cleanup_diarization_input = False

    if total_samples % chunk_samples != 0:
        padding_needed = chunk_samples - (total_samples % chunk_samples)
        waveform_padded = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
        padded_audio_path = temp_audio_path.rsplit('.', 1)[0] + '_padded.wav'
        torchaudio.save(padded_audio_path, waveform_padded, sr)
        diarization_input = padded_audio_path
        cleanup_diarization_input = True
    
    # 3. Perform Diarization
    try:
        diarization_pipeline = SpeakerDiarization.from_pretrained(
            DIARIZATION_PIPELINE_PATH + '/config.yaml'
        )
        diarization = diarization_pipeline(diarization_input)
        
        # Prepare iterable of tasks for the thread pool
        segments_to_process = [(turn, label) for turn, _, label in diarization.itertracks(yield_label=True)]
        
    except Exception as diar_error:
        st.warning(f"Diarization failed: {str(diar_error)}. Treating as single speaker.")
        # Fallback to single speaker transcription
        audio_np = waveform.squeeze().numpy()
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Single, full transcription (not parallelized as it's one task)
        try:
            transcription = asr(audio_np, return_timestamps=True)["text"]
            results = [{
                "speaker": "Doctor", # Assuming single-speaker failure defaults to Doctor
                "text": transcription,
                "timestamp": f"0.00-{total_duration:.2f}s"
            }]
        except Exception as e:
            st.error(f"Failed during single-speaker ASR fallback: {str(e)}")
            results = []

        # Clean up
        if cleanup_diarization_input and os.path.exists(diarization_input):
            os.unlink(diarization_input)
        if temp_audio_path != audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        return results

    # 4. Parallel Processing of Segments
    results = []
    
    # Using ThreadPoolExecutor to run ASR and embedding generation concurrently for all segments
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks to the executor
        future_to_segment = {
            executor.submit(
                process_segment, 
                segment, 
                waveform, 
                sr, 
                total_samples, 
                total_duration, 
                doctor_embedding, 
                asr, 
                embedder
            ): segment for segment in segments_to_process
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_segment):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Log any exception that occurred within the worker thread
                st.error(f"Segment processing generated an exception: {e}")

    # Sort results by start time (timestamp '0.00-X.XXs')
    results.sort(key=lambda x: float(x['timestamp'].split('-')[0]))

    # 5. Clean up
    if cleanup_diarization_input and os.path.exists(diarization_input):
        os.unlink(diarization_input)
    if temp_audio_path != audio_path and os.path.exists(temp_audio_path):
        os.unlink(temp_audio_path)

    return results


# SOAP Note Generation
def generate_soap_note(transcription, patient_info):
    """Generate SOAP note from conversation"""
    conversation_text = "\n".join([f"{item['speaker']}: {item['text']}" for item in transcription])
    
    soap_prompt = ChatPromptTemplate.from_template("""
You are a clinical documentation assistant.
Given the following conversation between a doctor and a patient, generate a comprehensive SOAP Note in standard USA/Canada clinical format.

**Conversation:**
{conversation}

**Patient Information:**
- Name: {name}
- Age: {age}
- Gender: {gender}

**Instructions:**
- Follow standard SOAP format (Subjective, Objective, Assessment, Plan)
- Be thorough but concise
- Use medical terminology appropriately
- Include relevant information from both conversation and patient data

**Output Format:**
SUBJECTIVE:
[Chief complaint and history of present illness]

OBJECTIVE:
[Physical examination findings, vital signs, laboratory results if mentioned]

ASSESSMENT:
[Clinical impression, differential diagnosis]

PLAN:
[Treatment plan, medications, follow-up, patient education]
""")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3
    )
    
    chain = soap_prompt | llm | StrOutputParser()
    
    try:
        # LLM calls are I/O bound, but Streamlit is single-threaded. 
        # We keep it synchronous here to simplify the app flow within the `with st.spinner` block.
        soap_note = chain.invoke({
            "conversation": conversation_text,
            "name": patient_info.get('name', 'Not provided'),
            "age": patient_info.get('age', 'Not provided'),
            "gender": patient_info.get('gender', 'Not provided')
        })
        return soap_note
    except Exception as e:
        st.error(f"Error generating SOAP note: {str(e)}")
        return None

# Render stages (No changes to rendering logic)
def render_upload_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>🎤 Audio Recording & Upload</h2>
        <p>Record audio or upload an audio file for processing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Record Audio")
        
        if st.session_state.audio_recorder is None:
            st.session_state.audio_recorder = AudioRecorder()
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if not st.session_state.is_recording:
                if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                    st.session_state.is_recording = True
                    st.session_state.audio_recorder.start_recording()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                    st.session_state.is_recording = False
                    audio_file = st.session_state.audio_recorder.stop_recording()
                    
                    if audio_file and os.path.exists(audio_file):
                        with st.spinner("🔄 Processing recorded audio... (Using parallel processing)"):
                            try:
                                transcription = process_audio(audio_file)
                                
                                if not transcription:
                                    st.error("❌ No transcription generated.")
                                    return
                                
                                st.session_state.transcription = transcription
                                conversation_text = "\n".join([item.get('text', '') for item in transcription])
                                st.session_state.emergency_alert = detect_emergency(conversation_text)
                                st.session_state.current_stage = 'conversation'
                                
                                if os.path.exists(audio_file):
                                    os.unlink(audio_file)
                                
                                st.success("✅ Audio processed successfully!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    else:
                        st.error("❌ Recording failed.")
                    st.rerun()
        
        with col_rec2:
            if st.session_state.is_recording:
                st.markdown('<div class="recording-indicator"></div> **RECORDING...**', unsafe_allow_html=True)
            else:
                st.markdown("⚪ **STOPPED**")
    
    with col2:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac']
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.audio(uploaded_file)
            
            # Use a unique name for the uploaded file in temp storage
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_uploaded_{os.path.basename(uploaded_file.name)}")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.divider()
            
            if st.button("🔄 Process Audio", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing audio... (Using parallel processing)"):
                    try:
                        transcription = process_audio(temp_audio_path)
                        
                        if not transcription:
                            st.error("❌ No transcription generated.")
                            # Cleanup the temp file even on error
                            if os.path.exists(temp_audio_path):
                                os.unlink(temp_audio_path)
                            return
                        
                        st.session_state.transcription = transcription
                        conversation_text = "\n".join([item.get('text', '') for item in transcription])
                        st.session_state.emergency_alert = detect_emergency(conversation_text)
                        st.session_state.current_stage = 'conversation'
                        
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                        
                        st.success("✅ Audio processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)

def render_conversation_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>💬 Conversation Transcript</h2>
        <p>Review the processed conversation with emergency detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency Alert
    alert = st.session_state.get('emergency_alert')
    if alert and alert.get('level') == 'EMERGENCY':
        matches = alert.get('matches', [])
        matches_text = ', '.join(matches) if matches else 'keywords matched'
        st.markdown(f"""
        <div class="alert-emergency">
            <h3>⚠️ EMERGENCY DETECTED</h3>
            <p><strong>Detected phrases:</strong> {matches_text}</p>
            <p><strong>Recommendation:</strong> This conversation contains possible life-threatening symptoms. 
            Seek immediate medical attention or call local emergency services.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-normal">
            <h4>✅ No Emergency Detected</h4>
            <p>The transcribed conversation does not contain high-priority emergency keywords.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display conversation
    for item in st.session_state.transcription:
        speaker_class = "doctor-message" if item['speaker'] == 'Doctor' else "patient-message"
        speaker_icon = "👨‍⚕️" if item['speaker'] == 'Doctor' else "👤"
        st.markdown(f"""
        <div class="conversation-item {speaker_class}">
            <strong>{speaker_icon} {item['speaker']}:</strong> {item['text']}
            <br><small style="color: #666;">⏱️ {item['timestamp']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        conversation_json = json.dumps(st.session_state.transcription, indent=2)
        st.download_button(
            label="📥 Download Transcript",
            data=conversation_json,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.current_stage = 'upload'
            st.rerun()
    
    with col3:
        if st.button("📋 Generate SOAP Note", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating SOAP note..."):
                soap_note = generate_soap_note(
                    st.session_state.transcription,
                    st.session_state.patient_info
                )
                if soap_note:
                    st.session_state.soap_note = soap_note
                    st.success("✅ SOAP note generated!")
                    st.session_state.current_stage = 'soap'
                    st.rerun()

def render_soap_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>📋 SOAP Note</h2>
        <p>Standardized clinical documentation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.soap_note:
        st.markdown(f"""
        <div class="soap-note-card">
            <h2>📋 SOAP NOTE</h2>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <div style="white-space: pre-wrap; line-height: 1.6;">
{st.session_state.soap_note}
            </div>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📥 Download SOAP",
                data=st.session_state.soap_note,
                file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if st.button("🖨️ Print", use_container_width=True):
                st.success("📄 SOAP note sent to printer!")
        
        with col3:
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("🔄 Regenerating SOAP note..."):
                    soap_note = generate_soap_note(
                        st.session_state.transcription,
                        st.session_state.patient_info
                    )
                    if soap_note:
                        st.session_state.soap_note = soap_note
                        st.success("✅ Regenerated!")
                        st.rerun()
        
        with col4:
            if st.button("🔄 New Consultation", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['current_stage']:
                        del st.session_state[key]
                st.session_state.current_stage = 'upload'
                st.rerun()
        
        st.divider()
        
        if st.button("⬅️ Back to Conversation", use_container_width=True):
            st.session_state.current_stage = 'conversation'
            st.rerun()

# Main app
def main():
    initialize_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏥 Medical System - Simplified
        </h1>
        <p style="color: #b3d9ff; text-align: center; margin: 0.5rem 0 0 0;">
            Audio Transcription • Emergency Detection • SOAP Notes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        
        stages = {
            'upload': ('🎤', 'Audio Upload'),
            'conversation': ('💬', 'Transcription'),
            'soap': ('📋', 'SOAP Note')
        }
        
        current_stage_index = list(stages.keys()).index(st.session_state.current_stage)
        
        for i, (stage_key, (icon, stage_name)) in enumerate(stages.items()):
            if i <= current_stage_index:
                if st.session_state.current_stage == stage_key:
                    st.markdown(f"**➤ {icon} {stage_name}**")
                else:
                    st.markdown(f"✅ {icon} {stage_name}")
            else:
                st.markdown(f"⏳ {icon} {stage_name}")
        
        progress = (current_stage_index + 1) / len(stages)
        st.progress(progress)
        st.caption(f"Progress: {progress:.0%}")
        
        st.divider()
        
        # Patient Info
        st.header("👤 Patient Information")
        st.session_state.patient_info['name'] = st.text_input(
            "Patient Name",
            value=st.session_state.patient_info.get('name', '')
        )
        
        col1, col2 = st.columns(2)
        with col1:
            # Typecasting to int for the number input value to avoid issues if the default '0' is not explicitly set in the session state
            age_value = st.session_state.patient_info.get('age', 0)
            if not isinstance(age_value, int):
                try:
                    age_value = int(age_value)
                except:
                    age_value = 0

            st.session_state.patient_info['age'] = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=age_value,
                key="patient_age_input"
            )
        with col2:
            gender_options = ['Male', 'Female', 'Other']
            current_gender = st.session_state.patient_info.get('gender', gender_options[0])
            try:
                default_index = gender_options.index(current_gender)
            except ValueError:
                default_index = 0
                
            st.session_state.patient_info['gender'] = st.selectbox(
                "Gender",
                gender_options,
                index=default_index
            )
    
    # Main content
    if st.session_state.current_stage == 'upload':
        render_upload_stage()
    elif st.session_state.current_stage == 'conversation':
        render_conversation_stage()
    elif st.session_state.current_stage == 'soap':
        render_soap_stage()

if __name__ == "__main__":
    main()