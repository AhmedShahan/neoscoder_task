import streamlit as st
import pandas as pd
import json
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pyaudio
import wave
import threading
import queue

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Transcription System",
    page_icon="🏥",
    layout="wide"
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
    .alert-emergency {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: black;
    }
    .alert-normal {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: black;
    }
    .soap-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Email Configuration
SMTP_HOST = os.getenv('SMTP_HOST', 'localhost')
SMTP_PORT = int(os.getenv('SMTP_PORT', '1025'))
SENDER_EMAIL = 'medical@system.com'

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
        self.audio.terminate()

def initialize_session_state():
    if 'stage' not in st.session_state:
        st.session_state.stage = 'upload'
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if 'emergency_status' not in st.session_state:
        st.session_state.emergency_status = None
    if 'soap_note' not in st.session_state:
        st.session_state.soap_note = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None
    if 'patient_email' not in st.session_state:
        st.session_state.patient_email = ""

def detect_emergency(text):
    """Detect emergency keywords in transcription"""
    emergency_keywords = [
        'chest pain', 'heart attack', 'stroke', 'unconscious',
        'severe bleeding', 'difficulty breathing', 'not breathing',
        'severe burn', 'overdose', 'suicide', 'seizure'
    ]
    text_lower = text.lower()
    matches = [kw for kw in emergency_keywords if kw in text_lower]
    return {
        'level': 'EMERGENCY' if matches else 'NORMAL',
        'matches': matches
    }

def generate_soap_note(transcription):
    """Generate SOAP note from transcription"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    
    soap_prompt = ChatPromptTemplate.from_template("""
You are a clinical documentation assistant.
Generate a comprehensive SOAP Note from this medical conversation:

{transcription}

Format:
SUBJECTIVE:
[Chief complaint and history]

OBJECTIVE:
[Physical findings and observations]

ASSESSMENT:
[Clinical impressions]

PLAN:
[Treatment recommendations]
""")
    
    chain = soap_prompt | llm | StrOutputParser()
    return chain.invoke({"transcription": transcription})

def generate_email_content(transcription, emergency_status):
    """Generate email content based on transcription and emergency status"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    
    email_prompt = ChatPromptTemplate.from_template("""
Generate a professional medical follow-up email based on this consultation.

Emergency Status: {emergency_level}
Transcription: {transcription}

Create an email with:
- Subject line
- Professional greeting
- Brief summary of consultation
- Next steps or urgent actions (if emergency)
- Closing

Return in this format:
SUBJECT: [subject line]

BODY:
[email body]
""")
    
    chain = email_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "emergency_level": emergency_status['level'],
        "transcription": transcription
    })
    
    # Parse result
    parts = result.split('BODY:', 1)
    subject = parts[0].replace('SUBJECT:', '').strip()
    body = parts[1].strip() if len(parts) > 1 else result
    
    return subject, body

def send_email(recipient, subject, body):
    """Send email via MailHog"""
    try:
        msg = MIMEText(body, 'plain')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email failed: {str(e)}")
        return False

def process_audio_simple(audio_path):
    """Simplified audio processing - just returns mock transcription"""
    # In real implementation, this would use speech recognition
    # For now, return a mock transcription
    return """Doctor: Hello, how are you feeling today?
Patient: I've been having chest pain for the last two hours.
Doctor: Can you describe the pain? Is it sharp or dull?
Patient: It's a sharp pain in the center of my chest.
Doctor: Any shortness of breath?
Patient: Yes, a little bit.
Doctor: We need to get you to the ER immediately. This could be serious."""

def render_upload_stage():
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Audio Upload/Recording</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Record Audio")
        if st.session_state.audio_recorder is None:
            st.session_state.audio_recorder = AudioRecorder()
        
        if not st.session_state.is_recording:
            if st.button("🔴 Start Recording", type="primary"):
                st.session_state.is_recording = True
                st.session_state.audio_recorder.start_recording()
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", type="secondary"):
                st.session_state.is_recording = False
                audio_file = st.session_state.audio_recorder.stop_recording()
                
                with st.spinner("Processing..."):
                    st.session_state.transcription = process_audio_simple(audio_file)
                    st.session_state.emergency_status = detect_emergency(st.session_state.transcription)
                    st.session_state.stage = 'transcription'
                    st.rerun()
    
    with col2:
        st.subheader("📁 Upload Audio")
        uploaded_file = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'm4a'])
        
        if uploaded_file:
            st.audio(uploaded_file)
            if st.button("🔄 Process Audio", type="primary"):
                temp_path = "temp_upload.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                with st.spinner("Processing..."):
                    st.session_state.transcription = process_audio_simple(temp_path)
                    st.session_state.emergency_status = detect_emergency(st.session_state.transcription)
                    st.session_state.stage = 'transcription'
                    st.rerun()

def render_transcription_stage():
    st.markdown("""
    <div class="main-header">
        <h1>💬 Transcription & Emergency Detection</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency Alert
    if st.session_state.emergency_status['level'] == 'EMERGENCY':
        matches = ', '.join(st.session_state.emergency_status['matches'])
        st.markdown(f"""
        <div class="alert-emergency">
            <h3>⚠️ EMERGENCY DETECTED</h3>
            <p><strong>Keywords:</strong> {matches}</p>
            <p><strong>Action:</strong> Immediate medical attention required!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-normal">
            <h4>✅ No Emergency Detected</h4>
            <p>Standard consultation procedure</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Transcription
    st.subheader("📝 Transcription")
    st.text_area("", value=st.session_state.transcription, height=300, disabled=True)
    
    # Actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.stage = 'upload'
            st.rerun()
    
    with col2:
        if st.button("📋 Generate SOAP Note", type="primary", use_container_width=True):
            with st.spinner("Generating SOAP note..."):
                st.session_state.soap_note = generate_soap_note(st.session_state.transcription)
                st.session_state.stage = 'soap'
                st.rerun()
    
    with col3:
        if st.button("📥 Download", use_container_width=True):
            st.download_button(
                "Download Transcription",
                st.session_state.transcription,
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

def render_soap_stage():
    st.markdown("""
    <div class="main-header">
        <h1>📋 SOAP Note</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="soap-card">
        <h2>SOAP NOTE</h2>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <div style="white-space: pre-wrap;">
{st.session_state.soap_note}
        </div>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient Email Input
    st.subheader("📧 Send Follow-up Email")
    patient_email = st.text_input("Patient Email Address", value=st.session_state.patient_email)
    st.session_state.patient_email = patient_email
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.stage = 'transcription'
            st.rerun()
    
    with col2:
        st.download_button(
            "📥 Download SOAP",
            st.session_state.soap_note,
            file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True
        )
    
    with col3:
        if st.button("📧 Generate & Send Email", type="primary", use_container_width=True):
            if not patient_email:
                st.error("Please enter patient email address")
            else:
                with st.spinner("Generating and sending email..."):
                    subject, body = generate_email_content(
                        st.session_state.transcription,
                        st.session_state.emergency_status
                    )
                    
                    if send_email(patient_email, subject, body):
                        st.success(f"✅ Email sent to {patient_email}")
                        st.info(f"**Subject:** {subject}")
                        with st.expander("View Email Content"):
                            st.text(body)
                    else:
                        st.error("Failed to send email")

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        stages = {
            'upload': '🎤 Upload/Record',
            'transcription': '💬 Transcription',
            'soap': '📋 SOAP Note'
        }
        for key, label in stages.items():
            if st.session_state.stage == key:
                st.markdown(f"**➤ {label}**")
            else:
                st.markdown(f"   {label}")
        
        st.divider()
        st.header("⚙️ Settings")
        st.text(f"SMTP: {SMTP_HOST}:{SMTP_PORT}")
        st.text(f"Sender: {SENDER_EMAIL}")
    
    # Main content
    if st.session_state.stage == 'upload':
        render_upload_stage()
    elif st.session_state.stage == 'transcription':
        render_transcription_stage()
    elif st.session_state.stage == 'soap':
        render_soap_stage()

if __name__ == "__main__":
    main()