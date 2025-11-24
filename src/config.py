"""
Configuration and Constants for Medical Diagnostic System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio Recording Settings
AUDIO_CONFIG = {
    'FORMAT': 'paInt16',
    'CHANNELS': 1,
    'RATE': 16000,
    'CHUNK': 1024,
    'RECORD_SECONDS': 300,
    'TARGET_SAMPLE_RATE': 16000,
    'MIN_SEGMENT_SAMPLES': 1600
}

# Model Paths
MODEL_PATHS = {
    'EMBEDDING_PATH': os.getenv('EMBEDDING_PATH', str(MODEL_DIR / 'doctor_embedding.npy')),
    'ASR_MODEL_PATH': os.getenv('ASR_MODEL_PATH', str(MODEL_DIR / 'whisper-base-local')),
    'EMBEDDING_MODEL_PATH': os.getenv('EMBEDDING_MODEL_PATH', str(MODEL_DIR / 'pyannote-embedding-local/pytorch_model.bin')),
    'EMBEDDING_CONFIG_PATH': os.getenv('EMBEDDING_CONFIG_PATH', str(MODEL_DIR / 'pyannote-embedding-local/config.yaml')),
    'DIARIZATION_PIPELINE_PATH': os.getenv('DIARIZATION_PIPELINE_PATH', str(MODEL_DIR / 'pyannote-diarization-local'))
}

# Processing Settings
PROCESSING_CONFIG = {
    'SIMILARITY_THRESHOLD': float(os.getenv('SIMILARITY_THRESHOLD', '0.60')),
    'MIN_SEGMENT_DURATION': 0.5,
    'DEVICE': 0  # Use GPU if available, else -1 for CPU
}

# API Configuration
API_CONFIG = {
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
    'GEMINI_MODEL': 'gemini-1.5-flash',
    'TEMPERATURE': 0.3
}

# RxNorm/FDA API URLs
DRUG_API_URLS = {
    'RXNORM_BASE': 'https://rxnav.nlm.nih.gov/REST',
    'OPENFDA_BASE': 'https://api.fda.gov/drug',
    'RXNORM_TIMEOUT': 5
}

# PHI Replacement Tags
PHI_REPLACE_TAGS = {
    "PER": "[NAME]",
    "LOC": "[LOCATION]", 
    "ORG": "[ORGANIZATION]",
    "DATE": "[DATE]",
    "MISC": "[INFO]",
}

# Output File Paths
OUTPUT_FILES = {
    'PATIENT_INFO': OUTPUT_DIR / 'result.json',
    'DIAGNOSIS': OUTPUT_DIR / 'diagnosis_result.json',
    'MEDICINE': OUTPUT_DIR / 'medicine_result.json',
    'ALERTS': OUTPUT_DIR / 'alert_result.json',
    'TEMP_AUDIO': OUTPUT_DIR / 'temp_audio.wav'
}

# Streamlit Page Configuration
PAGE_CONFIG = {
    'page_title': "Medical Diagnostic System",
    'page_icon': "🏥",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Stage Names
STAGES = {
    'upload': ('🎤', 'Audio Upload'),
    'conversation': ('💬', 'Conversation'),
    'soap': ('📋', 'SOAP Note'),
    'diagnostic': ('🔍', 'Diagnostic'),
    'medicines': ('💊', 'Medicines'),
    'alerts': ('⚠️', 'Drug Alerts'),
    'suggestions': ('💡', 'Suggestions'),
    'prescription': ('📄', 'Prescription')
}