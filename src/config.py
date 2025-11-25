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
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

# Hugging Face Model Names (instead of local paths)
HUGGINGFACE_MODELS = {
    # Whisper ASR Model
    'ASR_MODEL': os.getenv('ASR_MODEL', 'openai/whisper-base'),
    
    # PyAnnote Models
    'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'pyannote/embedding'),
    'DIARIZATION_MODEL': os.getenv('DIARIZATION_MODEL', 'pyannote/speaker-diarization-3.1'),
    
    # NER Model for De-identification
    'NER_MODEL': os.getenv('NER_MODEL', 'dslim/bert-base-NER'),
}

# Model Paths (for cached embeddings and local storage)
MODEL_PATHS = {
    'DOCTOR_EMBEDDING_PATH': os.getenv('DOCTOR_EMBEDDING_PATH', str(MODEL_DIR / 'doctor_embedding.npy')),
    'CACHE_DIR': str(MODEL_DIR / 'huggingface_cache')
}

# Processing Settings
PROCESSING_CONFIG = {
    'SIMILARITY_THRESHOLD': float(os.getenv('SIMILARITY_THRESHOLD', '0.60')),
    'MIN_SEGMENT_DURATION': 0.5,
    'DEVICE': 0,  # Use GPU if available, else -1 for CPU
    'USE_AUTH_TOKEN': os.getenv('HUGGINGFACE_TOKEN', None)  # Optional HF token for gated models
}

# API Configuration
API_CONFIG = {
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
    'GEMINI_MODEL': 'gemini-2.5-flash',
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