import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import io
import base64
from typing import List, Dict, Any, Optional
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
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import requests
from dotenv import load_dotenv
from pathlib import Path
import uuid

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Diagnostic System",
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
        color: black;
    }
    .patient-message {
        border-left-color: #28a745 !important;
        background: #e8f5e8 !important;
        color: black;
    }
    .alert-high {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: black;
    }
    .alert-moderate {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: black;
    }
    .alert-low {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: black;
    }
    .prescription-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .soap-note-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .recording-controls {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #28a745;
        margin: 1rem 0;
        text-align: center;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'upload'
    if 'transcription' not in st.session_state:
        st.session_state.transcription = []
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {}
    if 'disease_suggestions' not in st.session_state:
        st.session_state.disease_suggestions = []
    if 'selected_diseases' not in st.session_state:
        st.session_state.selected_diseases = []
    if 'medicine_suggestions' not in st.session_state:
        st.session_state.medicine_suggestions = []
    if 'selected_medicines' not in st.session_state:
        st.session_state.selected_medicines = []
    if 'drug_alerts' not in st.session_state:
        st.session_state.drug_alerts = DrugAlertAnalysis(alerts=[], safe_combinations=[], overall_risk_level="LOW")
    if 'final_prescription' not in st.session_state:
        st.session_state.final_prescription = None
    if 'medicine_ids' not in st.session_state:
        st.session_state.medicine_ids = {}
    if 'disease_ids' not in st.session_state:
        st.session_state.disease_ids = {}
    if 'soap_note' not in st.session_state:
        st.session_state.soap_note = None
    if 'deidentified_conversation' not in st.session_state:
        st.session_state.deidentified_conversation = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None
    if 'prescription_confirmed' not in st.session_state:
        st.session_state.prescription_confirmed = False
    # Add new session state variables for suggestions
    if 'conversation_suggestions' not in st.session_state:
        st.session_state.conversation_suggestions = []
    if 'ai_suggestions' not in st.session_state:
        st.session_state.ai_suggestions = []
    if 'custom_suggestions' not in st.session_state:
        st.session_state.custom_suggestions = []
    if 'selected_suggestions' not in st.session_state:
        st.session_state.selected_suggestions = []

# Helper function to get unique ID for medicines
def get_medicine_unique_id(medicine_name):
    if medicine_name not in st.session_state.medicine_ids:
        st.session_state.medicine_ids[medicine_name] = str(uuid.uuid4())[:8]
    return st.session_state.medicine_ids[medicine_name]

# Helper function to get unique ID for diseases
def get_disease_unique_id(disease_name):
    if disease_name not in st.session_state.disease_ids:
        st.session_state.disease_ids[disease_name] = str(uuid.uuid4())[:8]
    return st.session_state.disease_ids[disease_name]

# Schemas
class PatientInformation(BaseModel):
    Patient_Name: Optional[str] = Field(default=None)
    Age: Optional[int] = Field(default=None)
    Gender: Optional[str] = Field(default=None)
    Recent_Problem: Optional[str] = Field(default=None)
    Previous_Medical_History: Optional[str] = Field(default=None)
    Previous_Drug_History: Optional[str] = Field(default=None)
    Allergies: Optional[str] = Field(default=None)
    Family_Medical_History: Optional[str] = Field(default=None)
    Lifestyle_Details: Optional[Dict[str, str]] = Field(default=None)
    Current_Medical_Tests_Ordered: Optional[str] = Field(default=None)
    Previous_Medical_Test_History: Optional[str] = Field(default=None)
    Follow_Up_Actions: Optional[List[str]] = Field(default=None)
    Emotional_State: Optional[str] = Field(default=None)

class DiseaseSuggestion(BaseModel):
    disease: str
    score: float
    reason: str

class DiagnosisSuggestions(BaseModel):
    suggestions: List[DiseaseSuggestion]

class MedicineSuggestion(BaseModel):
    medicine: str
    score: float
    reason: str
    purpose: Optional[str] = None
    side_effects: Optional[str] = None

class MedicineSuggestions(BaseModel):
    suggestions: List[MedicineSuggestion]

class DrugAlert(BaseModel):
    alert_type: str
    severity: str
    drug1: str
    drug2: Optional[str] = None
    description: str
    recommendation: str

class DrugAlertAnalysis(BaseModel):
    alerts: List[DrugAlert]
    safe_combinations: List[str] = []
    overall_risk_level: str

# NER and De-identification Setup
@st.cache_resource
def load_ner_model():
    """Load NER model for de-identification"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return ner_pipeline
    except Exception as e:
        st.warning(f"Could not load NER model: {e}. Using fallback method.")
        return None

# PHI Replacement Map
REPLACE_TAGS = {
    "PER": "[NAME]",
    "LOC": "[LOCATION]", 
    "ORG": "[ORGANIZATION]",
    "DATE": "[DATE]",
    "MISC": "[INFO]",
}

def deidentify_conversation(conversation_text, ner_pipeline=None):
    """De-identify conversation text using NER"""
    if ner_pipeline is None:
        # Fallback: simple pattern-based replacement
        import re
        # Replace common patterns
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', conversation_text)  # Names
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)  # Dates
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)  # Phone numbers
        text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', '[ADDRESS]', text)  # Addresses
        return text
    
    try:
        entities = ner_pipeline(conversation_text)
        new_text = conversation_text
        
        # Sort entities by start position in reverse order to avoid index shifting
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            label = entity['entity_group']
            replacement = REPLACE_TAGS.get(label, "[REDACTED]")
            new_text = new_text[:entity['start']] + replacement + new_text[entity['end']:]
        
        return new_text
    except Exception as e:
        st.warning(f"NER de-identification failed: {e}. Using original text.")
        return conversation_text

def format_conversation_for_soap(transcription):
    """Format transcription for SOAP note generation"""
    return "\n".join([f"{item['speaker']}: {item['text']}" for item in transcription])

def generate_soap_note(conversation_text, patient_info):
    """Generate SOAP note from de-identified conversation"""
    # De-identify the conversation
    ner_pipeline = load_ner_model()
    deidentified_text = deidentify_conversation(conversation_text, ner_pipeline)
    
    # Store for display purposes
    st.session_state.deidentified_conversation = deidentified_text
    
    # SOAP Note Prompt
    soap_prompt = ChatPromptTemplate.from_template("""
You are a clinical documentation assistant.
Given the following **de-identified** conversation between a doctor and a patient, generate a comprehensive SOAP Note in standard USA/Canada clinical format.
**De-identified Conversation:**
{conversation}
**Additional Patient Information:**
- Age: {age}
- Gender: {gender}
- Previous Medical History: {medical_history}
- Previous Drug History: {drug_history}
- Allergies: {allergies}
- Family Medical History: {family_history}
**Instructions:**
- Follow standard SOAP format (Subjective, Objective, Assessment, Plan)
- Be thorough but concise
- Use medical terminology appropriately
- Include relevant information from both conversation and patient data
- Do not include any personal identifying information
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
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )
    
    # Create chain
    chain = soap_prompt | llm | StrOutputParser()
    
    try:
        soap_note = chain.invoke({
            "conversation": deidentified_text,
            "age": patient_info.get('Age', 'Not provided'),
            "gender": patient_info.get('Gender', 'Not provided'),
            "medical_history": patient_info.get('Previous_Medical_History', 'Not provided'),
            "drug_history": patient_info.get('Previous_Drug_History', 'Not provided'),
            "allergies": patient_info.get('Allergies', 'Not provided'),
            "family_history": patient_info.get('Family_Medical_History', 'Not provided')
        })
        
        return soap_note
    except Exception as e:
        st.error(f"Error generating SOAP note: {str(e)}")
        return None

# RxNorm-based drug information
@tool
def get_drug_info(drug_name: str) -> dict:
    """
    Gets drug purpose and side effects using RxNorm, RxClass, and OpenFDA.
    Returns a dictionary with purpose and side effects or error messages.
    """
    # Step 1: Get RxCUI from RxNorm
    rxcui_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
    try:
        rxcui_response = requests.get(rxcui_url, timeout=5)
        if rxcui_response.status_code != 200:
            return {"error": f"Error fetching RxCUI: {rxcui_response.status_code}", "purpose": None, "side_effects": None}
        
        rxcui_data = rxcui_response.json()
        rxcui_list = rxcui_data.get('idGroup', {}).get('rxnormId', [])
        if not rxcui_list:
            return {"error": f"No RxNorm ID found for {drug_name}", "purpose": None, "side_effects": None}
        
        rxcui = rxcui_list[0]
        
        # Step 2: Get Purpose (Therapeutic Class) from RxClass
        purpose_url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
        purpose_response = requests.get(purpose_url, timeout=5)
        purpose = "Not found"
        if purpose_response.status_code == 200:
            purpose_data = purpose_response.json()
            class_membership = purpose_data.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
            if class_membership:
                # Get the first 3 classes to keep it concise
                classes = [info['rxclassMinConceptItem']['className'] for info in class_membership[:3]]
                purpose = ", ".join(classes)
        
        # Step 3: Get Side Effects from OpenFDA
        openfda_url = f"https://api.fda.gov/drug/event.json?search=patient.drug.openfda.brand_name:{drug_name}&limit=3"
        fda_response = requests.get(openfda_url, timeout=5)
        side_effects = []
        if fda_response.status_code == 200:
            fda_data = fda_response.json()
            results = fda_data.get('results', [])
            for result in results:
                reactions = result.get('patient', {}).get('reaction', [])
                for r in reactions:
                    reaction_name = r.get('reactionmeddrapt')
                    if reaction_name:
                        side_effects.append(reaction_name)
        
        # Deduplicate, limit to top 5 most common, and create a concise list
        side_effects = list(set(side_effects))[:5]
        side_effects_text = ", ".join(side_effects) if side_effects else "Not found"
        
        return {
            "error": None,
            "purpose": purpose,
            "side_effects": side_effects_text,
            "contraindications": None,
            "warnings": None
        }
    except Exception as e:
        return {"error": f"Drug info request failed: {str(e)}", "purpose": None, "side_effects": None, "contraindications": None, "warnings": None}

# RxNorm-based drug interactions
@tool
def get_drug_interactions(drug_list: List[str]) -> dict:
    """
    Gets drug-drug interactions using RxNorm.
    Returns a dictionary with interaction information.
    """
    interactions = []
    
    # If we have less than 2 drugs, no interactions possible
    if len(drug_list) < 2:
        return {"interactions": [], "error": "Need at least 2 drugs to check interactions"}
    
    try:
        # Get RxCUI for each drug
        rxcui_map = {}
        for drug in drug_list:
            rxcui_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug}"
            rxcui_response = requests.get(rxcui_url, timeout=5)
            if rxcui_response.status_code == 200:
                rxcui_data = rxcui_response.json()
                rxcui_list = rxcui_data.get('idGroup', {}).get('rxnormId', [])
                if rxcui_list:
                    rxcui_map[drug] = rxcui_list[0]
        
        # Check interactions between each pair of drugs
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                if drug1 in rxcui_map and drug2 in rxcui_map:
                    rxcui1 = rxcui_map[drug1]
                    rxcui2 = rxcui_map[drug2]
                    
                    # Get interaction information
                    interaction_url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui1}&rxcui={rxcui2}"
                    interaction_response = requests.get(interaction_url, timeout=5)
                    
                    if interaction_response.status_code == 200:
                        interaction_data = interaction_response.json()
                        interaction_list = interaction_data.get('interactionTypeGroup', [])
                        
                        for group in interaction_list:
                            for interaction_type in group.get('interactionType', []):
                                for interaction_pair in interaction_type.get('interactionPair', []):
                                    description = interaction_pair.get('description', '')
                                    severity = interaction_pair.get('severity', 'Unknown')
                                    
                                    if description:
                                        interactions.append({
                                            "drug1": drug1,
                                            "drug2": drug2,
                                            "description": description,
                                            "severity": severity,
                                            "source": "RxNorm"
                                        })
        
        return {"interactions": interactions, "error": None}
    
    except Exception as e:
        return {"interactions": [], "error": f"Failed to get drug interactions: {str(e)}"}

# Model and prompts
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

diagnosis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Medical Diagnostic Suggestion Engine."),
    ("human", """
Patient data:
Recent Problem: {Recent_Problem}
Previous Medical History: {Previous_Medical_History}
Previous Drug History: {Previous_Drug_History}
Allergies: {Allergies}
Family Medical History: {Family_Medical_History}
Lifestyle Details: {Lifestyle_Details}
Suggest at least 5 possible diseases with confidence scores (0 to 1) and brief reasons.
Respond ONLY in JSON in this format:
{{
  "suggestions": [
    {{
      "disease": "Disease Name",
      "score": 0.85,
      "reason": "Explanation"
    }},
    ...
  ]
}}
{format_instructions}
""")
])

medicine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Medical Treatment Recommendation Engine."),
    ("human", """
Patient data:
Recent Problem: {Recent_Problem}
Previous Medical History: {Previous_Medical_History}
Family Medical History: {Family_Medical_History}
Selected Diseases: {Selected_Diseases}
Suggest around 10 medicines with confidence scores (0 to 1). 
Focus on suggesting actual brand name medicines that would be available in FDA databases.
Provide brief initial reasons, but note that detailed purpose and side effects will be fetched from OpenFDA.
Respond ONLY in JSON in this format:
{{
  "suggestions": [
    {{
      "medicine": "Medicine Brand Name",
      "score": 0.90,
      "reason": "Brief explanation for recommendation"
    }},
    ...
  ]
}}
{format_instructions}
""")
])

drug_interaction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Clinical Pharmacology Safety Expert specialized in drug interactions and contraindications analysis.
    You must analyze drug combinations for potential interactions, allergic reactions, and medical history conflicts.
    Be extremely thorough and err on the side of caution for patient safety."""),
    ("human", """
Analyze the following for drug safety alerts:
PATIENT INFORMATION:
- Allergies: {allergies}
- Previous Medical History: {medical_history}
- Current Medical Problems: {current_problems}
- Previous Drug History: {previous_drugs}
PROPOSED MEDICATIONS:
{proposed_medications}
ANALYSIS REQUIRED:
1. Drug-to-Drug Interactions between proposed medications
2. Allergy conflicts with any proposed medication
3. Medical history contraindications 
4. Any other safety concerns
For each potential issue, provide:
- Alert type (DRUG_INTERACTION, ALLERGY_CONFLICT, MEDICAL_HISTORY_CONFLICT, CONTRAINDICATION)
- Severity (HIGH, MODERATE, LOW)
- Detailed description
- Clinical recommendation
Also identify any safe combinations and provide an overall risk assessment.
{format_instructions}
""")
])

diagnosis_parser = PydanticOutputParser(pydantic_object=DiagnosisSuggestions)
medicine_parser = PydanticOutputParser(pydantic_object=MedicineSuggestions)
drug_alert_parser = PydanticOutputParser(pydantic_object=DrugAlertAnalysis)

# Audio recording functionality
def start_recording():
    """Placeholder for audio recording start"""
    st.session_state.is_recording = True

def stop_recording():
    """Placeholder for audio recording stop"""
    st.session_state.is_recording = False
    # In a real implementation, this would save the recorded audio
    return None

# Audio processing
def process_audio(audio_path):
    EMBEDDING_PATH = "/home/shahanahmed/Office_Shellow_EMR/doctor_embedding.npy"
    ASR_MODEL_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/whisper-base-local"
    EMBEDDING_MODEL_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-embedding-local/pytorch_model.bin"
    EMBEDDING_CONFIG_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-embedding-local/config.yaml"
    DIARIZATION_PIPELINE_PATH = "/home/shahanahmed/Office_Shellow_EMR/model/pyannote-diarization-local"
    SIMILARITY_THRESHOLD = 0.60
    DEVICE = 0 if torch.cuda.is_available() else -1

    if audio_path.endswith('.mp3'):
        output_wav_path = audio_path.replace('.mp3', '.wav')
        audio = AudioSegment.from_mp3(audio_path)
        audio.export(output_wav_path, format="wav")
        audio_path = output_wav_path

    try:
        doctor_embedding = np.load(EMBEDDING_PATH)
        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL_PATH, device=DEVICE)
        embedding_model = Model.from_pretrained(EMBEDDING_MODEL_PATH, config_yaml=EMBEDDING_CONFIG_PATH)
        embedder = Inference(embedding_model, window="whole")
        waveform_conv, sr_conv = torchaudio.load(audio_path)
        if sr_conv != 16000:
            waveform_conv = torchaudio.transforms.Resample(sr_conv, 16000)(waveform_conv)
            sr_conv = 16000
        diarization_pipeline = SpeakerDiarization.from_pretrained(DIARIZATION_PIPELINE_PATH + '/config.yaml')
        diarization = diarization_pipeline(audio_path)
        results = []
        for turn in diarization.itertracks(yield_label=True):
            segment: Segment = turn[0]
            if segment.duration < 0.5:
                continue
            start = int(segment.start * sr_conv)
            end = int(segment.end * sr_conv)
            speaker_waveform = waveform_conv[:, start:end]
            if speaker_waveform.dim() == 1:
                speaker_waveform = speaker_waveform.unsqueeze(0)
            if speaker_waveform.shape[1] < 1600:
                continue
            audio_np = speaker_waveform.squeeze().numpy()
            if audio_np.dtype != 'float32':
                audio_np = audio_np.astype('float32')
            transcription = asr(audio_np)["text"]
            try:
                file_dict = {"waveform": speaker_waveform, "sample_rate": sr_conv}
                segment_embedding = embedder(file_dict)
            except Exception as e:
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        torchaudio.save(tmp_file.name, speaker_waveform, sr_conv)
                        segment_embedding = embedder(tmp_file.name)
                        os.unlink(tmp_file.name)
                except Exception as e2:
                    continue
            similarity = 1 - cosine(doctor_embedding, segment_embedding)
            label = "Doctor" if similarity > SIMILARITY_THRESHOLD else "Patient"
            results.append({"speaker": label, "text": transcription, "timestamp": f"{segment.start:.2f}-{segment.end:.2f}s"})
        return results
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return []

# Structured data extraction
def extract_structured_data(transcription):
    patient_parser = PydanticOutputParser(pydantic_object=PatientInformation)
    prescription_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Patient Information Extractor. Respond in valid JSON only."),
        ("human", "{conversation}\n\n{format_instructions}")
    ])
    chain = prescription_prompt | model | patient_parser
    conversation_text = "\n".join([f"{item['speaker']}: {item['text']}" for item in transcription])
    try:
        result = chain.invoke({
            "conversation": conversation_text,
            "format_instructions": patient_parser.get_format_instructions()
        })
        return result.dict()
    except Exception as e:
        st.error(f"Error extracting patient information: {str(e)}")
        return {}

# Diagnostic and medicine suggestion engines
def generate_disease_suggestions(patient_info, transcription):
    diagnosis_chain = diagnosis_prompt | model | diagnosis_parser
    try:
        result = diagnosis_chain.invoke({
            "Recent_Problem": patient_info.get("Recent_Problem", ""),
            "Previous_Medical_History": patient_info.get("Previous_Medical_History", ""),
            "Previous_Drug_History": patient_info.get("Previous_Drug_History", ""),
            "Allergies": patient_info.get("Allergies", ""),
            "Family_Medical_History": patient_info.get("Family_Medical_History", ""),
            "Lifestyle_Details": patient_info.get("Lifestyle_Details", {}),
            "format_instructions": diagnosis_parser.get_format_instructions()
        })
        with open("/home/shahanahmed/Office_Shellow_EMR/diagnosis_result.json", "w") as json_file:
            json.dump(result.dict(), json_file, indent=4)
        return result.dict()['suggestions']
    except Exception as e:
        st.error(f"Error generating diagnostic suggestions: {str(e)}")
        return []

def enrich_medicines_with_fda_data(medicines: List[Dict]) -> List[Dict]:
    enriched_medicines = []
    for med in medicines:
        drug_info = get_drug_info.invoke({"drug_name": med['medicine']})
        enriched_med = {
            "medicine": med['medicine'],
            "score": med['score'],
            "reason": med['reason'],
            "purpose": drug_info.get("purpose"),
            "side_effects": drug_info.get("side_effects")
        }
        enriched_medicines.append(enriched_med)
    return enriched_medicines

def generate_medicine_suggestions(diseases):
    medicine_chain = medicine_prompt | model | medicine_parser
    diseases_string = ", ".join([d['disease'] for d in diseases]) if diseases else "No specific diseases selected - base suggestions on symptoms and patient history"
    try:
        result = medicine_chain.invoke({
            "Recent_Problem": st.session_state.patient_info.get("Recent_Problem", ""),
            "Previous_Medical_History": st.session_state.patient_info.get("Previous_Medical_History", ""),
            "Family_Medical_History": st.session_state.patient_info.get("Family_Medical_History", ""),
            "Selected_Diseases": diseases_string,
            "format_instructions": medicine_parser.get_format_instructions()
        })
        enriched_medicines = enrich_medicines_with_fda_data(result.dict()['suggestions'])
        with open("/home/shahanahmed/Office_Shellow_EMR/medicine_result.json", "w") as json_file:
            json.dump({"suggestions": enriched_medicines}, json_file, indent=4)
        return enriched_medicines
    except Exception as e:
        st.error(f"Error generating medicine suggestions: {str(e)}")
        return []

def analyze_drug_interactions(selected_medicines, patient_info):
    # Extract drug names
    drug_names = [med['medicine'] for med in selected_medicines]
    
    # Get RxNorm interactions
    rxnorm_interactions = get_drug_interactions.invoke({"drug_list": drug_names})
    
    # Also use our existing LLM-based analysis for comprehensive coverage
    medicine_list = "\n".join([f"- {med['medicine']}: {med['reason']}" for med in selected_medicines])
    interaction_chain = drug_interaction_prompt | model | drug_alert_parser
    
    try:
        # Get LLM-based analysis
        alert_analysis = interaction_chain.invoke({
            "allergies": patient_info.get("Allergies", "None reported"),
            "medical_history": patient_info.get("Previous_Medical_History", "None reported"),
            "current_problems": patient_info.get("Recent_Problem", "None reported"),
            "previous_drugs": patient_info.get("Previous_Drug_History", "None reported"),
            "proposed_medicines": medicine_list,
            "format_instructions": drug_alert_parser.get_format_instructions()
        })
        
        # Convert RxNorm interactions to DrugAlert objects
        rxnorm_alerts = []
        for interaction in rxnorm_interactions.get("interactions", []):
            # Map RxNorm severity to our severity levels
            severity_map = {
                "high": "HIGH",
                "moderate": "MODERATE", 
                "low": "LOW",
                "unknown": "UNKNOWN"
            }
            severity = severity_map.get(interaction["severity"].lower(), "MODERATE")
            
            rxnorm_alerts.append(DrugAlert(
                alert_type="DRUG_INTERACTION",
                severity=severity,
                drug1=interaction["drug1"],
                drug2=interaction["drug2"],
                description=interaction["description"],
                recommendation="Consult with pharmacist or consider alternative medications"
            ))
        
        # Combine RxNorm alerts with LLM alerts
        combined_alerts = alert_analysis.alerts + rxnorm_alerts
        
        # Update the alert analysis with combined alerts
        alert_analysis.alerts = combined_alerts
        
        # If we have HIGH severity alerts, update overall risk level
        if any(alert.severity == "HIGH" for alert in combined_alerts):
            alert_analysis.overall_risk_level = "HIGH"
        elif any(alert.severity == "MODERATE" for alert in combined_alerts):
            alert_analysis.overall_risk_level = "MODERATE to HIGH"
        
        # Save results
        with open("/home/shahanahmed/Office_Shellow_EMR/alert_result.json", "w") as json_file:
            json.dump(alert_analysis.dict(), json_file, indent=4)
        
        return alert_analysis
    
    except Exception as e:
        st.error(f"Error in drug interaction analysis: {str(e)}")
        
        # If LLM analysis fails, fall back to just RxNorm interactions
        rxnorm_alerts = []
        for interaction in rxnorm_interactions.get("interactions", []):
            severity_map = {
                "high": "HIGH",
                "moderate": "MODERATE", 
                "low": "LOW",
                "unknown": "UNKNOWN"
            }
            severity = severity_map.get(interaction["severity"].lower(), "MODERATE")
            
            rxnorm_alerts.append(DrugAlert(
                alert_type="DRUG_INTERACTION",
                severity=severity,
                drug1=interaction["drug1"],
                drug2=interaction["drug2"],
                description=interaction["description"],
                recommendation="Consult with pharmacist or consider alternative medications"
            ))
        
        # Determine overall risk level
        overall_risk = "LOW"
        if any(alert.severity == "HIGH" for alert in rxnorm_alerts):
            overall_risk = "HIGH"
        elif any(alert.severity == "MODERATE" for alert in rxnorm_alerts):
            overall_risk = "MODERATE"
        
        return DrugAlertAnalysis(
            alerts=rxnorm_alerts,
            safe_combinations=[],
            overall_risk_level=overall_risk
        )

# AI suggestions generator
def generate_ai_suggestions(selected_medicines, patient_info, selected_diseases):
    """Generate AI-based suggestions based on drug side effects and patient condition"""
    
    # Format medication information
    medications_text = "\n".join([
        f"- {med['medicine']}: Side effects include {med.get('side_effects', 'Not available')}" 
        for med in selected_medicines
    ])
    
    # Create prompt for AI suggestions
    suggestion_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Medical Suggestion Engine. Provide practical suggestions for the patient based on their medications and condition."),
        ("human", """
        Patient Information:
        - Diagnoses: {diagnoses}
        - Medications: {medications}
        - Medical History: {medical_history}
        - Allergies: {allergies}
        - Age: {age}
        - Gender: {gender}

        Based on the above information, generate 5 practical suggestions for the patient to manage their condition and medication side effects.
        Focus on lifestyle adjustments, monitoring, and when to seek medical help.

        Respond ONLY in JSON format as a list of strings:
        {{
            "suggestions": [
                "Suggestion 1",
                "Suggestion 2",
                ...
            ]
        }}
        """)
    ])
    
    # Create chain
    chain = suggestion_prompt | model | JsonOutputParser()
    
    try:
        result = chain.invoke({
            "diagnoses": ", ".join(selected_diseases),
            "medications": medications_text,
            "medical_history": patient_info.get("Previous_Medical_History", "None"),
            "allergies": patient_info.get("Allergies", "None"),
            "age": patient_info.get("Age", "Not specified"),
            "gender": patient_info.get("Gender", "Not specified")
        })
        return result.get("suggestions", [])
    except Exception as e:
        st.error(f"Error generating AI suggestions: {str(e)}")
        return []

# Helper function to escape LaTeX special characters
def escape_latex(text):
    if text is None:
        return 'Not provided'
    replacements = {
        '&': r'\&',
        '%': r'\%',
        ',': r'\,',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde',
        '^': r'\textasciicircum',
        '\\': r'\textbackslash'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

# Calculate consultation duration
def calculate_consultation_duration(transcription):
    """Calculate actual consultation duration from transcription timestamps"""
    if not transcription:
        return 0
    
    # Extract all timestamps
    timestamps = []
    for item in transcription:
        timestamp_str = item.get('timestamp', '0-0s')
        # Remove 's' and split into start and end
        timestamp_parts = timestamp_str.replace('s', '').split('-')
        if len(timestamp_parts) == 2:
            try:
                start = float(timestamp_parts[0])
                end = float(timestamp_parts[1])
                timestamps.append((start, end))
            except ValueError:
                continue
    
    if not timestamps:
        return 0
    
    # Find the earliest start and latest end times
    min_start = min(t[0] for t in timestamps)
    max_end = max(t[1] for t in timestamps)
    
    # Calculate duration in minutes
    duration_seconds = max_end - min_start
    duration_minutes = duration_seconds / 60
    
    return duration_minutes

# Generate final prescription
def generate_final_prescription():
    """Generate the final prescription including selected suggestions"""
    st.session_state.final_prescription = {
        'patient_name': st.session_state.patient_info.get('Patient_Name', 'Patient Name'),
        'age': st.session_state.patient_info.get('Age', 'N/A'),
        'gender': st.session_state.patient_info.get('Gender', 'N/A'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'diseases': st.session_state.selected_diseases,
        'medicines': st.session_state.selected_medicines,
        'alerts': st.session_state.drug_alerts.dict()['alerts'],
        'medical_history': st.session_state.patient_info.get('Previous_Medical_History', 'Not provided'),
        'previous_drug_history': st.session_state.patient_info.get('Previous_Drug_History', 'Not provided'),
        'lifestyle_details': st.session_state.patient_info.get('Lifestyle_Details', {}),
        'doctor_suggestions': st.session_state.selected_suggestions,  # Use selected suggestions
        'medical_tests_advised': st.session_state.patient_info.get('Current_Medical_Tests_Ordered', 'None advised'),
        'soap_note': st.session_state.soap_note
    }
    st.session_state.current_stage = 'prescription'
    st.rerun()

# Rendering functions
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
        
        st.markdown("""
        <div class="recording-controls">
            <h4>🎵 Live Audio Recording</h4>
            <p>Click the button below to start/stop recording</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if not st.session_state.is_recording:
                if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                    start_recording()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                    stop_recording()
                    st.warning("⚠️ Recording functionality requires additional implementation with audio capture libraries.")
                    st.rerun()
        
        with col_rec2:
            if st.session_state.is_recording:
                st.markdown("🔴 **RECORDING...**")
            else:
                st.markdown("⚪ **STOPPED**")
        
        if st.session_state.is_recording:
            st.info("🎤 Recording in progress... Click 'Stop Recording' when finished.")
    
    with col2:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Supported formats: WAV, MP3, M4A, FLAC"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.audio(uploaded_file)
            
            # Save uploaded file temporarily
            temp_audio_path = "/home/shahanahmed/Office_Shellow_EMR/temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.divider()
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Process Audio", type="primary", use_container_width=True):
                    with st.spinner("🔄 Processing audio... This may take a few moments."):
                        try:
                            # Process the audio
                            transcription = process_audio(temp_audio_path)
                            
                            if not transcription:
                                st.error("❌ No transcription generated. Please try another audio file.")
                                return
                            
                            # Store transcription
                            st.session_state.transcription = transcription
                            
                            # Extract patient information
                            patient_info = extract_structured_data(transcription)
                            st.session_state.patient_info.update(patient_info)
                            
                            # Save patient info to file
                            with open("/home/shahanahmed/Office_Shellow_EMR/result.json", "w") as json_file:
                                json.dump(patient_info, json_file, indent=4)
                            
                            # Move to next stage without generating suggestions
                            st.session_state.current_stage = 'conversation'
                            
                            # Clean up temporary file
                            if os.path.exists(temp_audio_path):
                                os.unlink(temp_audio_path)
                            
                            st.session_state.prescription_confirmed = False  # Reset confirmation
                            st.success("✅ Audio processed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error processing audio: {str(e)}")

def render_conversation_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>💬 Conversation Transcript</h2>
        <p>Review the processed conversation between doctor and patient.</p>
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🖨️ Print Conversation", use_container_width=True):
            st.success("📄 Conversation sent to printer! (Simulated)")
    
    with col2:
        conversation_json = json.dumps(st.session_state.transcription, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=conversation_json,
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        if st.button("📋 Generate SOAP Note", use_container_width=True):
            with st.spinner("🔄 Generating SOAP note..."):
                conversation_text = format_conversation_for_soap(st.session_state.transcription)
                soap_note = generate_soap_note(conversation_text, st.session_state.patient_info)
                if soap_note:
                    st.session_state.soap_note = soap_note
                    st.success("✅ SOAP note generated!")
                    st.session_state.current_stage = 'soap'
                    st.rerun()
    
    with col4:
        if st.button("🔍 Diagnostic Engine", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating diagnostic suggestions..."):
                # Generate disease suggestions here
                st.session_state.disease_suggestions = generate_disease_suggestions(
                    st.session_state.patient_info, 
                    st.session_state.transcription
                )
                st.session_state.current_stage = 'diagnostic'
                st.rerun()

def render_soap_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>📋 SOAP Note</h2>
        <p>Standardized clinical documentation with de-identified patient information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.soap_note:
        # Display SOAP note
        st.markdown(f"""
        <div class="soap-note-card">
            <h2>📋 SOAP NOTE</h2>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <div style="white-space: pre-wrap; line-height: 1.6;">
{st.session_state.soap_note}
            </div>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Note:</strong> Personal identifying information has been removed for privacy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show de-identified conversation
        if st.session_state.deidentified_conversation:
            with st.expander("🔒 De-identified Conversation Used", expanded=False):
                st.text_area(
                    "De-identified conversation text:",
                    value=st.session_state.deidentified_conversation,
                    height=200,
                    disabled=True
                )
        
        st.divider()
        
        # Action buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🖨️ Print SOAP", use_container_width=True):
                st.success("📄 SOAP note sent to printer!")
        
        with col2:
            st.download_button(
                label="📥 Download SOAP",
                data=st.session_state.soap_note,
                file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Generate LaTeX version for professional formatting
            escaped_soap_note = escape_latex(st.session_state.soap_note)
            latex_soap = f"""
\\documentclass[a4paper,11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{parskip}}
\\setlength{{\\parskip}}{{0.5em}}
\\usepackage{{titlesec}}
\\titleformat{{\\section}}{{\\large\\bfseries}}{{\\thesection}}{{1em}}{{}}
\\begin{{document}}
\\begin{{center}}
{{\\LARGE\\bfseries SOAP NOTE}} \\\\
\\vspace{{0.5em}}
{{\\large Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}
\\end{{center}}
\\begin{{quote}}
\\textit{{Note: Personal identifying information has been removed for privacy.}}
\\end{{quote}}
\\section*{{Clinical Documentation}}
{escaped_soap_note}
\\end{{document}}
"""
            st.download_button(
                label="📥 Download PDF",
                data=latex_soap,
                file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                mime="text/latex",
                use_container_width=True
            )
        
        with col4:
            if st.button("📧 Email SOAP", use_container_width=True):
                st.success("📧 SOAP note emailed!")
        
        with col5:
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("🔄 Regenerating SOAP note..."):
                    conversation_text = format_conversation_for_soap(st.session_state.transcription)
                    soap_note = generate_soap_note(conversation_text, st.session_state.patient_info)
                    if soap_note:
                        st.session_state.soap_note = soap_note
                        st.success("✅ SOAP note regenerated!")
                        st.rerun()
        
        st.divider()
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Conversation", use_container_width=True):
                st.session_state.current_stage = 'conversation'
                st.rerun()
        
        with col2:
            if st.button("🔍 Continue to Diagnostic", type="primary", use_container_width=True):
                st.session_state.current_stage = 'diagnostic'
                st.rerun()
    
    else:
        st.error("❌ No SOAP note available. Please generate one first.")
        if st.button("⬅️ Back to Conversation"):
            st.session_state.current_stage = 'conversation'
            st.rerun()

def render_diagnostic_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>🔍 Diagnostic Suggestions</h2>
        <p>AI-powered diagnostic suggestions based on conversation analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show SOAP note access button
    if st.session_state.soap_note:
        with st.expander("📋 View SOAP Note", expanded=False):
            st.text_area("Generated SOAP Note:", value=st.session_state.soap_note, height=300, disabled=True)
    
    st.subheader("🎯 AI Diagnostic Suggestions")
    
    for i, disease in enumerate(st.session_state.disease_suggestions):
        disease_id = get_disease_unique_id(disease['disease'])
        with st.expander(f"🔸 {disease['disease']} (Confidence: {disease['score']:.0%})", expanded=i<2):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Reasoning:** {disease['reason']}")
            with col2:
                st.metric("Confidence", f"{disease['score']:.0%}")
            
            is_selected = st.checkbox(f"Select {disease['disease']}", key=f"disease_{disease_id}_{i}")
            if is_selected and disease['disease'] not in st.session_state.selected_diseases:
                st.session_state.selected_diseases.append(disease['disease'])
            elif not is_selected and disease['disease'] in st.session_state.selected_diseases:
                st.session_state.selected_diseases.remove(disease['disease'])
    
    st.divider()
    
    # Add custom diagnosis
    st.subheader("➕ Add Custom Diagnosis")
    custom_disease = st.text_input("Enter additional diagnosis:", placeholder="e.g., Secondary headache syndrome")
    if st.button("Add Diagnosis") and custom_disease:
        if custom_disease not in st.session_state.selected_diseases:
            st.session_state.selected_diseases.append(custom_disease)
            st.success(f"✅ Added: {custom_disease}")
    
    # Show selected diagnoses
    if st.session_state.selected_diseases:
        st.subheader("📝 Selected Diagnoses")
        for i, disease in enumerate(st.session_state.selected_diseases):
            disease_id = get_disease_unique_id(disease)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {disease}")
            with col2:
                if st.button("❌", key=f"remove_disease_{disease_id}_{i}"):
                    st.session_state.selected_diseases.remove(disease)
                    st.rerun()
    
    st.divider()
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back to Conversation", use_container_width=True):
            st.session_state.current_stage = 'conversation'
            st.rerun()
    
    with col2:
        if st.session_state.soap_note:
            if st.button("📋 View SOAP Note", use_container_width=True):
                st.session_state.current_stage = 'soap'
                st.rerun()
        else:
            if st.button("📋 Generate SOAP Note", use_container_width=True):
                with st.spinner("🔄 Generating SOAP note..."):
                    conversation_text = format_conversation_for_soap(st.session_state.transcription)
                    soap_note = generate_soap_note(conversation_text, st.session_state.patient_info)
                    if soap_note:
                        st.session_state.soap_note = soap_note
                        st.success("✅ SOAP note generated!")
                        st.session_state.current_stage = 'soap'
                        st.rerun()
    
    with col3:
        if st.button("💊 Generate Medicines", type="primary", use_container_width=True):
            # Regenerate medicine suggestions based on selected diseases
            st.session_state.medicine_suggestions = generate_medicine_suggestions(st.session_state.disease_suggestions)
            st.session_state.drug_alerts = analyze_drug_interactions(st.session_state.medicine_suggestions, st.session_state.patient_info)
            st.session_state.current_stage = 'medicines'
            st.rerun()

def render_medicines_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>💊 Medicine Recommendations</h2>
        <p>AI-suggested medications based on selected diagnoses.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show selected diagnoses
    if st.session_state.selected_diseases:
        st.info(f"**Selected Diagnoses:** {', '.join(st.session_state.selected_diseases)}")
    
    st.subheader("💊 Recommended Medications")
    
    for i, medicine in enumerate(st.session_state.medicine_suggestions):
        medicine_id = get_medicine_unique_id(medicine['medicine'])
        with st.expander(f"💉 {medicine['medicine']} (Score: {medicine['score']:.0%})", expanded=i<2):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Recommendation Reason:** {medicine['reason']}")
                st.write(f"**Purpose:** {medicine['purpose'] or 'Not available'}")
                st.write(f"**Side Effects:** {medicine['side_effects'] or 'Not available'}")
            with col2:
                st.metric("Recommendation Score", f"{medicine['score']:.0%}")
                is_selected = st.checkbox(f"Select {medicine['medicine']}", key=f"med_{medicine_id}_{i}")
                if is_selected and medicine['medicine'] not in [m['medicine'] for m in st.session_state.selected_medicines]:
                    st.session_state.selected_medicines.append(medicine)
                elif not is_selected and medicine['medicine'] in [m['medicine'] for m in st.session_state.selected_medicines]:
                    st.session_state.selected_medicines = [m for m in st.session_state.selected_medicines if m['medicine'] != medicine['medicine']]
    
    st.divider()
    
    # Add custom medication
    st.subheader("➕ Add Custom Medication")
    custom_medicine = st.text_input("Enter additional medication:", placeholder="e.g., Topiramate")
    if st.button("Add Medication") and custom_medicine:
        basic_med = {"medicine": custom_medicine, "score": 0.0, "reason": "User added"}
        enriched_med = enrich_medicines_with_fda_data([basic_med])[0]
        if enriched_med['medicine'] not in [m['medicine'] for m in st.session_state.selected_medicines]:
            st.session_state.selected_medicines.append(enriched_med)
            st.success(f"✅ Added: {custom_medicine}")
    
    # Show selected medications
    if st.session_state.selected_medicines:
        st.subheader("📝 Selected Medications")
        for i, medicine in enumerate(st.session_state.selected_medicines):
            medicine_id = get_medicine_unique_id(medicine['medicine'])
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {medicine['medicine']}")
            with col2:
                if st.button("❌", key=f"remove_med_{medicine_id}_{i}"):
                    st.session_state.selected_medicines = [m for m in st.session_state.selected_medicines if m['medicine'] != medicine['medicine']]
                    st.rerun()
    
    st.divider()
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back to Diagnostic", use_container_width=True):
            st.session_state.current_stage = 'diagnostic'
            st.rerun()
    
    with col2:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.medicine_suggestions = generate_medicine_suggestions(st.session_state.disease_suggestions)
            st.session_state.drug_alerts = analyze_drug_interactions(st.session_state.medicine_suggestions, st.session_state.patient_info)
            st.rerun()
    
    with col3:
        if st.button("⚠️ Check Drug Alerts", type="primary", use_container_width=True, disabled=not st.session_state.selected_medicines):
            st.session_state.drug_alerts = analyze_drug_interactions(st.session_state.selected_medicines, st.session_state.patient_info)
            st.session_state.prescription_confirmed = False  # Reset confirmation
            st.session_state.current_stage = 'alerts'
            st.rerun()

def render_alerts_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>⚠️ Drug Interaction & Safety Alerts</h2>
        <p>Safety analysis of selected medications for interactions and contraindications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    risk_colors = {
        "HIGH": "🔴", 
        "MODERATE": "🟡", 
        "LOW": "🟢",
        "LOW to MODERATE": "🟡",
        "MODERATE to HIGH": "🔴",
        "UNKNOWN": "⚪",
        "MINIMAL": "🟢",
        "SEVERE": "🔴"
    }
    
    overall_risk = st.session_state.drug_alerts.overall_risk_level
    risk_icon = risk_colors.get(overall_risk, "🟡")
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0; color: black;">
        <h3>{risk_icon} Overall Risk Level: {overall_risk}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show selected medications being analyzed
    st.subheader("💊 Selected Medications Being Analyzed")
    for med in st.session_state.selected_medicines:
        st.write(f"• {med['medicine']}")
    
    st.divider()
    
    # Show alerts
    if st.session_state.drug_alerts.alerts:
        st.subheader("🚨 Safety Alerts")
        for i, alert in enumerate(st.session_state.drug_alerts.alerts):
            severity_class = f"alert-{alert.severity.lower().replace(' ', '-').replace('to', '').strip()}"
            severity_icons = {
                "HIGH": "🔴", 
                "MODERATE": "🟡", 
                "LOW": "🟢",
                "LOW to MODERATE": "🟡",
                "MODERATE to HIGH": "🔴",
                "UNKNOWN": "⚪",
                "MINIMAL": "🟢",
                "SEVERE": "🔴"
            }
            severity_icon = severity_icons.get(alert.severity, "🟡")
            st.markdown(f"""
            <div class="conversation-item {severity_class}">
                <h4>{severity_icon} {alert.severity} - {alert.alert_type.replace('_', ' ').title()}</h4>
                <p><strong>Drugs Involved:</strong> {alert.drug1}{f" & {alert.drug2}" if alert.drug2 else ""}</p>
                <p><strong>Description:</strong> {alert.description}</p>
                <p><strong>Recommendation:</strong> {alert.recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical drug interactions or safety alerts detected!")
    
    # Show safe combinations
    if st.session_state.drug_alerts.safe_combinations:
        st.subheader("✅ Safe Combinations")
        for combo in st.session_state.drug_alerts.safe_combinations:
            st.write(f"• {combo}")
    
    st.divider()
    
    # Navigation and prescription generation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Edit Medicines", use_container_width=True):
            st.session_state.current_stage = 'medicines'
            st.session_state.prescription_confirmed = False  # Reset confirmation
            st.rerun()
    
    with col2:
        if st.button("🔄 Reanalyze", use_container_width=True):
            st.session_state.drug_alerts = analyze_drug_interactions(st.session_state.selected_medicines, st.session_state.patient_info)
            st.session_state.prescription_confirmed = False  # Reset confirmation
            st.rerun()
    
    with col3:
        high_risk_alerts = [a for a in st.session_state.drug_alerts.alerts if a.severity == "HIGH"]
        
        if high_risk_alerts:
            st.warning("⚠️ High-risk alerts detected! Please confirm you want to proceed.")
            if st.button("✅ Continue to Suggestions", key="continue_to_suggestions_with_alerts", use_container_width=True):
                st.session_state.prescription_confirmed = True
                st.session_state.current_stage = 'suggestions'
                st.rerun()
        else:
            if st.button("💡 Continue to Suggestions", type="primary", use_container_width=True):
                st.session_state.prescription_confirmed = True
                st.session_state.current_stage = 'suggestions'
                st.rerun()

def render_suggestions_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>💡 Suggestions Engine</h2>
        <p>Review and customize suggestions for the patient.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract conversation suggestions if not already done
    if not st.session_state.conversation_suggestions and st.session_state.patient_info.get('Follow_Up_Actions'):
        st.session_state.conversation_suggestions = st.session_state.patient_info.get('Follow_Up_Actions', [])
    
    # Generate AI suggestions if not already done
    if not st.session_state.ai_suggestions:
        with st.spinner("🔄 Generating AI-based suggestions..."):
            st.session_state.ai_suggestions = generate_ai_suggestions(
                st.session_state.selected_medicines,
                st.session_state.patient_info,
                st.session_state.selected_diseases
            )
    
    # Display conversation suggestions
    st.subheader("📝 Suggestions from Conversation")
    if st.session_state.conversation_suggestions:
        for i, suggestion in enumerate(st.session_state.conversation_suggestions):
            is_selected = st.checkbox(
                suggestion, 
                key=f"conv_suggestion_{i}",
                value=suggestion in st.session_state.selected_suggestions
            )
            if is_selected and suggestion not in st.session_state.selected_suggestions:
                st.session_state.selected_suggestions.append(suggestion)
            elif not is_selected and suggestion in st.session_state.selected_suggestions:
                st.session_state.selected_suggestions.remove(suggestion)
    else:
        st.info("No suggestions found in the conversation.")
    
    st.divider()
    
    # Display AI suggestions
    st.subheader("🤖 AI-Based Suggestions")
    if st.session_state.ai_suggestions:
        for i, suggestion in enumerate(st.session_state.ai_suggestions):
            is_selected = st.checkbox(
                suggestion, 
                key=f"ai_suggestion_{i}",
                value=suggestion in st.session_state.selected_suggestions
            )
            if is_selected and suggestion not in st.session_state.selected_suggestions:
                st.session_state.selected_suggestions.append(suggestion)
            elif not is_selected and suggestion in st.session_state.selected_suggestions:
                st.session_state.selected_suggestions.remove(suggestion)
    else:
        st.info("No AI suggestions available.")
    
    st.divider()
    
    # Add custom suggestions
    st.subheader("➕ Add Custom Suggestions")
    custom_suggestion = st.text_area(
        "Enter a custom suggestion:",
        placeholder="e.g., Avoid driving while taking this medication",
        key="custom_suggestion_input"
    )
    
    if st.button("Add Suggestion") and custom_suggestion:
        if custom_suggestion not in st.session_state.custom_suggestions:
            st.session_state.custom_suggestions.append(custom_suggestion)
            st.success(f"✅ Added: {custom_suggestion}")
            # Clear the input
            st.session_state.custom_suggestion_input = ""
    
    # Display custom suggestions
    if st.session_state.custom_suggestions:
        st.write("**Your Custom Suggestions:**")
        for i, suggestion in enumerate(st.session_state.custom_suggestions):
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = st.checkbox(
                    suggestion, 
                    key=f"custom_suggestion_{i}",
                    value=suggestion in st.session_state.selected_suggestions
                )
                if is_selected and suggestion not in st.session_state.selected_suggestions:
                    st.session_state.selected_suggestions.append(suggestion)
                elif not is_selected and suggestion in st.session_state.selected_suggestions:
                    st.session_state.selected_suggestions.remove(suggestion)
            with col2:
                if st.button("❌", key=f"remove_custom_{i}"):
                    st.session_state.custom_suggestions.remove(suggestion)
                    if suggestion in st.session_state.selected_suggestions:
                        st.session_state.selected_suggestions.remove(suggestion)
                    st.rerun()
    
    st.divider()
    
    # Show selected suggestions
    if st.session_state.selected_suggestions:
        st.subheader("✅ Selected Suggestions")
        for suggestion in st.session_state.selected_suggestions:
            st.write(f"• {suggestion}")
    else:
        st.warning("No suggestions selected. Please select at least one suggestion.")
    
    st.divider()
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Back to Drug Alerts", use_container_width=True):
            st.session_state.current_stage = 'alerts'
            st.rerun()
    
    with col2:
        if st.button("🔄 Regenerate AI Suggestions", use_container_width=True):
            st.session_state.ai_suggestions = []
            st.rerun()
    
    with col3:
        if st.button("📄 Generate Prescription", type="primary", use_container_width=True, 
                    disabled=not st.session_state.selected_suggestions):
            generate_final_prescription()
            st.session_state.current_stage = 'prescription'
            st.rerun()

def render_prescription_stage():
    st.markdown("""
    <div class="stage-card">
        <h2>📄 Final Prescription</h2>
        <p>Generated prescription ready for review and export.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.final_prescription:
        # Update prescription with latest patient information from session state
        prescription = st.session_state.final_prescription.copy()
        prescription.update({
            'patient_name': st.session_state.patient_info.get('Patient_Name', 'Not provided'),
            'age': st.session_state.patient_info.get('Age', 'N/A'),
            'gender': st.session_state.patient_info.get('Gender', 'N/A'),
            'medical_history': st.session_state.patient_info.get('Previous_Medical_History', 'Not provided'),
            'previous_drug_history': st.session_state.patient_info.get('Previous_Drug_History', 'Not provided'),
            'lifestyle_details': st.session_state.patient_info.get('Lifestyle_Details', {}),
            'doctor_suggestions': st.session_state.selected_suggestions,
            'medical_tests_advised': st.session_state.patient_info.get('Current_Medical_Tests_Ordered', 'None advised')
        })
        
        # Display prescription
        medicines_text = '<br>'.join([f"• {med['medicine']}" for med in prescription['medicines']]) if prescription['medicines'] else 'No medications prescribed'
        lifestyle_text = '<br>'.join([f"{key}: {value}" for key, value in prescription['lifestyle_details'].items()]) if prescription.get('lifestyle_details') else 'Not provided'
        doctor_suggestions_text = '<br>'.join([f"• {suggestion}" for suggestion in prescription['doctor_suggestions']]) if prescription.get('doctor_suggestions') else 'Follow up as needed'
        medical_tests_text = prescription['medical_tests_advised'] if prescription.get('medical_tests_advised') else 'None advised'
        
        st.markdown(f"""
        <div class="prescription-card">
            <h2>🏥 MEDICAL PRESCRIPTION</h2>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                <div>
                    <strong>Patient:</strong> {prescription.get('patient_name', 'Not provided')}<br>
                    <strong>Age:</strong> {prescription.get('age', 'N/A')}<br>
                    <strong>Gender:</strong> {prescription.get('gender', 'N/A')}
                </div>
                <div>
                    <strong>Date:</strong> {prescription.get('date', datetime.now().strftime('%Y-%m-%d'))}<br>
                    <strong>Doctor:</strong> Dr. [Sinthiya Chowdhury]<br>
                    <strong>License:</strong> [MBBS, Jalalabad Ragib Rabeya Medical Hospital, Sylhet]
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <h3>📋 DIAGNOSIS</h3>
            {' • '.join(prescription['diseases']) if prescription.get('diseases') else 'No specific diagnosis'}
            <h3>💊 PRESCRIBED MEDICATIONS</h3>
            {medicines_text}
            <h3>🏥 MEDICAL HISTORY</h3>
            {prescription.get('medical_history', 'Not provided') or 'Not provided'}
            <h3>💉 PREVIOUS DRUG HISTORY</h3>
            {prescription.get('previous_drug_history', 'Not provided') or 'Not provided'}
            <h3>🏃 LIFESTYLE DETAILS</h3>
            {lifestyle_text}
            <h3>👨‍⚕️ DOCTOR'S SUGGESTIONS</h3>
            {doctor_suggestions_text}
            <h3>🩺 ADVISED MEDICAL TESTS</h3>
            {medical_tests_text}
            <h3>⚠️ IMPORTANT NOTES</h3>
            • Follow dosage instructions carefully<br>
            • Complete the full course of medication<br>
            • Contact doctor if side effects occur<br>
            • Follow-up appointment recommended in 2 weeks
        </div>
        """, unsafe_allow_html=True)
        
        # Show SOAP note if available
        if prescription.get('soap_note'):
            with st.expander("📋 Associated SOAP Note", expanded=False):
                st.text_area(
                    "SOAP Note:",
                    value=prescription['soap_note'],
                    height=300,
                    disabled=True
                )
        
        # Show complete patient information
        with st.expander("📋 Complete Patient Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Medical History:**")
                st.write(st.session_state.patient_info.get('Previous_Medical_History', 'Not provided'))
                st.write("**Allergies:**")
                st.write(st.session_state.patient_info.get('Allergies', 'None reported'))
            with col2:
                st.write("**Family History:**")
                st.write(st.session_state.patient_info.get('Family_Medical_History', 'Not provided'))
                st.write("**Previous Medications:**")
                st.write(st.session_state.patient_info.get('Previous_Drug_History', 'Not provided'))
        
        # Show safety alerts summary
        if prescription.get('alerts'):
            with st.expander("⚠️ Safety Alerts Summary", expanded=True):
                for alert in prescription['alerts']:
                    severity_icons = {
                        "HIGH": "🔴", 
                        "MODERATE": "🟡", 
                        "LOW": "🟢",
                        "LOW to MODERATE": "🟡",
                        "MODERATE to HIGH": "🔴",
                        "UNKNOWN": "⚪",
                        "MINIMAL": "🟢",
                        "SEVERE": "🔴"
                    }
                    severity_icon = severity_icons.get(alert['severity'], "🟡")
                    st.warning(f"{severity_icon} **{alert['severity']}:** {alert['description']} - {alert['recommendation']}")
        
        st.divider()
        
        # Action buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🖨️ Print Prescription", use_container_width=True):
                st.success("📄 Prescription sent to printer!")
        
        with col2:
            # Update the prescription with latest info before downloading
            updated_prescription = prescription.copy()
            prescription_json = json.dumps(updated_prescription, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=prescription_json,
                file_name=f"prescription_{prescription.get('patient_name', 'patient')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Generate LaTeX for PDF with updated info
            diagnoses_text = '\n\\newline\n'.join([f"• {escape_latex(d)}" for d in prescription.get('diseases', [])]) if prescription.get('diseases') else 'No specific diagnosis'
            medicines_latex = '\n'.join([f"\\item {escape_latex(med['medicine'])}" for med in prescription.get('medicines', [])]) if prescription.get('medicines') else r'\item No medications prescribed'
            lifestyle_latex = '\n'.join([f"\\item {key}: {escape_latex(value)}" for key, value in prescription.get('lifestyle_details', {}).items()]) if prescription.get('lifestyle_details') else r'\item Not provided'
            doctor_suggestions_latex = '\n'.join([f"\\item {escape_latex(suggestion)}" for suggestion in prescription.get('doctor_suggestions', [])]) if prescription.get('doctor_suggestions') else r'\item Follow up as needed'
            medical_history_latex = escape_latex(str(prescription.get('medical_history', 'Not provided')))
            previous_drug_history_latex = escape_latex(str(prescription.get('previous_drug_history', 'Not provided')))
            medical_tests_latex = escape_latex(str(prescription.get('medical_tests_advised', 'None advised')))
            
            # Convert age to string before passing to escape_latex
            patient_age = str(prescription.get('age', 'Not provided'))
            patient_name = str(prescription.get('patient_name', 'Not provided'))
            patient_gender = str(prescription.get('gender', 'Not provided'))
            prescription_date = str(prescription.get('date', datetime.now().strftime('%Y-%m-%d')))
            
            latex_content = f"""
\\documentclass[a4paper,11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{parskip}}
\\setlength{{\\parskip}}{{0.5em}}
\\usepackage{{enumitem}}
\\setlist[itemize]{{leftmargin=*}}
\\usepackage{{titlesec}}
\\titleformat{{\\section}}{{\\large\\bfseries}}{{\\thesection}}{{1em}}{{}}
\\usepackage{{xcolor}}
\\definecolor{{headerblue}}{{RGB}}{{0,51,102}}
\\begin{{document}}
\\begin{{center}}
{{\\LARGE\\bfseries MEDICAL PRESCRIPTION}} \\\\
\\vspace{{0.5em}}
{{\\color{{headerblue}}\\rule{{0.5\\textwidth}}{{1pt}}}}
\\end{{center}}
\\begin{{minipage}}[t]{{0.45\\textwidth}}
\\textbf{{Patient:}} {escape_latex(patient_name)} \\\\
\\textbf{{Age:}} {escape_latex(patient_age)} \\\\
\\textbf{{Gender:}} {escape_latex(patient_gender)}
\\end{{minipage}}
\\begin{{minipage}}[t]{{0.45\\textwidth}}
\\textbf{{Date:}} {escape_latex(prescription_date)} \\\\
\\textbf{{Doctor:}} Dr. [Sinthiya Chowdhury] \\\\
\\textbf{{Designation:}} [MBBS, Jalalabad Ragib Rabeya Medical Hospital Sylhet]\\\\
\\textbf{{License:}} [ABC123333456]
\\end{{minipage}}
\\section*{{Diagnosis}}
{diagnoses_text}
\\section*{{Prescribed Medications}}
\\begin{{itemize}}
{medicines_latex}
\\end{{itemize}}
\\section*{{Medical History}}
{medical_history_latex}
\\section*{{Previous Drug History}}
{previous_drug_history_latex}
\\section*{{Lifestyle Details}}
\\begin{{itemize}}
{lifestyle_latex}
\\end{{itemize}}
\\section*{{Doctor's Suggestions}}
\\begin{{itemize}}
{doctor_suggestions_latex}
\\end{{itemize}}
\\section*{{Advised Medical Tests}}
{medical_tests_latex}
\\section*{{Important Notes}}
\\begin{{itemize}}
\\item Follow dosage instructions carefully
\\item Complete the full course of medication
\\item Contact doctor if side effects occur
\\item Follow-up appointment recommended in 2 weeks
\\end{{itemize}}
\\end{{document}}
"""
            st.download_button(
                label="📥 Download PDF",
                data=latex_content,
                file_name=f"prescription_{escape_latex(patient_name)}_{datetime.now().strftime('%Y%m%d')}.tex",
                mime="text/latex",
                use_container_width=True
            )
        
        with col4:
            if st.button("📧 Email Patient", use_container_width=True):
                st.success("📧 Prescription emailed to patient!")
        
        with col5:
            if st.button("🔄 New Consultation", use_container_width=True):
                # Reset all session state except current_stage
                for key in list(st.session_state.keys()):
                    if key not in ['current_stage']:
                        del st.session_state[key]
                st.session_state.current_stage = 'upload'
                st.rerun()
        
        # Consultation summary
        st.subheader("📊 Consultation Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Diagnoses", len(prescription.get('diseases', [])))
        
        with col2:
            st.metric("Medications", len(prescription.get('medicines', [])))
        
        with col3:
            st.metric("Safety Alerts", len(prescription.get('alerts', [])))
        
        with col4:
            # Calculate actual consultation duration from timestamps
            consultation_time = calculate_consultation_duration(st.session_state.transcription)
            st.metric("Consultation Duration", f"{consultation_time:.1f} min")

def enhanced_main():
    initialize_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏥 Medical Diagnostic & Prescription System
        </h1>
        <p style="color: #b3d9ff; text-align: center; margin: 0.5rem 0 0 0;">
            AI-Powered Medical Consultation Assistant with SOAP Notes v3.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        
        stages = {
            'upload': ('🎤', 'Audio Upload'),
            'conversation': ('💬', 'Conversation'),
            'soap': ('📋', 'SOAP Note'),
            'diagnostic': ('🔍', 'Diagnostic'),
            'medicines': ('💊', 'Medicines'),
            'alerts': ('⚠️', 'Drug Alerts'),
            'suggestions': ('💡', 'Suggestions'),
            'prescription': ('📄', 'Prescription')
        }
        
        current_stage_index = list(stages.keys()).index(st.session_state.current_stage)
        
        for i, (stage_key, (icon, stage_name)) in enumerate(stages.items()):
            if i <= current_stage_index or (stage_key == 'soap' and st.session_state.soap_note):
                if st.session_state.current_stage == stage_key:
                    st.markdown(f"**➤ {icon} {stage_name}**")
                else:
                    st.markdown(f"✅ {icon} {stage_name}")
            else:
                st.markdown(f"⏳ {icon} {stage_name}")
        
        # Progress calculation (SOAP note is optional)
        completed_stages = current_stage_index + 1
        if st.session_state.soap_note and st.session_state.current_stage != 'soap':
            completed_stages += 0.5  # Bonus for having SOAP note
        
        progress = min(completed_stages / len(stages), 1.0)
        st.progress(progress)
        st.caption(f"Progress: {progress:.0%}")
        
        st.divider()
        
        # Patient Information Panel
        st.header("👤 Patient Information")
        
        patient_info = {}
        patient_info['name'] = st.text_input(
            "Patient Name", 
            value=st.session_state.patient_info.get('Patient_Name', ''), 
            key="sidebar_patient_name"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            patient_info['age'] = st.number_input(
                "Age", 
                min_value=0, 
                max_value=120, 
                value=st.session_state.patient_info.get('Age', 0), 
                key="sidebar_patient_age"
            )
        
        with col2:
            patient_info['gender'] = st.selectbox(
                "Gender", 
                ['Male', 'Female', 'Other'], 
                index=['Male', 'Female', 'Other'].index(st.session_state.patient_info.get('Gender', 'Male')) 
                if st.session_state.patient_info.get('Gender') else 0, 
                key="sidebar_patient_gender"
            )
        
        # Contact Information
        with st.expander("📞 Contact Information"):
            patient_info['phone'] = st.text_input(
                "Phone", 
                value=st.session_state.patient_info.get('phone', ''), 
                key="sidebar_patient_phone"
            )
            patient_info['email'] = st.text_input(
                "Email", 
                value=st.session_state.patient_info.get('email', ''), 
                key="sidebar_patient_email"
            )
            patient_info['emergency_contact'] = st.text_input(
                "Emergency Contact", 
                value=st.session_state.patient_info.get('emergency_contact', ''), 
                key="sidebar_emergency_contact"
            )
        
        # Medical Information
        with st.expander("🏥 Medical Information"):
            patient_info['medical_history'] = st.text_area(
                "Medical History", 
                value=st.session_state.patient_info.get('Previous_Medical_History', ''), 
                key="sidebar_medical_history"
            )
            patient_info['drug_history'] = st.text_area(
                "Previous Drug History", 
                value=st.session_state.patient_info.get('Previous_Drug_History', ''), 
                key="sidebar_drug_history"
            )
            patient_info['allergies'] = st.text_area(
                "Allergies", 
                value=st.session_state.patient_info.get('Allergies', ''), 
                key="sidebar_allergies"
            )
            patient_info['family_history'] = st.text_area(
                "Family History", 
                value=st.session_state.patient_info.get('Family_Medical_History', ''), 
                key="sidebar_family_history"
            )
        
        # Lifestyle Information
        with st.expander("🏃 Lifestyle Information"):
            current_lifestyle = st.session_state.patient_info.get('Lifestyle_Details', {})
            
            patient_info['smoking'] = st.selectbox(
                "Smoking Status", 
                ['Never', 'Former', 'Current'], 
                index=['Never', 'Former', 'Current'].index(current_lifestyle.get('smoking', 'Never')) 
                if current_lifestyle.get('smoking') in ['Never', 'Former', 'Current'] else 0, 
                key="sidebar_smoking"
            )
            
            patient_info['alcohol'] = st.selectbox(
                "Alcohol Use", 
                ['None', 'Occasional', 'Regular', 'Heavy'], 
                index=['None', 'Occasional', 'Regular', 'Heavy'].index(current_lifestyle.get('alcohol', 'None')) 
                if current_lifestyle.get('alcohol') in ['None', 'Occasional', 'Regular', 'Heavy'] else 0, 
                key="sidebar_alcohol"
            )
            
            patient_info['exercise'] = st.selectbox(
                "Exercise Frequency", 
                ['None', 'Rarely', 'Weekly', 'Daily'], 
                index=['None', 'Rarely', 'Weekly', 'Daily'].index(current_lifestyle.get('exercise', 'None')) 
                if current_lifestyle.get('exercise') in ['None', 'Rarely', 'Weekly', 'Daily'] else 0, 
                key="sidebar_exercise"
            )
            
            patient_info['Lifestyle_Details'] = {
                'smoking': patient_info['smoking'], 
                'alcohol': patient_info['alcohol'], 
                'exercise': patient_info['exercise']
            }
        
        # Update session state with sidebar inputs
        st.session_state.patient_info.update(patient_info)
        
        # System Information
        st.divider()
        st.header("ℹ️ System Info")
        
        if st.session_state.transcription:
            st.success(f"✅ Audio processed: {len(st.session_state.transcription)} segments")
        
        if st.session_state.soap_note:
            st.success("✅ SOAP note generated")
        
        if st.session_state.selected_diseases:
            st.success(f"✅ {len(st.session_state.selected_diseases)} diagnoses selected")
        
        if st.session_state.selected_medicines:
            st.success(f"✅ {len(st.session_state.selected_medicines)} medicines selected")
        
        if st.session_state.selected_suggestions:
            st.success(f"✅ {len(st.session_state.selected_suggestions)} suggestions selected")
    
    # Main content area
    if st.session_state.current_stage == 'upload':
        render_upload_stage()
    elif st.session_state.current_stage == 'conversation':
        render_conversation_stage()
    elif st.session_state.current_stage == 'soap':
        render_soap_stage()
    elif st.session_state.current_stage == 'diagnostic':
        render_diagnostic_stage()
    elif st.session_state.current_stage == 'medicines':
        render_medicines_stage()
    elif st.session_state.current_stage == 'alerts':
        render_alerts_stage()
    elif st.session_state.current_stage == 'suggestions':
        render_suggestions_stage()
    elif st.session_state.current_stage == 'prescription':
        render_prescription_stage()

if __name__ == "__main__":
    enhanced_main()