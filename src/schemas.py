"""
Pydantic Models and Schemas for Medical Diagnostic System
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class PatientInformation(BaseModel):
    """Patient information extracted from conversation"""
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
    """Single disease suggestion with confidence score"""
    disease: str
    score: float
    reason: str


class DiagnosisSuggestions(BaseModel):
    """Collection of diagnosis suggestions"""
    suggestions: List[DiseaseSuggestion]


class MedicineSuggestion(BaseModel):
    """Single medicine suggestion with metadata"""
    medicine: str
    score: float
    reason: str
    purpose: Optional[str] = None
    side_effects: Optional[str] = None


class MedicineSuggestions(BaseModel):
    """Collection of medicine suggestions"""
    suggestions: List[MedicineSuggestion]


class DrugAlert(BaseModel):
    """Drug interaction or safety alert"""
    alert_type: str
    severity: str
    drug1: str
    drug2: Optional[str] = None
    description: str
    recommendation: str


class DrugAlertAnalysis(BaseModel):
    """Complete drug safety analysis"""
    alerts: List[DrugAlert]
    safe_combinations: List[str] = []
    overall_risk_level: str


class DrugInfo(BaseModel):
    """Drug information from external APIs"""
    drug_name: str
    purpose: Optional[str] = None
    side_effects: Optional[str] = None
    contraindications: Optional[str] = None
    warnings: Optional[str] = None
    error: Optional[str] = None


class DrugInteraction(BaseModel):
    """Drug-drug interaction information"""
    drug1: str
    drug2: str
    description: str
    severity: str
    source: str = "RxNorm"


class TranscriptionSegment(BaseModel):
    """Single segment of transcribed audio"""
    speaker: str
    text: str
    timestamp: str


class Prescription(BaseModel):
    """Final prescription document"""
    patient_name: str
    age: Any  # Can be int or str
    gender: str
    date: str
    diseases: List[str]
    medicines: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    medical_history: str
    previous_drug_history: str
    lifestyle_details: Dict[str, str]
    doctor_suggestions: List[str]
    medical_tests_advised: str
    soap_note: Optional[str] = None