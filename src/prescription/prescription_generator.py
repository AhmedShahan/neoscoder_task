"""
Prescription Generator Module
Creates final prescription documents
"""
from datetime import datetime
from typing import List, Dict, Optional
from ..schemas import Prescription


class PrescriptionGenerator:
    """Generates final prescription documents"""
    
    def generate(
        self,
        patient_info: Dict,
        selected_diseases: List[str],
        selected_medicines: List[Dict],
        drug_alerts: Dict,
        selected_suggestions: List[str],
        soap_note: Optional[str] = None
    ) -> Prescription:
        """Generate complete prescription"""
        
        prescription = Prescription(
            patient_name=patient_info.get('Patient_Name', 'Not provided'),
            age=patient_info.get('Age', 'N/A'),
            gender=patient_info.get('Gender', 'N/A'),
            date=datetime.now().strftime('%Y-%m-%d'),
            diseases=selected_diseases,
            medicines=selected_medicines,
            alerts=drug_alerts.get('alerts', []),
            medical_history=patient_info.get('Previous_Medical_History', 'Not provided'),
            previous_drug_history=patient_info.get('Previous_Drug_History', 'Not provided'),
            lifestyle_details=patient_info.get('Lifestyle_Details', {}),
            doctor_suggestions=selected_suggestions,
            medical_tests_advised=patient_info.get('Current_Medical_Tests_Ordered', 'None advised'),
            soap_note=soap_note
        )
        
        return prescription