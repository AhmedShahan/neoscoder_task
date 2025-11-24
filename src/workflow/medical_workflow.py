"""
Medical Workflow Orchestrator
Main workflow that coordinates all components
"""
from typing import Dict, List, Optional
from pathlib import Path

from ..transcription.transcription_service import TranscriptionService
from ..extraction.patient_info_extractor import PatientInfoExtractor
from ..extraction.deidentification import Deidentifier
from ..medical_suggestion.diagnosis_engine import DiagnosisEngine
from ..medical_suggestion.suggestion_generator import SuggestionGenerator
from ..drug_suggestion.medicine_engine import MedicineEngine
from ..medical_alert.interaction_analyzer import InteractionAnalyzer
from ..prescription.soap_note_generator import SOAPNoteGenerator
from ..prescription.prescription_generator import PrescriptionGenerator
from ..prescription.document_exporter import DocumentExporter
from ..schemas import Prescription


class MedicalWorkflow:
    """
    Orchestrates the complete medical diagnostic and prescription workflow
    """
    
    def __init__(self):
        """Initialize all services"""
        self.transcription_service = TranscriptionService()
        self.patient_extractor = PatientInfoExtractor()
        self.deidentifier = Deidentifier()
        self.diagnosis_engine = DiagnosisEngine()
        self.suggestion_generator = SuggestionGenerator()
        self.medicine_engine = MedicineEngine()
        self.interaction_analyzer = InteractionAnalyzer()
        self.soap_generator = SOAPNoteGenerator()
        self.prescription_generator = PrescriptionGenerator()
        self.document_exporter = DocumentExporter()
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process uploaded audio file
        
        Returns:
            Dictionary with transcription and patient info
        """
        # Step 1: Transcribe audio
        transcription = self.transcription_service.transcribe_file(audio_path)
        
        # Step 2: Extract patient information
        patient_info = self.patient_extractor.extract(transcription)
        
        return {
            'transcription': transcription,
            'patient_info': patient_info
        }
    
    def generate_diagnoses(self, patient_info: Dict) -> List[Dict]:
        """Generate diagnostic suggestions"""
        return self.diagnosis_engine.generate_suggestions(patient_info)
    
    def generate_medicines(
        self, 
        patient_info: Dict,
        selected_diseases: List[str]
    ) -> List[Dict]:
        """Generate medicine suggestions"""
        return self.medicine_engine.generate_suggestions(patient_info, selected_diseases)
    
    def analyze_safety(
        self,
        selected_medicines: List[Dict],
        patient_info: Dict
    ) -> Dict:
        """Analyze drug safety"""
        return self.interaction_analyzer.analyze(
            selected_medicines,
            patient_info
        ).dict()
    
    def generate_soap_note(
        self,
        transcription: List[Dict],
        patient_info: Dict
    ) -> tuple[str, str]:
        """Generate SOAP note"""
        return self.soap_generator.generate(transcription, patient_info)
    
    def generate_suggestions(
        self,
        selected_medicines: List[Dict],
        patient_info: Dict,
        selected_diseases: List[str]
    ) -> Dict[str, List[str]]:
        """Generate patient suggestions"""
        ai_suggestions = self.suggestion_generator.generate_ai_suggestions(
            selected_medicines,
            patient_info,
            selected_diseases
        )
        
        conversation_suggestions = self.suggestion_generator.extract_conversation_suggestions(
            patient_info
        )
        
        return {
            'ai_suggestions': ai_suggestions,
            'conversation_suggestions': conversation_suggestions
        }
    
    def generate_final_prescription(
        self,
        patient_info: Dict,
        selected_diseases: List[str],
        selected_medicines: List[Dict],
        drug_alerts: Dict,
        selected_suggestions: List[str],
        soap_note: Optional[str] = None
    ) -> Prescription:
        """Generate final prescription"""
        return self.prescription_generator.generate(
            patient_info,
            selected_diseases,
            selected_medicines,
            drug_alerts,
            selected_suggestions,
            soap_note
        )
    
    def export_prescription(
        self,
        prescription: Prescription,
        output_dir: Path,
        formats: List[str] = ['json', 'latex']
    ) -> Dict[str, str]:
        """Export prescription to multiple formats"""
        exported_files = {}
        
        if 'json' in formats:
            json_path = output_dir / f"prescription_{prescription.patient_name}_{prescription.date}.json"
            exported_files['json'] = self.document_exporter.export_to_json(
                prescription.dict(),
                json_path
            )
        
        if 'latex' in formats:
            latex_path = output_dir / f"prescription_{prescription.patient_name}_{prescription.date}.tex"
            exported_files['latex'] = self.document_exporter.export_prescription_to_latex(
                prescription,
                latex_path
            )
        
        return exported_files
    
    def run_complete_workflow(
        self,
        audio_path: str,
        selected_diseases: List[str],
        selected_medicines: List[Dict],
        selected_suggestions: List[str],
        output_dir: Path
    ) -> Dict:
        """
        Run complete workflow from audio to prescription
        
        Returns:
            Dictionary with all results
        """
        # 1. Process audio
        audio_results = self.process_audio_file(audio_path)
        transcription = audio_results['transcription']
        patient_info = audio_results['patient_info']
        
        # 2. Generate diagnoses (if not provided)
        if not selected_diseases:
            diagnoses = self.generate_diagnoses(patient_info)
            selected_diseases = [d['disease'] for d in diagnoses[:3]]
        
        # 3. Generate medicines (if not provided)
        if not selected_medicines:
            selected_medicines = self.generate_medicines(patient_info, selected_diseases)
        
        # 4. Analyze safety
        drug_alerts = self.analyze_safety(selected_medicines, patient_info)
        
        # 5. Generate SOAP note
        soap_note, deidentified_conv = self.generate_soap_note(transcription, patient_info)
        
        # 6. Generate suggestions (if not provided)
        if not selected_suggestions:
            suggestions = self.generate_suggestions(
                selected_medicines,
                patient_info,
                selected_diseases
            )
            selected_suggestions = suggestions['ai_suggestions'][:5]
        
        # 7. Generate prescription
        prescription = self.generate_final_prescription(
            patient_info,
            selected_diseases,
            selected_medicines,
            drug_alerts,
            selected_suggestions,
            soap_note
        )
        
        # 8. Export
        exported_files = self.export_prescription(prescription, output_dir)
        
        return {
            'transcription': transcription,
            'patient_info': patient_info,
            'prescription': prescription,
            'soap_note': soap_note,
            'exported_files': exported_files
        }