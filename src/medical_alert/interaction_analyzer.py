"""
Drug Interaction Analyzer Module
Analyzes drug-drug interactions and safety
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG
from ..schemas import DrugAlertAnalysis, DrugAlert
from ..drug_suggestion.drug_info_service import DrugInfoService


class InteractionAnalyzer:
    """Analyzes drug interactions for safety"""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self.parser = PydanticOutputParser(pydantic_object=DrugAlertAnalysis)
        self.drug_info_service = DrugInfoService()
        self._setup_chain()
    
    def _setup_chain(self):
        self.prompt = ChatPromptTemplate.from_messages([
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
        
        self.chain = self.prompt | self.model | self.parser
    
    def analyze(
        self, 
        selected_medicines: List[Dict],
        patient_info: Dict
    ) -> DrugAlertAnalysis:
        """Analyze drug interactions and safety"""
        medicine_list = "\n".join([
            f"- {med['medicine']}: {med['reason']}" 
            for med in selected_medicines
        ])
        
        drug_names = [med['medicine'] for med in selected_medicines]
        rxnorm_interactions = self.drug_info_service.check_all_interactions(drug_names)
        
        try:
            alert_analysis = self.chain.invoke({
                "allergies": patient_info.get("Allergies", "None reported"),
                "medical_history": patient_info.get("Previous_Medical_History", "None reported"),
                "current_problems": patient_info.get("Recent_Problem", "None reported"),
                "previous_drugs": patient_info.get("Previous_Drug_History", "None reported"),
                "proposed_medications": medicine_list,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Add RxNorm interactions
            for interaction in rxnorm_interactions:
                severity_map = {
                    "high": "HIGH",
                    "moderate": "MODERATE", 
                    "low": "LOW",
                    "unknown": "MODERATE"
                }
                severity = severity_map.get(interaction.severity.lower(), "MODERATE")
                
                alert_analysis.alerts.append(DrugAlert(
                    alert_type="DRUG_INTERACTION",
                    severity=severity,
                    drug1=interaction.drug1,
                    drug2=interaction.drug2,
                    description=interaction.description,
                    recommendation="Consult with pharmacist or consider alternative medications"
                ))
            
            # Update overall risk level
            if any(alert.severity == "HIGH" for alert in alert_analysis.alerts):
                alert_analysis.overall_risk_level = "HIGH"
            elif any(alert.severity == "MODERATE" for alert in alert_analysis.alerts):
                alert_analysis.overall_risk_level = "MODERATE to HIGH"
            
            return alert_analysis
        
        except Exception as e:
            raise Exception(f"Error analyzing drug interactions: {str(e)}")


def analyze_drug_safety(
    selected_medicines: List[Dict],
    patient_info: Dict
) -> DrugAlertAnalysis:
    """Convenience function"""
    analyzer = InteractionAnalyzer()
    return analyzer.analyze(selected_medicines, patient_info)