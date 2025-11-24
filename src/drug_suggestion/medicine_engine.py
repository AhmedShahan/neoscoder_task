"""
Medicine Recommendation Engine Module
Generates medicine suggestions based on diagnoses and patient information
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG
from ..schemas import MedicineSuggestions
from .drug_info_service import DrugInfoService


class MedicineEngine:
    """
    Generates medicine recommendations using AI and enriches with FDA data
    """
    
    def __init__(self):
        """Initialize medicine engine"""
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self.parser = PydanticOutputParser(pydantic_object=MedicineSuggestions)
        self.drug_info_service = DrugInfoService()
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain processing chain"""
        self.prompt = ChatPromptTemplate.from_messages([
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
        
        self.chain = self.prompt | self.model | self.parser
    
    def generate_suggestions(
        self, 
        patient_info: Dict, 
        selected_diseases: List[str]
    ) -> List[Dict]:
        """
        Generate medicine suggestions
        
        Args:
            patient_info: Patient information dictionary
            selected_diseases: List of selected diseases
            
        Returns:
            List of medicine suggestions with enriched data
        """
        diseases_string = ", ".join(selected_diseases) if selected_diseases else \
            "No specific diseases selected - base suggestions on symptoms and patient history"
        
        try:
            # Generate AI suggestions
            result = self.chain.invoke({
                "Recent_Problem": patient_info.get("Recent_Problem", ""),
                "Previous_Medical_History": patient_info.get("Previous_Medical_History", ""),
                "Family_Medical_History": patient_info.get("Family_Medical_History", ""),
                "Selected_Diseases": diseases_string,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            suggestions = result.dict()['suggestions']
            
            # Enrich with FDA data
            enriched_suggestions = self._enrich_with_fda_data(suggestions)
            
            return enriched_suggestions
        
        except Exception as e:
            raise Exception(f"Error generating medicine suggestions: {str(e)}")
    
    def _enrich_with_fda_data(self, medicines: List[Dict]) -> List[Dict]:
        """
        Enrich medicine suggestions with FDA data
        
        Args:
            medicines: List of medicine suggestions
            
        Returns:
            Enriched list with purpose and side effects
        """
        enriched_medicines = []
        
        for med in medicines:
            drug_info = self.drug_info_service.get_drug_info(med['medicine'])
            
            enriched_med = {
                "medicine": med['medicine'],
                "score": med['score'],
                "reason": med['reason'],
                "purpose": drug_info.purpose,
                "side_effects": drug_info.side_effects
            }
            
            enriched_medicines.append(enriched_med)
        
        return enriched_medicines
    
    def add_custom_medicine(
        self, 
        existing_suggestions: List[Dict],
        medicine_name: str,
        reason: str = "User added"
    ) -> List[Dict]:
        """
        Add custom medicine to suggestions
        
        Args:
            existing_suggestions: Current list of suggestions
            medicine_name: Custom medicine name
            reason: Reason for addition
            
        Returns:
            Updated list of suggestions
        """
        # Check if already exists
        if any(m['medicine'] == medicine_name for m in existing_suggestions):
            return existing_suggestions
        
        # Get FDA data
        drug_info = self.drug_info_service.get_drug_info(medicine_name)
        
        custom_medicine = {
            "medicine": medicine_name,
            "score": 0.0,  # User added
            "reason": reason,
            "purpose": drug_info.purpose,
            "side_effects": drug_info.side_effects
        }
        
        existing_suggestions.append(custom_medicine)
        return existing_suggestions
    
    def filter_by_allergies(
        self, 
        medicines: List[Dict],
        allergies: str
    ) -> List[Dict]:
        """
        Filter out medicines that might conflict with allergies
        (Basic implementation - would need more sophisticated matching in production)
        
        Args:
            medicines: List of medicine suggestions
            allergies: Patient allergies string
            
        Returns:
            Filtered list
        """
        if not allergies or allergies.lower() in ['none', 'no known allergies']:
            return medicines
        
        # Simple keyword matching (enhance in production)
        allergy_keywords = allergies.lower().split(',')
        
        filtered = []
        for med in medicines:
            medicine_name = med['medicine'].lower()
            # Check if any allergy keyword is in medicine name
            if not any(keyword.strip() in medicine_name for keyword in allergy_keywords):
                filtered.append(med)
        
        return filtered


def generate_medicine_suggestions(
    patient_info: Dict,
    selected_diseases: List[str]
) -> List[Dict]:
    """
    Convenience function to generate medicine suggestions
    
    Args:
        patient_info: Patient information
        selected_diseases: List of selected diseases
        
    Returns:
        List of medicine suggestions
    """
    engine = MedicineEngine()
    return engine.generate_suggestions(patient_info, selected_diseases)