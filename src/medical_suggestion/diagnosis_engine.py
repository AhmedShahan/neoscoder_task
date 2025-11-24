"""
Diagnosis Engine Module
Generates disease suggestions based on patient information
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG
from ..schemas import DiagnosisSuggestions, DiseaseSuggestion


class DiagnosisEngine:
    """
    Generates diagnostic suggestions using AI
    """
    
    def __init__(self):
        """Initialize diagnosis engine"""
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self.parser = PydanticOutputParser(pydantic_object=DiagnosisSuggestions)
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain processing chain"""
        self.prompt = ChatPromptTemplate.from_messages([
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
        
        self.chain = self.prompt | self.model | self.parser
    
    def generate_suggestions(self, patient_info: Dict) -> List[Dict]:
        """
        Generate diagnostic suggestions
        
        Args:
            patient_info: Dictionary of patient information
            
        Returns:
            List of disease suggestions with scores and reasons
        """
        try:
            result = self.chain.invoke({
                "Recent_Problem": patient_info.get("Recent_Problem", ""),
                "Previous_Medical_History": patient_info.get("Previous_Medical_History", ""),
                "Previous_Drug_History": patient_info.get("Previous_Drug_History", ""),
                "Allergies": patient_info.get("Allergies", ""),
                "Family_Medical_History": patient_info.get("Family_Medical_History", ""),
                "Lifestyle_Details": patient_info.get("Lifestyle_Details", {}),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return result.dict()['suggestions']
        
        except Exception as e:
            raise Exception(f"Error generating diagnostic suggestions: {str(e)}")
    
    def add_custom_diagnosis(
        self, 
        existing_suggestions: List[Dict], 
        custom_disease: str,
        reason: str = "User added"
    ) -> List[Dict]:
        """
        Add custom diagnosis to suggestions
        
        Args:
            existing_suggestions: Current list of suggestions
            custom_disease: Custom disease name
            reason: Reason for addition
            
        Returns:
            Updated list of suggestions
        """
        custom_suggestion = {
            "disease": custom_disease,
            "score": 0.0,  # User added, no AI score
            "reason": reason
        }
        
        # Check if already exists
        if not any(s['disease'] == custom_disease for s in existing_suggestions):
            existing_suggestions.append(custom_suggestion)
        
        return existing_suggestions
    
    def filter_suggestions(
        self, 
        suggestions: List[Dict], 
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Filter suggestions by minimum score
        
        Args:
            suggestions: List of disease suggestions
            min_score: Minimum confidence score
            
        Returns:
            Filtered list of suggestions
        """
        return [s for s in suggestions if s['score'] >= min_score]
    
    def sort_suggestions(
        self, 
        suggestions: List[Dict], 
        by: str = 'score',
        reverse: bool = True
    ) -> List[Dict]:
        """
        Sort suggestions by specified field
        
        Args:
            suggestions: List of disease suggestions
            by: Field to sort by ('score' or 'disease')
            reverse: Sort in descending order
            
        Returns:
            Sorted list of suggestions
        """
        return sorted(suggestions, key=lambda x: x[by], reverse=reverse)


def generate_disease_suggestions(patient_info: Dict) -> List[Dict]:
    """
    Convenience function to generate disease suggestions
    
    Args:
        patient_info: Dictionary of patient information
        
    Returns:
        List of disease suggestions
    """
    engine = DiagnosisEngine()
    return engine.generate_suggestions(patient_info)