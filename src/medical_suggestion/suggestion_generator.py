"""
Patient Suggestion Generator Module
Generates lifestyle and monitoring suggestions for patients
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG


class SuggestionGenerator:
    """
    Generates practical suggestions for patients based on their condition and medications
    """
    
    def __init__(self):
        """Initialize suggestion generator"""
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain processing chain"""
        self.prompt = ChatPromptTemplate.from_messages([
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
        
        self.chain = self.prompt | self.model | JsonOutputParser()
    
    def generate_ai_suggestions(
        self, 
        selected_medicines: List[Dict],
        patient_info: Dict,
        selected_diseases: List[str]
    ) -> List[str]:
        """
        Generate AI-based suggestions
        
        Args:
            selected_medicines: List of selected medicines
            patient_info: Patient information dictionary
            selected_diseases: List of selected diseases
            
        Returns:
            List of suggestion strings
        """
        # Format medication information
        medications_text = "\n".join([
            f"- {med['medicine']}: Side effects include {med.get('side_effects', 'Not available')}" 
            for med in selected_medicines
        ])
        
        try:
            result = self.chain.invoke({
                "diagnoses": ", ".join(selected_diseases),
                "medications": medications_text,
                "medical_history": patient_info.get("Previous_Medical_History", "None"),
                "allergies": patient_info.get("Allergies", "None"),
                "age": patient_info.get("Age", "Not specified"),
                "gender": patient_info.get("Gender", "Not specified")
            })
            
            return result.get("suggestions", [])
        
        except Exception as e:
            raise Exception(f"Error generating AI suggestions: {str(e)}")
    
    def extract_conversation_suggestions(self, patient_info: Dict) -> List[str]:
        """
        Extract suggestions mentioned in the conversation
        
        Args:
            patient_info: Patient information dictionary
            
        Returns:
            List of suggestions from conversation
        """
        follow_up_actions = patient_info.get('Follow_Up_Actions', [])
        
        if not follow_up_actions:
            return []
        
        return follow_up_actions if isinstance(follow_up_actions, list) else [follow_up_actions]
    
    def merge_suggestions(
        self, 
        conversation_suggestions: List[str],
        ai_suggestions: List[str],
        custom_suggestions: List[str]
    ) -> Dict[str, List[str]]:
        """
        Merge suggestions from different sources
        
        Args:
            conversation_suggestions: Suggestions from conversation
            ai_suggestions: AI-generated suggestions
            custom_suggestions: User-added custom suggestions
            
        Returns:
            Dictionary categorizing suggestions by source
        """
        return {
            'conversation': conversation_suggestions,
            'ai': ai_suggestions,
            'custom': custom_suggestions
        }
    
    def deduplicate_suggestions(self, suggestions: List[str]) -> List[str]:
        """
        Remove duplicate suggestions (case-insensitive)
        
        Args:
            suggestions: List of suggestions
            
        Returns:
            De-duplicated list
        """
        seen = set()
        unique = []
        
        for suggestion in suggestions:
            normalized = suggestion.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(suggestion)
        
        return unique


def generate_patient_suggestions(
    selected_medicines: List[Dict],
    patient_info: Dict,
    selected_diseases: List[str]
) -> List[str]:
    """
    Convenience function to generate patient suggestions
    
    Args:
        selected_medicines: List of selected medicines
        patient_info: Patient information
        selected_diseases: List of selected diseases
        
    Returns:
        List of suggestions
    """
    generator = SuggestionGenerator()
    return generator.generate_ai_suggestions(
        selected_medicines,
        patient_info,
        selected_diseases
    )