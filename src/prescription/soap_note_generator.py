"""
SOAP Note Generator Module
Generates clinical SOAP notes
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG
from ..extraction.deidentification import Deidentifier


class SOAPNoteGenerator:
    """Generates SOAP notes from conversations"""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self.deidentifier = Deidentifier(use_ner=True)
        self._setup_chain()
    
    def _setup_chain(self):
        self.prompt = ChatPromptTemplate.from_template("""
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
        
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def generate(
        self, 
        transcription: List[Dict],
        patient_info: Dict
    ) -> tuple[str, str]:
        """
        Generate SOAP note
        
        Returns:
            Tuple of (soap_note, deidentified_conversation)
        """
        conversation_text = "\n".join([
            f"{item['speaker']}: {item['text']}" 
            for item in transcription
        ])
        
        deidentified_text = self.deidentifier.deidentify(conversation_text)
        
        soap_note = self.chain.invoke({
            "conversation": deidentified_text,
            "age": patient_info.get('Age', 'Not provided'),
            "gender": patient_info.get('Gender', 'Not provided'),
            "medical_history": patient_info.get('Previous_Medical_History', 'Not provided'),
            "drug_history": patient_info.get('Previous_Drug_History', 'Not provided'),
            "allergies": patient_info.get('Allergies', 'Not provided'),
            "family_history": patient_info.get('Family_Medical_History', 'Not provided')
        })
        
        return soap_note, deidentified_text