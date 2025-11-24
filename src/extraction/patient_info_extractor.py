"""
Patient Information Extraction Module
Extracts structured patient information from conversation
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import API_CONFIG
from ..schemas import PatientInformation


class PatientInfoExtractor:
    """
    Extracts structured patient information from medical conversation
    """
    
    def __init__(self):
        """Initialize patient info extractor"""
        self.model = ChatGoogleGenerativeAI(
            model=API_CONFIG['GEMINI_MODEL'],
            temperature=API_CONFIG['TEMPERATURE']
        )
        self.parser = PydanticOutputParser(pydantic_object=PatientInformation)
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup LangChain processing chain"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Patient Information Extractor. Respond in valid JSON only."),
            ("human", "{conversation}\n\n{format_instructions}")
        ])
        
        self.chain = self.prompt | self.model | self.parser
    
    def extract(self, transcription: List[Dict]) -> Dict:
        """
        Extract patient information from transcription
        
        Args:
            transcription: List of transcription segments
            
        Returns:
            Dictionary of patient information
        """
        # Format conversation
        conversation_text = "\n".join([
            f"{item['speaker']}: {item['text']}" 
            for item in transcription
        ])
        
        try:
            result = self.chain.invoke({
                "conversation": conversation_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result.dict()
        
        except Exception as e:
            raise Exception(f"Error extracting patient information: {str(e)}")
    
    def extract_from_text(self, conversation_text: str) -> Dict:
        """
        Extract patient information from conversation text
        
        Args:
            conversation_text: Formatted conversation string
            
        Returns:
            Dictionary of patient information
        """
        try:
            result = self.chain.invoke({
                "conversation": conversation_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result.dict()
        
        except Exception as e:
            raise Exception(f"Error extracting patient information: {str(e)}")


def extract_patient_info(transcription: List[Dict]) -> Dict:
    """
    Convenience function to extract patient information
    
    Args:
        transcription: List of transcription segments
        
    Returns:
        Dictionary of patient information
    """
    extractor = PatientInfoExtractor()
    return extractor.extract(transcription)