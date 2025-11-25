"""
De-identification Module
Removes PHI (Protected Health Information) from text
Uses Hugging Face models directly
"""
import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from ..config import PHI_REPLACE_TAGS, HUGGINGFACE_MODELS, MODEL_PATHS


class Deidentifier:
    """
    Removes Protected Health Information from medical text
    """
    
    def __init__(self, use_ner: bool = True):
        """
        Initialize de-identifier
        
        Args:
            use_ner: Whether to use NER model (slower but more accurate)
        """
        self.use_ner = use_ner
        self.ner_pipeline = None
        
        if use_ner:
            self._load_ner_model()
    
    def _load_ner_model(self):
        """Load NER model from Hugging Face"""
        try:
            print(f"Loading NER model: {HUGGINGFACE_MODELS['NER_MODEL']}")
            tokenizer = AutoTokenizer.from_pretrained(
                HUGGINGFACE_MODELS['NER_MODEL'],
                cache_dir=MODEL_PATHS.get('CACHE_DIR')
            )
            model = AutoModelForTokenClassification.from_pretrained(
                HUGGINGFACE_MODELS['NER_MODEL'],
                cache_dir=MODEL_PATHS.get('CACHE_DIR')
            )
            self.ner_pipeline = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="simple"
            )
            print("NER model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}. Using fallback method.")
            self.ner_pipeline = None
            self.use_ner = False
    
    def _deidentify_with_patterns(self, text: str) -> str:
        """
        De-identify using regex patterns (fallback method)
        
        Args:
            text: Input text
            
        Returns:
            De-identified text
        """
        # Replace names (simple pattern)
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        # Replace dates
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]', text)
        
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Replace addresses
        text = re.sub(
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', 
            '[ADDRESS]', 
            text
        )
        
        # Replace Social Security Numbers
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Replace Medical Record Numbers (common patterns)
        text = re.sub(r'\bMRN:?\s*\d+\b', '[MRN]', text, flags=re.IGNORECASE)
        
        return text
    
    def _deidentify_with_ner(self, text: str) -> str:
        """
        De-identify using NER model
        
        Args:
            text: Input text
            
        Returns:
            De-identified text
        """
        if not self.ner_pipeline:
            return self._deidentify_with_patterns(text)
        
        try:
            entities = self.ner_pipeline(text)
            new_text = text
            
            # Sort entities by start position in reverse to avoid index shifting
            for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
                label = entity['entity_group']
                replacement = PHI_REPLACE_TAGS.get(label, "[REDACTED]")
                new_text = new_text[:entity['start']] + replacement + new_text[entity['end']:]
            
            return new_text
        
        except Exception as e:
            print(f"Warning: NER de-identification failed: {e}. Using fallback.")
            return self._deidentify_with_patterns(text)
    
    def deidentify(self, text: str) -> str:
        """
        Remove PHI from text
        
        Args:
            text: Input text with potential PHI
            
        Returns:
            De-identified text
        """
        if self.use_ner and self.ner_pipeline:
            return self._deidentify_with_ner(text)
        else:
            return self._deidentify_with_patterns(text)
    
    def deidentify_conversation(self, transcription: list) -> str:
        """
        De-identify a conversation transcription
        
        Args:
            transcription: List of transcription segments
            
        Returns:
            De-identified conversation text
        """
        conversation_text = "\n".join([
            f"{item['speaker']}: {item['text']}" 
            for item in transcription
        ])
        
        return self.deidentify(conversation_text)


def deidentify_text(text: str, use_ner: bool = True) -> str:
    """
    Convenience function to de-identify text
    
    Args:
        text: Input text
        use_ner: Whether to use NER model
        
    Returns:
        De-identified text
    """
    deidentifier = Deidentifier(use_ner=use_ner)
    return deidentifier.deidentify(text)