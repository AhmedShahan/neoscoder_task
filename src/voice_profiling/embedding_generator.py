"""
Embedding Generator Module
Generates speaker embeddings from audio files
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from ..config import HUGGINGFACE_MODELS, MODEL_PATHS, PROCESSING_CONFIG


class EmbeddingGenerator:
    """
    Generates speaker embeddings from audio files using PyAnnote
    """
    
    def __init__(self):
        """Initialize embedding generator"""
        self.device = "cuda" if torch.cuda.is_available() and PROCESSING_CONFIG['DEVICE'] >= 0 else "cpu"
        self.auth_token = PROCESSING_CONFIG.get('USE_AUTH_TOKEN')
        self.target_sample_rate = 16000
        self._load_model()
    
    def _load_model(self):
        """Load PyAnnote embedding model from Hugging Face"""
        print(f"Loading embedding model: {HUGGINGFACE_MODELS['EMBEDDING_MODEL']}")
        
        self.embedding_model = PretrainedSpeakerEmbedding(
            HUGGINGFACE_MODELS['EMBEDDING_MODEL'],
            device=torch.device(self.device),
            use_auth_token=self.auth_token
        )
        
        print("✅ Embedding model loaded successfully!")
    
    def preprocess_audio(self, audio_path: str) -> tuple:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
            sr = self.target_sample_rate
        
        return waveform, sr
    
    def generate_embedding_from_file(self, audio_path: str) -> np.ndarray:
        """
        Generate embedding from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Numpy array containing speaker embedding
        """
        try:
            # Preprocess audio
            waveform, sample_rate = self.preprocess_audio(audio_path)
            
            # Generate embedding
            embedding = self.embedding_model(waveform[None])
            
            # Handle both tensor and numpy array
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.squeeze().cpu().numpy()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.squeeze()
            else:
                embedding = np.array(embedding).flatten()
            
            return embedding
        
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def generate_embedding_from_waveform(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Generate embedding from waveform
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of audio
            
        Returns:
            Numpy array containing speaker embedding
        """
        try:
            # Ensure correct format
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Generate embedding
            embedding = self.embedding_model(waveform[None])
            
            # Handle both tensor and numpy array
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.squeeze().cpu().numpy()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.squeeze()
            else:
                embedding = np.array(embedding).flatten()
            
            return embedding
        
        except Exception as e:
            raise Exception(f"Error generating embedding from waveform: {str(e)}")
    
    def save_embedding(self, embedding: np.ndarray, output_path: Union[str, Path]) -> str:
        """
        Save embedding to file
        
        Args:
            embedding: Embedding array
            output_path: Path to save embedding
            
        Returns:
            Path where embedding was saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, embedding)
        return str(output_path)
    
    def load_embedding(self, embedding_path: Union[str, Path]) -> np.ndarray:
        """
        Load embedding from file
        
        Args:
            embedding_path: Path to embedding file
            
        Returns:
            Embedding array
        """
        return np.load(embedding_path)


def generate_embedding(audio_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to generate embedding from audio file
    
    Args:
        audio_path: Path to audio file
        output_path: Optional path to save embedding
        
    Returns:
        Embedding array
    """
    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding_from_file(audio_path)
    
    if output_path:
        generator.save_embedding(embedding, output_path)
    
    return embedding