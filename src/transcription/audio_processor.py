"""
Audio Processing Module
Handles audio diarization and transcription using Hugging Face models
"""
import os
import torch
import torchaudio
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from pydub import AudioSegment
from scipy.spatial.distance import cosine

from ..config import HUGGINGFACE_MODELS, MODEL_PATHS, PROCESSING_CONFIG, AUDIO_CONFIG


class AudioProcessor:
    """
    Processes audio files for speaker diarization and transcription
    Uses Hugging Face models directly
    """
    
    def __init__(self):
        """Initialize audio processor with models from Hugging Face"""
        self.device = "cuda" if torch.cuda.is_available() and PROCESSING_CONFIG['DEVICE'] >= 0 else "cpu"
        self.target_sample_rate = AUDIO_CONFIG['TARGET_SAMPLE_RATE']
        self.min_segment_samples = AUDIO_CONFIG['MIN_SEGMENT_SAMPLES']
        self.similarity_threshold = PROCESSING_CONFIG['SIMILARITY_THRESHOLD']
        self.auth_token = PROCESSING_CONFIG.get('USE_AUTH_TOKEN')
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load required models from Hugging Face"""
        print("Loading models from Hugging Face...")
        
        # Check for authentication token
        if not self.auth_token:
            print("\n" + "="*80)
            print("WARNING: No Hugging Face token found!")
            print("PyAnnote models require authentication. Please:")
            print("1. Create a token at: https://huggingface.co/settings/tokens")
            print("2. Accept conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Accept conditions at: https://huggingface.co/pyannote/segmentation-3.0")
            print("4. Add HUGGINGFACE_TOKEN to your .env file")
            print("="*80 + "\n")
            raise ValueError("HUGGINGFACE_TOKEN is required for PyAnnote models")
        
        # Load doctor embedding if exists
        doctor_embedding_path = MODEL_PATHS['DOCTOR_EMBEDDING_PATH']
        if os.path.exists(doctor_embedding_path):
            self.doctor_embedding = np.load(doctor_embedding_path)
            print(f"Loaded doctor embedding from {doctor_embedding_path}")
        else:
            print("Warning: Doctor embedding not found. Will use default speaker labels.")
            self.doctor_embedding = None
        
        # Load Whisper ASR model from Hugging Face
        print(f"Loading ASR model: {HUGGINGFACE_MODELS['ASR_MODEL']}")
        device_id = 0 if self.device == "cuda" else -1
        
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=HUGGINGFACE_MODELS['ASR_MODEL'],
            chunk_length_s=30,
            device=device_id
        )
        
        # Load PyAnnote embedding model from Hugging Face
        print(f"Loading embedding model: {HUGGINGFACE_MODELS['EMBEDDING_MODEL']}")
        try:
            self.embedding_model = PretrainedSpeakerEmbedding(
                HUGGINGFACE_MODELS['EMBEDDING_MODEL'],
                device=torch.device(self.device),
                use_auth_token=self.auth_token
            )
        except Exception as e:
            print(f"\nError loading embedding model: {e}")
            print("Make sure you've accepted the model conditions at:")
            print(f"https://huggingface.co/{HUGGINGFACE_MODELS['EMBEDDING_MODEL']}")
            raise
        
        # Load PyAnnote diarization pipeline from Hugging Face
        print(f"Loading diarization pipeline: {HUGGINGFACE_MODELS['DIARIZATION_MODEL']}")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                HUGGINGFACE_MODELS['DIARIZATION_MODEL'],
                use_auth_token=self.auth_token
            )
        except Exception as e:
            print(f"\nError loading diarization pipeline: {e}")
            print("Make sure you've accepted the model conditions at:")
            print(f"https://huggingface.co/{HUGGINGFACE_MODELS['DIARIZATION_MODEL']}")
            print("https://huggingface.co/pyannote/segmentation-3.0")
            raise
        
        # Move to appropriate device
        if self.device == "cuda":
            self.diarization_pipeline.to(torch.device("cuda"))
        
        print("All models loaded successfully!")
    
    def convert_to_wav(self, audio_path: str) -> str:
        """
        Convert audio file to WAV format if needed
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to WAV file
        """
        if audio_path.endswith('.wav'):
            return audio_path
        
        temp_audio_path = audio_path.replace(Path(audio_path).suffix, '.wav')
        audio = AudioSegment.from_file(audio_path)
        audio.export(temp_audio_path, format="wav")
        return temp_audio_path
    
    def preprocess_audio(self, audio_path: str):
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
    
    def identify_speaker(self, segment_embedding: np.ndarray) -> str:
        """
        Identify if speaker is doctor or patient
        
        Args:
            segment_embedding: Audio embedding for segment
            
        Returns:
            Speaker label: "Doctor" or "Patient"
        """
        if self.doctor_embedding is None:
            # Fallback: use speaker index
            return "Speaker"
        
        similarity = 1 - cosine(self.doctor_embedding, segment_embedding)
        return "Doctor" if similarity > self.similarity_threshold else "Patient"
    
    def get_segment_embedding(self, audio_path: str, segment: Segment) -> np.ndarray:
        """
        Get embedding for a specific audio segment
        
        Args:
            audio_path: Path to audio file
            segment: Segment timing information
            
        Returns:
            Embedding array
        """
        # Create a crop of the segment
        from pyannote.audio import Audio
        from pyannote.core import Segment as PyannoteSegment
        
        audio = Audio(sample_rate=self.target_sample_rate, mono=True)
        waveform, sample_rate = audio.crop(audio_path, PyannoteSegment(segment.start, segment.end))
        
        # Get embedding
        embedding = self.embedding_model(waveform[None])
        
        # Handle both tensor and numpy array returns
        if isinstance(embedding, torch.Tensor):
            return embedding.squeeze().cpu().numpy()
        elif isinstance(embedding, np.ndarray):
            return embedding.squeeze()
        else:
            # If it's already squeezed numpy array
            return np.array(embedding).flatten()
    
    def process_segment(
        self, 
        audio_path: str,
        waveform: torch.Tensor, 
        segment: Segment, 
        sample_rate: int,
        speaker_label: str
    ) -> Optional[Dict]:
        """
        Process a single audio segment
        
        Args:
            audio_path: Path to audio file
            waveform: Audio waveform tensor
            segment: Segment timing information
            sample_rate: Audio sample rate
            speaker_label: Speaker label from diarization
            
        Returns:
            Dictionary with speaker, text, and timestamp
        """
        # Skip very short segments
        if segment.duration < PROCESSING_CONFIG['MIN_SEGMENT_DURATION']:
            return None
        
        # Calculate sample indices
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        
        # Bounds checking
        if end_sample > waveform.shape[1]:
            end_sample = waveform.shape[1]
        if start_sample >= end_sample:
            return None
        
        # Extract segment waveform
        segment_length = end_sample - start_sample
        if segment_length < self.min_segment_samples:
            # Pad short segments
            padding = self.min_segment_samples - segment_length
            speaker_waveform = torch.nn.functional.pad(
                waveform[:, start_sample:end_sample], 
                (0, padding), 
                mode='constant', 
                value=0
            )
        else:
            speaker_waveform = waveform[:, start_sample:end_sample]
        
        # Ensure 2D waveform
        if speaker_waveform.dim() == 1:
            speaker_waveform = speaker_waveform.unsqueeze(0)
        
        # Transcribe audio segment
        audio_np = speaker_waveform.squeeze().numpy()
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        transcription = self.asr(audio_np)["text"]
        
        # Get embedding and identify speaker
        try:
            segment_embedding = self.get_segment_embedding(audio_path, segment)
            speaker = self.identify_speaker(segment_embedding)
        except Exception as e:
            print(f"Warning: Could not identify speaker: {e}")
            # Fallback to diarization label
            speaker = speaker_label
        
        return {
            "speaker": speaker,
            "text": transcription,
            "timestamp": f"{segment.start:.2f}-{segment.end:.2f}s"
        }
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Process complete audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcription segments
        """
        # Convert to WAV if needed
        temp_audio_path = self.convert_to_wav(audio_path)
        
        try:
            # Preprocess audio
            waveform, sr = self.preprocess_audio(temp_audio_path)
            
            # Perform diarization
            print("Performing speaker diarization...")
            diarization = self.diarization_pipeline(temp_audio_path)
            
            results = []
            
            # Process each segment
            print("Transcribing segments...")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                result = self.process_segment(
                    temp_audio_path,
                    waveform, 
                    turn, 
                    sr,
                    speaker
                )
                if result:
                    results.append(result)
            
            # Clean up temporary WAV file
            if temp_audio_path != audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            print(f"Processed {len(results)} segments")
            return results
        
        except Exception as e:
            # Clean up on error
            if temp_audio_path != audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise Exception(f"Error processing audio: {str(e)}")


def process_audio_file(audio_path: str) -> List[Dict]:
    """
    Convenience function to process audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of transcription segments
    """
    processor = AudioProcessor()
    return processor.process_audio(audio_path)