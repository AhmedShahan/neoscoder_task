"""
Audio Processing Module
Handles audio diarization and transcription
"""
import os
import torch
import torchaudio
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import pipeline
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from pydub import AudioSegment
from scipy.spatial.distance import cosine

from ..config import MODEL_PATHS, PROCESSING_CONFIG, AUDIO_CONFIG
from ..schemas import TranscriptionSegment


class AudioProcessor:
    """
    Processes audio files for speaker diarization and transcription
    """
    
    def __init__(self):
        """Initialize audio processor with models"""
        self.device = PROCESSING_CONFIG['DEVICE'] if torch.cuda.is_available() else -1
        self.target_sample_rate = AUDIO_CONFIG['TARGET_SAMPLE_RATE']
        self.min_segment_samples = AUDIO_CONFIG['MIN_SEGMENT_SAMPLES']
        self.similarity_threshold = PROCESSING_CONFIG['SIMILARITY_THRESHOLD']
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load required models for processing"""
        # Load doctor embedding
        self.doctor_embedding = np.load(MODEL_PATHS['EMBEDDING_PATH'])
        
        # Load ASR model
        self.asr = pipeline(
            "automatic-speech-recognition", 
            model=MODEL_PATHS['ASR_MODEL_PATH'], 
            device=self.device
        )
        
        # Load embedding model
        embedding_model = Model.from_pretrained(
            MODEL_PATHS['EMBEDDING_MODEL_PATH'],
            config_yaml=MODEL_PATHS['EMBEDDING_CONFIG_PATH']
        )
        self.embedder = Inference(embedding_model, window="whole")
        
        # Load diarization pipeline
        self.diarization_pipeline = SpeakerDiarization.from_pretrained(
            MODEL_PATHS['DIARIZATION_PIPELINE_PATH'] + '/config.yaml'
        )
    
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
        similarity = 1 - cosine(self.doctor_embedding, segment_embedding)
        return "Doctor" if similarity > self.similarity_threshold else "Patient"
    
    def process_segment(
        self, 
        waveform: torch.Tensor, 
        segment: Segment, 
        sample_rate: int
    ) -> Dict:
        """
        Process a single audio segment
        
        Args:
            waveform: Audio waveform tensor
            segment: Segment timing information
            sample_rate: Audio sample rate
            
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
        
        transcription = self.asr(audio_np, return_timestamps=True)["text"]
        
        # Generate embedding for speaker identification
        try:
            file_dict = {"waveform": speaker_waveform, "sample_rate": sample_rate}
            segment_embedding = self.embedder(file_dict)
        except Exception:
            # Fallback: save temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, speaker_waveform, sample_rate)
                segment_embedding = self.embedder(tmp_file.name)
                os.unlink(tmp_file.name)
        
        # Identify speaker
        speaker = self.identify_speaker(segment_embedding)
        
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
            diarization = self.diarization_pipeline(temp_audio_path)
            
            results = []
            
            # Process each segment
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                result = self.process_segment(waveform, turn, sr)
                if result:
                    results.append(result)
            
            # Clean up temporary WAV file
            if temp_audio_path != audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
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