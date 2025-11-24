"""
Transcription Service
Main orchestration for audio transcription workflow
"""
from typing import List, Dict
from pathlib import Path

from .audio_processor import AudioProcessor
from .audio_recorder import AudioRecorder
from ..schemas import TranscriptionSegment


class TranscriptionService:
    """
    Orchestrates the complete transcription workflow
    """
    
    def __init__(self):
        """Initialize transcription service"""
        self.audio_processor = AudioProcessor()
        self.audio_recorder = None
    
    def transcribe_file(self, audio_path: str) -> List[Dict]:
        """
        Transcribe an uploaded audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcription segments with speaker labels
        """
        return self.audio_processor.process_audio(audio_path)
    
    def start_recording(self):
        """Start live audio recording"""
        if not self.audio_recorder:
            self.audio_recorder = AudioRecorder()
        self.audio_recorder.start_recording()
    
    def stop_recording_and_transcribe(self) -> List[Dict]:
        """
        Stop recording and transcribe the recorded audio
        
        Returns:
            List of transcription segments
        """
        if not self.audio_recorder:
            raise Exception("No active recording session")
        
        # Stop recording and get file path
        audio_path = self.audio_recorder.stop_recording()
        
        if not audio_path:
            raise Exception("Recording failed")
        
        # Transcribe the recorded audio
        return self.transcribe_file(audio_path)
    
    def format_conversation(self, transcription: List[Dict]) -> str:
        """
        Format transcription as readable conversation
        
        Args:
            transcription: List of transcription segments
            
        Returns:
            Formatted conversation string
        """
        return "\n".join([
            f"{item['speaker']}: {item['text']}" 
            for item in transcription
        ])
    
    def calculate_duration(self, transcription: List[Dict]) -> float:
        """
        Calculate total consultation duration from transcription
        
        Args:
            transcription: List of transcription segments
            
        Returns:
            Duration in minutes
        """
        if not transcription:
            return 0.0
        
        timestamps = []
        for item in transcription:
            timestamp_str = item.get('timestamp', '0-0s')
            timestamp_parts = timestamp_str.replace('s', '').split('-')
            
            if len(timestamp_parts) == 2:
                try:
                    start = float(timestamp_parts[0])
                    end = float(timestamp_parts[1])
                    timestamps.append((start, end))
                except ValueError:
                    continue
        
        if not timestamps:
            return 0.0
        
        # Find earliest start and latest end
        min_start = min(t[0] for t in timestamps)
        max_end = max(t[1] for t in timestamps)
        
        # Convert to minutes
        duration_seconds = max_end - min_start
        return duration_seconds / 60.0


# Convenience functions for direct use
def transcribe_audio(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of transcription segments
    """
    service = TranscriptionService()
    return service.transcribe_file(audio_path)


def record_and_transcribe(duration_seconds: int = None) -> List[Dict]:
    """
    Record audio and transcribe it
    
    Args:
        duration_seconds: Recording duration (None for manual stop)
        
    Returns:
        List of transcription segments
    """
    service = TranscriptionService()
    service.start_recording()
    
    if duration_seconds:
        import time
        time.sleep(duration_seconds)
    
    return service.stop_recording_and_transcribe()