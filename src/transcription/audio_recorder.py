"""
Audio Recording Module
Handles live audio recording from microphone
"""
import pyaudio
import wave
import threading
import queue
from pathlib import Path
from typing import Optional

from ..config import AUDIO_CONFIG, OUTPUT_FILES


class AudioRecorder:
    """
    Records audio from microphone in real-time
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize audio recorder
        
        Args:
            output_path: Path to save recorded audio (default: temp_audio.wav)
        """
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.output_path = output_path or OUTPUT_FILES['TEMP_AUDIO']
        
        # Audio configuration
        self.format = getattr(pyaudio, AUDIO_CONFIG['FORMAT'])
        self.channels = AUDIO_CONFIG['CHANNELS']
        self.rate = AUDIO_CONFIG['RATE']
        self.chunk = AUDIO_CONFIG['CHUNK']
        
    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.is_recording = True
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def _record(self):
        """Internal method to record audio (runs in separate thread)"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            while self.is_recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                
            self.stream.stop_stream()
            self.stream.close()
        except Exception as e:
            print(f"Recording error: {e}")
            self.is_recording = False
        
    def stop_recording(self) -> str:
        """
        Stop recording and save audio file
        
        Returns:
            Path to saved audio file
        """
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
            
        # Save the recorded data as a WAV file
        try:
            wf = wave.open(str(self.output_path), 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            return str(self.output_path)
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def __del__(self):
        """Cleanup audio resources"""
        try:
            self.audio.terminate()
        except:
            pass


def record_audio(duration_seconds: int = None, output_path: Optional[Path] = None) -> str:
    """
    Convenience function to record audio
    
    Args:
        duration_seconds: Maximum recording duration (None for manual stop)
        output_path: Path to save audio file
        
    Returns:
        Path to saved audio file
    """
    recorder = AudioRecorder(output_path)
    recorder.start_recording()
    
    if duration_seconds:
        import time
        time.sleep(duration_seconds)
        return recorder.stop_recording()
    
    return recorder  # Return recorder for manual stopping