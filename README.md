# Audio Recording and Enhancement Script

## Overview
This Python script records audio from a microphone, applies noise reduction using spectral subtraction, amplifies the volume, and saves the processed audio as a WAV file. The script uses a multi-threaded approach to record audio in real-time while allowing the user to stop recording by pressing 'q' and Enter.

## Technology Objectives
- **Real-time Audio Recording**: Capture audio from the microphone in 1-second chunks using the `sounddevice` library.
- **Noise Reduction**: Implement spectral subtraction using the `librosa` library to reduce background noise, assuming the first 0.5 seconds of the recording contains primarily noise.
- **Volume Amplification**: Apply a user-defined gain factor to increase the volume of the recorded audio.
- **Threading**: Use Python's `threading` module to record audio in the background while allowing user input to stop the recording.
- **Output**: Save the processed audio as a WAV file using `scipy.io.wavfile`.

## Requirements
### Software
- **Python**: Version 3.7 or higher.
- **Operating System**: Compatible with Windows, macOS, or Linux (with appropriate audio input devices).

### Dependencies
- `sounddevice`: For real-time audio recording.
- `numpy`: For numerical operations and array manipulation.
- `scipy`: For saving the audio as a WAV file.
- `librosa`: For audio processing and spectral subtraction.

Install the dependencies using pip:
```bash
pip install sounddevice numpy scipy librosa
```

### Hardware
- A microphone or audio input device compatible with the operating system.
- Sufficient memory and processing power to handle real-time audio processing (minimal requirements for most modern systems).

## Usage
1. Ensure all dependencies are installed.
2. Run the script in a Python environment.
3. The script will start recording audio immediately. Press 'q' followed by Enter to stop recording.
4. The recorded audio will be processed (noise reduction and volume amplification) and saved as `output_enhanced.wav` in the current directory.

## Code
```python
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import threading

fs = 44100  # Sample rate
channels = 2  # Stereo
gain_factor = 3.0  # Volume amplification factor (adjust as needed)

# A list to store recorded chunks
audio_chunks = []
recording = True

def reduce_noise(audio, sr):
    """Apply noise reduction using spectral subtraction."""
    # Convert stereo to mono for noise reduction (librosa works better with mono)
    audio_mono = np.mean(audio, axis=1)
    
    # Estimate noise profile from the first 0.5 seconds (assuming it's mostly noise)
    noise_clip = audio_mono[:int(0.5 * sr)]
    noise_spectrum = np.abs(librosa.stft(noise_clip))
    noise_mean = np.mean(noise_spectrum, axis=1)
    
    # Perform spectral subtraction
    audio_stft = librosa.stft(audio_mono)
    audio_spectrum = np.abs(audio_stft)
    # Subtract noise profile, ensuring non-negative values
    clean_spectrum = np.maximum(audio_spectrum - noise_mean[:, None], 0)
    
    # Reconstruct the signal
    clean_stft = clean_spectrum * np.exp(1j * np.angle(audio_stft))
    clean_audio_mono = librosa.istft(clean_stft)
    
    # Convert back to stereo by duplicating the mono signal
    clean_audio_stereo = np.stack([clean_audio_mono, clean_audio_mono], axis=1)
    return clean_audio_stereo

def amplify_volume(audio, gain):
    """Amplify the volume of the audio."""
    return audio * gain

def record_audio():
    global audio_chunks
    print("Recording... Press 'q' + Enter to stop.")
    while recording:
        chunk = sd.rec(int(fs * 1), samplerate=fs, channels=channels, dtype='float32')  # Record in 1-second chunks
        sd.wait()
        audio_chunks.append(chunk)

# Start recording in a separate thread
thread = threading.Thread(target=record_audio)
thread.start()

# Wait for user input to stop
while True:
    stop = input()  # Press "q" then Enter
    if stop.lower() == 'q':
        recording = False
        break

thread.join()  # Wait until recording thread finishes

# Concatenate all audio chunks
final_recording = np.concatenate(audio_chunks, axis=0)

# Apply noise reduction
cleaned_recording = reduce_noise(final_recording, fs)

# Apply volume amplification
amplified_recording = amplify_volume(cleaned_recording, gain_factor)

# Save the enhanced recording
write("output_enhanced.wav", fs, amplified_recording)
print("Enhanced recording saved to output_enhanced.wav")
```

## Notes
- The script assumes the first 0.5 seconds of the recording is noise for the spectral subtraction algorithm. Adjust this duration if the noise profile differs.
- The `channels` variable is set to 2 for stereo audio, as the original value of 3 is non-standard and may cause issues.
- The gain factor (`gain_factor`) can be adjusted to control the volume amplification. Be cautious with high values to avoid clipping.
- The script does not handle exceptions (e.g., missing audio devices or insufficient permissions). Add error handling for production use.

## Limitations
- The noise reduction assumes the initial 0.5 seconds is representative of background noise, which may not always be accurate.
- The script converts stereo to mono for noise reduction, which may not preserve spatial audio effects.
- No real-time feedback is provided during recording beyond the initial prompt.

## Future Improvements
- Add real-time audio visualization using a library like `matplotlib`.
- Implement more advanced noise reduction techniques (e.g., adaptive filtering).
- Add command-line arguments to customize sample rate, channels, or gain factor.
- Include error handling for audio device issues or file writing failures.