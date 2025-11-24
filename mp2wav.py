from pydub import AudioSegment

# Load MP3 file
mp3_file = "/home/shahanahmed/Office_Shellow_EMR/ElevenLabs_2025-07-07T10_35_21_Rachel_pre_sp100_s50_sb75_v3.mp3"
audio = AudioSegment.from_mp3(mp3_file)

# Export as WAV
wav_file = "output.wav"
audio.export(wav_file, format="wav")