from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from pydub import AudioSegment
import os
import tempfile
import shutil
import re
from typing import List, Dict, Optional


CACHE_BASE = os.getenv("HF_CACHE_BASE", r"G:\huggingface_cache")
os.environ["HF_HOME"] = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "models")

class VoiceModel:
    def __init__(self, model_name: str = "microsoft/speecht5_tts", 
                 vocoder_name: str = "microsoft/speecht5_hifigan"):
        
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.vocoder.to(self.device)
        
        self.speaker_embeddings = self._generate_stable_embedding()

    def _generate_stable_embedding(self):
        """Creates a consistent vocal profile to prevent 'grainy' audio."""
        torch.manual_seed(42) 
        embedding = torch.randn(1, 512) * 0.05
        return embedding.to(self.device)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Prevents stuttering by ensuring text segments aren't too long."""
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def _change_speed(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Adjusts the playback speed (1.2 is 20% faster, 0.8 is 20% slower)."""
        if speed == 1.0:
            return audio
        new_sample_rate = int(audio.frame_rate * speed)
        return audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(audio.frame_rate)

    def generate_audio_from_reels(self, reels: List[Dict], output_path: str = "final_output.mp3", speed: float = 1.0):
        """
      
            output_path: Path to save the output audio file
            speed: Playback speed multiplier (1.0 = normal, 1.2 = 20% faster)
        """
        segments = []
        current_time = 0
        
        for reel in reels:
            narration = reel.get("narration", {})
            text = narration.get("text", "")
            estimated_sec = narration.get("estimatedSpeechSec", 0)
            
            if text:
                segments.append({
                    "text": text,
                    "start": current_time,
                    "end": current_time + estimated_sec
                })
                current_time += estimated_sec
        
        return self.generate_audio_from_segments(segments, output_path, speed)

    def generate_audio_from_segments(self, segments: List[Dict], output_path: str = "final_output.mp3", speed: float = 1.0):
        """Generates clear speech with speed control."""
        temp_dir = tempfile.mkdtemp()
        audio_files = []

        try:
            for i, seg in enumerate(segments):
                sub_sentences = self._split_into_sentences(seg["text"])
                
                segment_audio = AudioSegment.silent(duration=0)
                for j, sentence in enumerate(sub_sentences):
                    tmp_wav = os.path.join(temp_dir, f"seg_{i}_{j}.wav")
                    
                    inputs = self.processor(text=sentence, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
                    
                    sf.write(tmp_wav, speech.cpu().numpy(), samplerate=16000)
                    
                    sentence_audio = AudioSegment.from_wav(tmp_wav)
                    segment_audio += sentence_audio

                segment_audio = self._change_speed(segment_audio, speed)
                
                final_wav = os.path.join(temp_dir, f"final_seg_{i}.wav")
                segment_audio.export(final_wav, format="wav")
                audio_files.append((final_wav, seg["start"], seg["end"]))

            final_combined = self._stitch_audio_segments(audio_files)
            
            file_ext = os.path.splitext(output_path)[1].lower()
            output_format = "mp3" if file_ext == ".mp3" else "wav"
            
            try:
                final_combined.export(output_path, format=output_format)
            except FileNotFoundError:
                if output_format == "mp3":
                    # Fall back to wav if mp3 export fails (ffmpeg not installed)
                    wav_path = os.path.splitext(output_path)[0] + ".wav"
                    final_combined.export(wav_path, format="wav")
                    print(f"Note: MP3 export requires ffmpeg. Saved as WAV instead: {wav_path}")
                    output_path = wav_path
                else:
                    raise
            
            print(f"Success! Saved to {output_path}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _stitch_audio_segments(self, audio_files: List[tuple]) -> AudioSegment:
        """Combines segments with correct timing."""
        final_audio = AudioSegment.silent(duration=0)
        for filename, start, end in audio_files:
            segment_audio = AudioSegment.from_wav(filename)
            silence_padding = (start * 1000) - len(final_audio)
            if silence_padding > 0:
                final_audio += AudioSegment.silent(duration=silence_padding)
            final_audio += segment_audio
        return final_audio

if __name__ == "__main__":
    model = VoiceModel()
    
    # Example using reels array format
    example_reels = [
        {
            "reelId": "reel_1",
            "topicId": "topic_1",
            "narration": {
                "text": "What is Android architecture? Android architecture consists of multiple layers that work together to provide a secure and efficient platform.",
                "estimatedSpeechSec": 8
            },
            "explanationLevel": "beginner",
            "visualIntent": []
        },
        {
            "reelId": "reel_2",
            "topicId": "topic_1",
            "narration": {
                "text": "Now let's explore the security components in detail.",
                "estimatedSpeechSec": 5
            },
            "explanationLevel": "intermediate",
            "visualIntent": []
        }
    ]
    
    model.generate_audio_from_reels(example_reels, speed=1.1)