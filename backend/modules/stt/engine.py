"""Fast Vosk-based Speech-to-Text engine - optimized for <250ms transcription."""
import json
import os
import wave
from pathlib import Path
from typing import Optional, Iterator
import vosk

class STTEngine:
    """Fast Speech-to-Text engine using Vosk - optimized for speed."""
    
    def __init__(self, model_path: str):
        """
        Initialize STT engine with fast Vosk model.
        
        Args:
            model_path: Path to Vosk model directory (use small model for speed)
        """
        self.model_path = model_path
        self.model = None
        self._validate_and_load_model()
    
    def _validate_and_load_model(self):
        """Validate model path and load Vosk model."""
        try:
            # Validate path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Vosk model not found at {self.model_path}\n"
                    f"Download small model (40MB): https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
                    f"Extract to: {self.model_path}"
                )
            
            # Validate it's a directory
            if not os.path.isdir(self.model_path):
                raise ValueError(f"Model path must be a directory: {self.model_path}")
            
            # Validate model structure (quick check)
            required_dirs = ['am', 'conf', 'graph']
            missing = [d for d in required_dirs if not os.path.exists(os.path.join(self.model_path, d))]
            
            if missing:
                raise ValueError(
                    f"Invalid Vosk model structure. Missing: {', '.join(missing)}\n"
                    f"Please ensure the model is correctly extracted to: {self.model_path}"
                )
            
            # Load model (this is the slow part, only done once at startup)
            print(f"Loading Vosk model from {self.model_path}...")
            self.model = vosk.Model(self.model_path)
            print(f"✓ Vosk model loaded successfully")
            
            # Log model info
            model_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(self.model_path)
                for filename in filenames
            ) / (1024 * 1024)  # Convert to MB
            
            print(f"Model size: {model_size:.1f} MB")
            
        except Exception as e:
            print(f"✗ Failed to load Vosk model: {e}")
            raise
    
    def create_recognizer(self, sample_rate: int = 16000):
        """
        Create a recognizer instance.
        
        Args:
            sample_rate: Audio sample rate (default 16000 for Vosk)
            
        Returns:
            vosk.KaldiRecognizer instance
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check model path.")
        
        return vosk.KaldiRecognizer(self.model, sample_rate)
    
    def validate_audio_file(self, audio_path: str) -> dict:
        """
        Quick audio file validation.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio properties
        """
        try:
            wf = wave.open(audio_path, "rb")
            
            properties = {
                'sample_rate': wf.getframerate(),
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'duration': wf.getnframes() / wf.getframerate(),
                'format': 'WAV'
            }
            
            wf.close()
            
            # Quick validation
            if properties['channels'] > 2:
                raise ValueError(f"Too many audio channels: {properties['channels']} (max 2)")
            
            if properties['sample_rate'] < 8000:
                raise ValueError(f"Sample rate too low: {properties['sample_rate']}Hz (min 8000Hz)")
            
            if properties['duration'] < 0.1:
                raise ValueError(f"Audio too short: {properties['duration']:.2f}s (min 0.1s)")
            
            if properties['duration'] > 300:
                raise ValueError(f"Audio too long: {properties['duration']:.1f}s (max 300s)")
            
            return properties
            
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {e}")
        except Exception as e:
            raise ValueError(f"Audio validation failed: {e}")
    
    def transcribe_audio_file(self, audio_path: str, sample_rate: int = 16000) -> str:
        """
        Fast transcription of audio file.
        
        Args:
            audio_path: Path to audio file (WAV format)
            sample_rate: Audio sample rate (ignored, uses actual rate)
            
        Returns:
            Transcribed text
        """
        try:
            # Quick validation
            audio_props = self.validate_audio_file(audio_path)
            
            wf = wave.open(audio_path, "rb")
            actual_sample_rate = wf.getframerate()
            
            # Create recognizer with actual sample rate
            rec = self.create_recognizer(actual_sample_rate)
            
            # Disable word-level details for speed
            rec.SetWords(False)
            
            text_parts = []
            
            # Fast processing with large chunks
            chunk_size = 8000  # Larger chunks = fewer iterations = faster
            
            try:
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    
                    if rec.AcceptWaveform(data):
                        try:
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                        except (json.JSONDecodeError, KeyError):
                            pass
                
                # Get final result (important for last words)
                try:
                    final_result = json.loads(rec.FinalResult())
                    final_text = final_result.get("text", "").strip()
                    if final_text:
                        text_parts.append(final_text)
                except (json.JSONDecodeError, KeyError):
                    pass
                
            finally:
                wf.close()
            
            # Join and clean transcript
            transcript = " ".join(text_parts)
            transcript = self._clean_transcript(transcript)
            
            if transcript:
                word_count = len(transcript.split())
                print(f"✓ Transcription: {len(transcript)} chars, {word_count} words")
            else:
                print("⚠ No speech detected in audio")
            
            return transcript.strip()
            
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _clean_transcript(self, text: str) -> str:
        """Quick transcript cleanup."""
        if not text:
            return text
        
        # Remove extra spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes (for in-memory audio).
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        rec = self.create_recognizer(sample_rate)
        rec.SetWords(False)
        
        text_parts = []
        
        # Process in chunks
        chunk_size = 8000
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            
            if rec.AcceptWaveform(chunk):
                try:
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        text_parts.append(result["text"])
                except:
                    pass
        
        # Get final result
        try:
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                text_parts.append(final_result["text"])
        except:
            pass
        
        transcript = " ".join(text_parts)
        return self._clean_transcript(transcript)
    
    def stream_transcribe(self, audio_stream: Iterator[bytes], sample_rate: int = 16000) -> Iterator[str]:
        """
        Stream transcription from audio chunks.
        
        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Audio sample rate
            
        Yields:
            Partial transcription results
        """
        rec = self.create_recognizer(sample_rate)
        rec.SetWords(False)
        
        for chunk in audio_stream:
            if rec.AcceptWaveform(chunk):
                try:
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        yield result["text"]
                except:
                    pass
        
        # Final result
        try:
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                yield final_result["text"]
        except:
            pass
    
    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        return {
            'model_path': self.model_path,
            'model_loaded': self.model is not None,
            'model_exists': os.path.exists(self.model_path) if self.model_path else False
        }