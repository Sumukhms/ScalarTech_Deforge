"""Vosk-based Speech-to-Text engine - ENHANCED with validation."""
import json
import os
import wave
from pathlib import Path
from typing import Optional, Iterator
import vosk
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class STTEngine:
    """Speech-to-Text engine using Vosk - ENHANCED."""
    
    def __init__(self, model_path: str):
        """
        Initialize STT engine with validation.
        
        Args:
            model_path: Path to Vosk model directory
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
                    f"Please download it from: https://alphacephei.com/vosk/models\n"
                    f"Recommended: vosk-model-en-us-0.22"
                )
            
            # Validate it's a directory
            if not os.path.isdir(self.model_path):
                raise ValueError(f"Model path must be a directory: {self.model_path}")
            
            # Validate model structure
            required_files = ['am/final.mdl', 'conf/mfcc.conf', 'graph/HCLG.fst']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(self.model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                raise ValueError(
                    f"Invalid Vosk model structure. Missing files:\n" +
                    "\n".join(f"  - {f}" for f in missing_files) +
                    f"\n\nPlease ensure the model is correctly extracted to: {self.model_path}"
                )
            
            # Load model
            logger.info(f"Loading Vosk model from {self.model_path}")
            self.model = vosk.Model(self.model_path)
            logger.info("✓ Vosk model loaded successfully")
            
            # Log model info
            model_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(self.model_path)
                for filename in filenames
            ) / (1024 * 1024)  # Convert to MB
            
            logger.info(f"Model size: {model_size:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise
    
    def create_recognizer(self, sample_rate: int = 16000):
        """
        Create a recognizer instance.
        
        Args:
            sample_rate: Audio sample rate
            
        Returns:
            vosk.KaldiRecognizer instance
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check model path.")
        
        return vosk.KaldiRecognizer(self.model, sample_rate)
    
    def validate_audio_file(self, audio_path: str) -> dict:
        """
        Validate audio file format and properties.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio properties
            
        Raises:
            ValueError: If audio file is invalid
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
            
            # Validate properties
            if properties['channels'] > 2:
                raise ValueError(f"Too many audio channels: {properties['channels']} (max 2)")
            
            if properties['sample_rate'] < 8000:
                raise ValueError(f"Sample rate too low: {properties['sample_rate']}Hz (min 8000Hz)")
            
            if properties['duration'] < 0.1:
                raise ValueError(f"Audio too short: {properties['duration']:.2f}s (min 0.1s)")
            
            if properties['duration'] > 300:
                raise ValueError(f"Audio too long: {properties['duration']:.1f}s (max 300s)")
            
            logger.info(
                f"Audio validation passed: "
                f"{properties['sample_rate']}Hz, "
                f"{properties['channels']} channel(s), "
                f"{properties['duration']:.2f}s"
            )
            
            return properties
            
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {e}")
        except Exception as e:
            raise ValueError(f"Audio validation failed: {e}")
    
    def transcribe_audio_file(self, audio_path: str, sample_rate: int = 16000) -> str:
        """
        Transcribe audio file to text with ENHANCED accuracy.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Audio sample rate (ignored, uses actual rate)
            
        Returns:
            Transcribed text
        """
        try:
            # Validate audio first
            audio_props = self.validate_audio_file(audio_path)
            
            wf = wave.open(audio_path, "rb")
            actual_sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            logger.info(
                f"Transcribing: {num_channels}ch, {actual_sample_rate}Hz, "
                f"{sample_width}B/sample, {audio_props['duration']:.2f}s"
            )
            
            # Use actual sample rate
            rec = self.create_recognizer(actual_sample_rate)
            
            # Enable alternatives for better accuracy
            rec.SetWords(True)
            
            text_parts = []
            
            # Optimal chunk size: 4000 frames = ~0.25s at 16kHz
            chunk_size = 4000
            frames_processed = 0
            total_frames = wf.getnframes()
            
            try:
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    
                    frames_processed += chunk_size
                    
                    # Ensure complete frames
                    frame_size = num_channels * sample_width
                    if frame_size > 0 and len(data) % frame_size != 0:
                        padding = frame_size - (len(data) % frame_size)
                        data += b'\x00' * padding
                    
                    # Process chunk
                    if rec.AcceptWaveform(data):
                        try:
                            result = json.loads(rec.Result())
                            text = result.get("text", "").strip()
                            if text:
                                text_parts.append(text)
                                logger.debug(f"Partial: {text}")
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Parse error: {e}")
                            continue
                    
                    # Log progress for long files
                    if audio_props['duration'] > 5 and frames_processed % (actual_sample_rate * 5) == 0:
                        progress = (frames_processed / total_frames) * 100
                        logger.info(f"Transcription progress: {progress:.0f}%")
                
                # Get final result (critical for last words)
                try:
                    final_result = json.loads(rec.FinalResult())
                    final_text = final_result.get("text", "").strip()
                    if final_text:
                        text_parts.append(final_text)
                        logger.debug(f"Final: {final_text}")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Final parse error: {e}")
                
            finally:
                wf.close()
            
            # Join and clean transcript
            transcript = " ".join(text_parts)
            transcript = self._clean_transcript(transcript)
            
            if transcript:
                word_count = len(transcript.split())
                logger.info(
                    f"✓ Transcription complete: {len(transcript)} chars, "
                    f"{word_count} words | '{transcript[:50]}...'"
                )
            else:
                logger.warning("⚠ No speech detected in audio")
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
    
    def _clean_transcript(self, text: str) -> str:
        """Clean up transcript (basic cleanup, before main pipeline)."""
        if not text:
            return text
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove duplicate spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        rec = self.create_recognizer(sample_rate)
        text_parts = []
        
        # Process in chunks
        chunk_size = 4000
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