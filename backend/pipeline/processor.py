"""Main dictation processing pipeline - OPTIMIZED for <1500ms."""
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.stt import STTEngine
from modules.fillers import FillerRemover
from modules.repetition import RepetitionDetector
from modules.grammar import GrammarCorrector
from modules.formatting import AutoFormatter
from modules.tone import ToneTransformer, ToneMode
from utils.latency import LatencyTracker
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

class DictationProcessor:
    """
    Main processing pipeline - OPTIMIZED.
    
    Target: <1500ms total latency
    Breakdown:
    - STT: Already done before this (0ms in pipeline)
    - Filler removal: ~50ms
    - Repetition removal: ~100ms
    - Grammar correction: ~300ms (rules only, no T5)
    - Formatting: ~50ms
    - Tone transformation: ~100ms
    Total target: ~600ms processing (well under 1500ms)
    """
    
    def __init__(self):
        """Initialize all processing modules with optimization."""
        logger.info("Initializing OPTIMIZED dictation processor...")
        
        start_time = time.time()
        
        # Initialize STT engine
        try:
            self.stt_engine = STTEngine(config.VOSK_MODEL_PATH)
            logger.info(f"STT engine initialized in {time.time()-start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}")
            self.stt_engine = None
        
        # Initialize processing modules (fast initialization)
        module_start = time.time()
        
        self.filler_remover = FillerRemover()
        self.repetition_detector = RepetitionDetector(
            similarity_threshold=0.88,  # 88% similarity to detect repetitions
            min_phrase_length=3
        )
        
        # Grammar: RULES ONLY for speed (no T5 model)
        self.grammar_corrector = GrammarCorrector(
            model_name=config.GRAMMAR_MODEL_NAME,
            use_model=False  # CRITICAL: Don't load T5 for speed
        )
        
        self.formatter = AutoFormatter()
        self.tone_transformer = ToneTransformer()
        
        logger.info(f"Processing modules initialized in {time.time()-module_start:.2f}s")
        logger.info(f"Total initialization: {time.time()-start_time:.2f}s")
        logger.info("Processor ready for <1500ms processing")
    
    def process_full(
        self,
        text: str,
        tone: ToneMode = "neutral",
        track_latency: bool = True
    ) -> Dict[str, Any]:
        """
        Process text through OPTIMIZED full pipeline.
        
        Args:
            text: Raw input text (from STT)
            tone: Desired tone
            track_latency: Whether to track processing latency
            
        Returns:
            Dictionary with processed text and metrics
        """
        if not text or not text.strip():
            return self._empty_result(text, tone)
        
        tracker = LatencyTracker()
        if track_latency:
            tracker.start()
        
        original_text = text
        processed_text = text
        
        try:
            # STEP 1: Remove fillers (~50ms)
            if track_latency:
                with tracker.stage("filler_removal"):
                    processed_text = self.filler_remover.remove(processed_text)
            else:
                processed_text = self.filler_remover.remove(processed_text)
            
            # Early exit if text is empty after filler removal
            if not processed_text.strip():
                processed_text = original_text  # Restore original
            
            # STEP 2: Remove repetitions (~100ms)
            if track_latency:
                with tracker.stage("repetition_removal"):
                    processed_text = self.repetition_detector.remove_repetitions(processed_text)
            else:
                processed_text = self.repetition_detector.remove_repetitions(processed_text)
            
            # STEP 3: Grammar correction (~300ms with rules only)
            if track_latency:
                with tracker.stage("grammar_correction"):
                    processed_text = self.grammar_corrector.correct(processed_text)
            else:
                processed_text = self.grammar_corrector.correct(processed_text)
            
            # STEP 4: Auto-formatting (~50ms)
            if track_latency:
                with tracker.stage("formatting"):
                    processed_text = self.formatter.format(processed_text)
            else:
                processed_text = self.formatter.format(processed_text)
            
            # STEP 5: Tone transformation (~100ms)
            if track_latency:
                with tracker.stage("tone_transformation"):
                    processed_text = self.tone_transformer.transform(processed_text, tone)
            else:
                processed_text = self.tone_transformer.transform(processed_text, tone)
            
            if track_latency:
                tracker.end()
            
            # Build result
            result = {
                "original_text": original_text,
                "processed_text": processed_text,
                "tone": tone,
                "improvement": {
                    "original_length": len(original_text),
                    "processed_length": len(processed_text),
                    "reduction_percent": round(
                        (len(original_text) - len(processed_text)) / len(original_text) * 100, 2
                    ) if original_text else 0,
                    "original_words": len(original_text.split()),
                    "processed_words": len(processed_text.split())
                }
            }
            
            if track_latency:
                latency_summary = tracker.get_summary()
                result["latency"] = latency_summary
                
                # Log performance
                total_ms = latency_summary["total_latency_ms"]
                meets_target = latency_summary["meets_target"]
                
                logger.info(
                    f"Processing complete: {total_ms:.1f}ms "
                    f"({'✓ PASS' if meets_target else '✗ FAIL'} <1500ms target)"
                )
                
                # Log stage breakdown
                for stage, ms in latency_summary["stage_breakdown"].items():
                    logger.debug(f"  {stage}: {ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            # Return original text on error
            return {
                "original_text": original_text,
                "processed_text": original_text,  # Fallback to original
                "tone": tone,
                "improvement": {"original_length": len(original_text), "processed_length": len(original_text), "reduction_percent": 0},
                "error": str(e)
            }
    
    def _empty_result(self, text: str, tone: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "original_text": text,
            "processed_text": text,
            "tone": tone,
            "improvement": {
                "original_length": 0,
                "processed_length": 0,
                "reduction_percent": 0
            },
            "latency": {
                "total_latency_ms": 0,
                "stage_breakdown": {},
                "meets_target": True
            }
        }
    
    def process_step(self, text: str, step: str, **kwargs) -> str:
        """
        Process text through a single step (for testing).
        
        Args:
            text: Input text
            step: Step name
            **kwargs: Additional arguments
            
        Returns:
            Processed text
        """
        if not text or not text.strip():
            return text
        
        if step == "fillers":
            return self.filler_remover.remove(text)
        elif step == "repetition":
            return self.repetition_detector.remove_repetitions(text)
        elif step == "grammar":
            return self.grammar_corrector.correct(text)
        elif step == "formatting":
            return self.formatter.format(text)
        elif step == "tone":
            tone = kwargs.get("mode", "neutral")
            return self.tone_transformer.transform(text, tone)
        else:
            raise ValueError(f"Unknown step: {step}")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if self.stt_engine is None:
            raise RuntimeError("STT engine not initialized")
        
        return self.stt_engine.transcribe_audio_file(audio_path)
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        if self.stt_engine is None:
            raise RuntimeError("STT engine not initialized")
        
        return self.stt_engine.transcribe_audio_bytes(audio_bytes, sample_rate)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics (for monitoring)."""
        return {
            "stt_available": self.stt_engine is not None,
            "modules_loaded": {
                "filler_remover": self.filler_remover is not None,
                "repetition_detector": self.repetition_detector is not None,
                "grammar_corrector": self.grammar_corrector is not None,
                "formatter": self.formatter is not None,
                "tone_transformer": self.tone_transformer is not None
            },
            "target_latency_ms": config.TARGET_LATENCY_MS,
            "grammar_mode": "rules_only (fast)"
        }