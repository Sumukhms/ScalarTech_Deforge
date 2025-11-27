"""Grammar correction - OPTIMIZED for speed (<500ms)."""
import re
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List
from functools import lru_cache

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

class GrammarCorrector:
    """
    Fast grammar correction using rule-based NLP + optional T5 fallback.
    
    OPTIMIZATION: Rules first (fast), T5 only for complex cases (slow).
    Target: <300ms for most texts.
    """
    
    # Common grammar patterns (fast regex)
    GRAMMAR_RULES: Dict[str, str] = {
        # Subject-verb agreement
        r'\bI is\b': 'I am',
        r'\bI was\b(?= [a-z]+ing)': 'I was',  # Keep "I was going"
        r'\bI were\b': 'I was',
        r'\byou was\b': 'you were',
        r'\bhe are\b': 'he is',
        r'\bshe are\b': 'she is',
        r'\bit are\b': 'it is',
        r'\bthey is\b': 'they are',
        r'\bwe was\b': 'we were',
        
        # Verb tenses
        r'\bI goed\b': 'I went',
        r'\bI runned\b': 'I ran',
        r'\bI buyed\b': 'I bought',
        r'\bI eated\b': 'I ate',
        r'\bI drinked\b': 'I drank',
        r'\bI seed\b': 'I saw',
        r'\bI builded\b': 'I built',
        
        # Double negatives
        r"\bdon't\s+no\b": "don't know",
        r"\bdon't\s+nothing\b": "don't have anything",
        r"\bain't\s+no\b": "isn't any",
        
        # Article errors
        r'\ba\s+([aeiou])': r'an \1',  # a apple -> an apple
        r'\ban\s+([^aeiou\s])': r'a \1',  # an car -> a car
        
        # Pronoun cases
        r'\bme and\s+(\w+)\s+(is|are|was|were|have|has|had)\b': r'\1 and I \2',
        r'\bhim and me\b': 'he and I',
        r'\bher and me\b': 'she and I',
    }
    
    def __init__(self, model_name: str = "t5-small", use_model: bool = False):
        """
        Initialize grammar corrector.
        
        Args:
            model_name: T5 model name (only loaded if use_model=True)
            use_model: Whether to load T5 model (slow, only for complex cases)
        """
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
        # Only load model if explicitly requested
        if use_model:
            self._load_model()
        else:
            logger.info("Grammar corrector initialized in FAST mode (rules only)")
    
    def _load_model(self):
        """Load T5 model (SLOW - only for fallback)."""
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            logger.info(f"Loading T5 model: {self.model_name} (this is SLOW)")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            
            logger.info(f"T5 model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            logger.warning("Falling back to rule-based only")
            self.model = None
            self.tokenizer = None
    
    @lru_cache(maxsize=256)
    def correct(self, text: str) -> str:
        """
        Correct grammar in text - OPTIMIZED.
        
        Args:
            text: Input text
            
        Returns:
            Grammar-corrected text
        """
        if not text or not text.strip():
            return text
        
        # FAST PATH: Rule-based corrections (< 50ms)
        corrected = self._apply_grammar_rules(text)
        
        # SLOW PATH: T5 model (only if enabled and text is complex)
        # Skip T5 for speed - rules are usually enough for STT output
        
        return corrected
    
    def _apply_grammar_rules(self, text: str) -> str:
        """
        Apply fast rule-based grammar corrections.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        corrected = text
        
        # Apply grammar rules
        for pattern, replacement in self.GRAMMAR_RULES.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Fix capitalization
        corrected = self._fix_capitalization(corrected)
        
        # Fix spacing
        corrected = self._fix_spacing(corrected)
        
        # Fix punctuation
        corrected = self._fix_punctuation(corrected)
        
        return corrected.strip()
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization issues."""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        text = re.sub(
            r'([.!?]\s+)([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            text
        )
        
        # Capitalize "I"
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Fix spaces after punctuation
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
        
        return text
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation issues."""
        # Remove multiple punctuation marks
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def correct_with_model(self, text: str, max_length: int = 256) -> str:
        """
        Apply T5 model-based correction (SLOW - avoid if possible).
        
        Args:
            text: Input text
            max_length: Max sequence length
            
        Returns:
            Corrected text
        """
        if not self.model or not self.tokenizer:
            logger.debug("Model not loaded, using rules only")
            return self._apply_grammar_rules(text)
        
        try:
            import torch
            
            # Split into sentences for better results
            sentences = re.split(r'([.!?]+)', text)
            corrected_parts = []
            
            for sentence in sentences:
                if not sentence.strip() or sentence.strip() in '.!?':
                    corrected_parts.append(sentence)
                    continue
                
                # Prepare input with grammar correction prompt
                input_text = f"grammar: {sentence.strip()}"
                inputs = self.tokenizer.encode(
                    input_text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True
                )
                
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                
                # Generate correction
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode result
                corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_parts.append(corrected)
            
            result = ''.join(corrected_parts)
            return result.strip()
            
        except Exception as e:
            logger.error(f"Model-based correction failed: {e}")
            return self._apply_grammar_rules(text)