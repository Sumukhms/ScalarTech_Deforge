"""Filler word and disfluency removal - ENHANCED."""
import re
import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class FillerRemover:
    """Remove filler words and disfluencies - ENHANCED."""
    
    # Comprehensive filler words list
    FILLER_WORDS: Set[str] = {
        # Basic fillers
        "um", "umm", "uh", "uhh", "eh", "ehh", "ah", "ahh",
        "er", "erm", "erm", "uhm", "uhmm", "hmm", "hmmm",
        
        # Discourse markers (often fillers in speech)
        "like", "you know", "you see", "I mean",
        "well", "so", "actually", "basically", "literally",
        "kind of", "sort of", "right", "okay", "ok",
        
        # Problem statement specific
        "matlab",  # Mentioned in requirements as filler
        
        # Hesitation phrases
        "I guess", "I think", "I suppose", "I believe",
        "let me see", "let's see", "you know what",
        
        # Additional common fillers
        "stuff", "thing", "things", "something",
        "whatever", "anyways", "anyway"
    }
    
    # Filler phrases (multi-word patterns)
    FILLER_PHRASES: List[str] = [
        r"\byou know\b",
        r"\bI mean\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\byou see\b",
        r"\bI guess\b",
        r"\bI think\b",
        r"\bI suppose\b",
        r"\bI believe\b",
        r"\blet me see\b",
        r"\blet's see\b",
        r"\byou know what\b",
        r"\bor something\b",
        r"\bor whatever\b",
        r"\band stuff\b",
        r"\band things\b",
    ]
    
    # Repeated sounds (stammering)
    STAMMER_PATTERN = r'\b(\w+)-\1\b'  # "t-t-the" -> "the"
    
    def __init__(self):
        """Initialize filler remover."""
        # Compile regex patterns for performance
        self.filler_phrase_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.FILLER_PHRASES
        ]
        
        # Single word filler pattern
        self.word_boundary_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )
        
        # Stammer pattern
        self.stammer_pattern = re.compile(self.STAMMER_PATTERN, re.IGNORECASE)
    
    def remove(self, text: str) -> str:
        """
        Remove filler words and disfluencies.
        
        Args:
            text: Input text with potential fillers
            
        Returns:
            Text with fillers removed
        """
        if not text or not text.strip():
            return text
        
        original_length = len(text)
        cleaned = text
        
        # Step 1: Remove stammering (e.g., "t-t-the" -> "the")
        cleaned = self.stammer_pattern.sub(r'\1', cleaned)
        
        # Step 2: Remove filler phrases (multi-word)
        for pattern in self.filler_phrase_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Step 3: Remove single filler words
        cleaned = self.word_boundary_pattern.sub('', cleaned)
        
        # Step 4: Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1', cleaned)  # Remove double punct
        cleaned = cleaned.strip()
        
        reduction = original_length - len(cleaned)
        if reduction > 5:  # Only log significant reductions
            logger.debug(f"Removed {reduction} chars of fillers ({reduction/original_length*100:.1f}%)")
        
        return cleaned
    
    def remove_with_context(self, text: str, preserve_meaning: bool = True) -> str:
        """
        Remove fillers with context awareness (smarter removal).
        
        Args:
            text: Input text
            preserve_meaning: Try to preserve sentence meaning
            
        Returns:
            Cleaned text
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+\s*)', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                cleaned_sentences.append(sentence)
                continue
            
            # Remove fillers
            cleaned = self.remove(sentence)
            
            # Check if sentence became too short (might have been mostly fillers)
            if preserve_meaning:
                original_words = len(sentence.split())
                cleaned_words = len(cleaned.split())
                
                # If we removed >80% of words, sentence was probably all fillers
                if original_words > 3 and cleaned_words < 2:
                    logger.debug(f"Skipping sentence (too much removed): {sentence[:30]}...")
                    continue
            
            if cleaned.strip():
                cleaned_sentences.append(cleaned)
        
        result = ''.join(cleaned_sentences)
        return re.sub(r'\s+', ' ', result).strip()
    
    def count_fillers(self, text: str) -> dict:
        """
        Count filler words in text (for analysis).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with filler counts
        """
        counts = {}
        
        # Count single filler words
        for word in self.FILLER_WORDS:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            count = len(pattern.findall(text))
            if count > 0:
                counts[word] = count
        
        return counts