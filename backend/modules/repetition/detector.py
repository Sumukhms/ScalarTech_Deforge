"""Repetition detection and removal - OPTIMIZED."""
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
from collections import Counter
try:
    from rapidfuzz import fuzz
except ImportError:
    try:
        from Levenshtein import ratio as levenshtein_ratio
        fuzz = None
    except ImportError:
        from python_Levenshtein import ratio as levenshtein_ratio
        fuzz = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class RepetitionDetector:
    """Detect and remove repeated phrases and segments - OPTIMIZED."""
    
    def __init__(self, similarity_threshold: float = 0.88, min_phrase_length: int = 3):
        """
        Initialize repetition detector.
        
        Args:
            similarity_threshold: Similarity threshold (0.88 = 88% similar)
            min_phrase_length: Minimum word count for phrase detection
        """
        self.similarity_threshold = similarity_threshold
        self.min_phrase_length = min_phrase_length
        self.use_rapidfuzz = fuzz is not None
        
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if self.use_rapidfuzz:
            return fuzz.ratio(text1, text2) / 100.0
        else:
            return levenshtein_ratio(text1, text2)
    
    def remove_consecutive_repetitions(self, text: str) -> str:
        """
        Remove consecutive repeated words or short phrases.
        
        Args:
            text: Input text
            
        Returns:
            Text with consecutive repetitions removed
        """
        # Remove consecutive word repetitions (e.g., "the the project")
        pattern = r'\b(\w+)(\s+\1\b)+'
        cleaned = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        
        # Remove consecutive 2-word phrase repetitions
        words = cleaned.split()
        result = []
        i = 0
        
        while i < len(words):
            # Check for 2-word phrase repetition
            if i + 3 < len(words):
                phrase1 = f"{words[i]} {words[i+1]}".lower()
                phrase2 = f"{words[i+2]} {words[i+3]}".lower()
                
                if phrase1 == phrase2:
                    # Keep only first occurrence
                    result.extend([words[i], words[i+1]])
                    i += 4  # Skip both phrases
                    continue
            
            result.append(words[i])
            i += 1
        
        return ' '.join(result)
    
    def remove_repetitions(self, text: str) -> str:
        """
        Remove repeated phrases from text - OPTIMIZED.
        
        Args:
            text: Input text with potential repetitions
            
        Returns:
            Text with repetitions removed
        """
        if not text or not text.strip():
            return text
        
        original_length = len(text)
        
        # Step 1: Remove consecutive repetitions (fast)
        cleaned = self.remove_consecutive_repetitions(text)
        
        # Step 2: Remove sentence-level repetitions
        cleaned = self._remove_sentence_repetitions(cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        reduction = original_length - len(cleaned)
        if reduction > 0:
            logger.debug(f"Removed {reduction} chars of repetition ({reduction/original_length*100:.1f}%)")
        
        return cleaned
    
    def _remove_sentence_repetitions(self, text: str) -> str:
        """
        Remove repeated sentences or very similar sentences.
        
        Args:
            text: Input text
            
        Returns:
            Text with sentence repetitions removed
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+\s*)', text)
        
        # Reconstruct sentence + punctuation pairs
        sentence_pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_pairs.append((sentences[i], sentences[i + 1]))
            else:
                sentence_pairs.append((sentences[i], ''))
        
        if not sentence_pairs:
            return text
        
        # Track seen sentences
        seen_sentences: List[str] = []
        result_pairs = []
        
        for sentence, punct in sentence_pairs:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Normalize for comparison
            normalized = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
            
            # Check if too short to be meaningful
            word_count = len(normalized.split())
            if word_count < 2:
                result_pairs.append((sentence, punct))
                continue
            
            # Check similarity with previously seen sentences
            is_duplicate = False
            for seen in seen_sentences:
                similarity = self._similarity_score(normalized, seen)
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Skipping similar sentence ({similarity:.2%}): {sentence[:40]}...")
                    break
            
            if not is_duplicate:
                result_pairs.append((sentence, punct))
                # Only track sentences with meaningful length
                if word_count >= self.min_phrase_length:
                    seen_sentences.append(normalized)
        
        # Reconstruct text
        result = ''.join([f"{s}{p}" for s, p in result_pairs])
        return result
    
    def detect_repetitions(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect repeated phrases in text (for analysis).
        
        Args:
            text: Input text
            
        Returns:
            List of (phrase, count) tuples
        """
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        words = normalized.split()
        
        if len(words) < self.min_phrase_length * 2:
            return []
        
        phrase_counts = Counter()
        
        # Check n-grams from min to max length
        max_ngram = min(len(words) // 2, 8)  # Cap at 8 words
        
        for n in range(self.min_phrase_length, max_ngram + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i + n])
                phrase_counts[phrase] += 1
        
        # Return phrases that appear multiple times
        repetitions = [
            (phrase, count)
            for phrase, count in phrase_counts.items()
            if count > 1
        ]
        
        # Sort by frequency and length
        repetitions.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        return repetitions[:10]  # Return top 10