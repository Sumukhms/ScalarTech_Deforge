"""Grammar correction - OPTIMIZED & ENHANCED for speed (<500ms)."""
import re
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

class GrammarCorrector:
    """
    Fast grammar correction using ENHANCED rule-based NLP.
    
    OPTIMIZATION: Comprehensive rules (fast), T5 only for complex cases (disabled for speed).
    Target: <300ms for most texts.
    """
    
    # EXPANDED Grammar Rules - Comprehensive Coverage
    GRAMMAR_RULES: Dict[str, str] = {
        # Subject-verb agreement
        r'\bI is\b': 'I am',
        r'\bI were\b': 'I was',
        r'\byou was\b': 'you were',
        r'\bhe are\b': 'he is',
        r'\bshe are\b': 'she is',
        r'\bit are\b': 'it is',
        r'\bthey is\b': 'they are',
        r'\bwe was\b': 'we were',
        r'\bhe were\b': 'he was',
        r'\bshe were\b': 'she was',
        r'\bit were\b': 'it was',
        
        # Common verb tenses - past tense
        r'\bI goed\b': 'I went',
        r'\bI runned\b': 'I ran',
        r'\bI buyed\b': 'I bought',
        r'\bI eated\b': 'I ate',
        r'\bI drinked\b': 'I drank',
        r'\bI seed\b': 'I saw',
        r'\bI builded\b': 'I built',
        r'\bI thinked\b': 'I thought',
        r'\bI speaked\b': 'I spoke',
        r'\bI writed\b': 'I wrote',
        r'\bI readed\b': 'I read',
        r'\bI bringed\b': 'I brought',
        r'\bI teached\b': 'I taught',
        r'\bI catched\b': 'I caught',
        r'\bI fighted\b': 'I fought',
        r'\bI selled\b': 'I sold',
        r'\bI telled\b': 'I told',
        r'\bI buyed\b': 'I bought',
        r'\bI keeped\b': 'I kept',
        r'\bI sleeped\b': 'I slept',
        r'\bI feeled\b': 'I felt',
        r'\bI holded\b': 'I held',
        r'\bI maked\b': 'I made',
        r'\bI taked\b': 'I took',
        r'\bI getted\b': 'I got',
        r'\bI gived\b': 'I gave',
        
        # Common verb forms with other subjects
        r'\b(he|she|it) goed\b': r'\1 went',
        r'\b(he|she|it) runned\b': r'\1 ran',
        r'\b(he|she|it) buyed\b': r'\1 bought',
        r'\b(we|they) goed\b': r'\1 went',
        r'\b(we|they) buyed\b': r'\1 bought',
        r'\b(you) goed\b': r'\1 went',
        
        # Double negatives
        r"\bdon't\s+no\b": "don't know",
        r"\bdon't\s+nothing\b": "don't have anything",
        r"\bain't\s+no\b": "isn't any",
        r"\bcan't\s+no\b": "can't",
        r"\bwon't\s+no\b": "won't",
        r"\bdidn't\s+no\b": "didn't know",
        r"\bhasn't\s+no\b": "hasn't any",
        r"\bhaven't\s+no\b": "haven't any",
        
        # Article errors
        r'\ba\s+([aeiou])': r'an \1',
        r'\ban\s+([^aeiouAEIOU\s])': r'a \1',
        
        # Pronoun cases
        r'\bme and\s+(\w+)\s+(is|are|was|were|have|has|had)\b': r'\1 and I \2',
        r'\bhim and me\b': 'he and I',
        r'\bher and me\b': 'she and I',
        r'\bthem and me\b': 'they and I',
        r'\bme and him\b': 'he and I',
        r'\bme and her\b': 'she and I',
        r'\bme and them\b': 'they and I',
        
        # Common contractions mistakes
        r"\bshould of\b": "should have",
        r"\bcould of\b": "could have",
        r"\bwould of\b": "would have",
        r"\bmight of\b": "might have",
        r"\bmust of\b": "must have",
        
        # To/too/two confusion
        r'\bto\s+(much|many|often|soon|late)\b': r'too \1',
        r'\btoo\s+(the|a|an|my|your|his|her)\b': r'to \1',
        
        # There/their/they're
        r'\btheir\s+(is|are|was|were)\b': r'there \1',
        r'\bthey\'re\s+(house|car|dog|cat|book)\b': r'their \1',
        
        # Your/you're
        r'\byour\s+(a|an|the|very|so|really)\b': r"you're \1",
        r'\byou\'re\s+(house|car|dog|cat|book|name)\b': r'your \1',
        
        # Its/it's
        r"\bits\s+(a|an|the|very|so|really)\b": r"it's \1",
        r"\bit\'s\s+(color|size|shape|place|time)\b": r"its \1",
        
        # Then/than
        r'\bthen\s+(better|worse|more|less|bigger|smaller)\b': r'than \1',
        
        # Loose/lose
        r'\bloose\s+(the|my|your|his|her)\b': r'lose \1',
        
        # Affect/effect
        r'\beffect\s+(on|my|your|his|her|the)\b': r'affect \1',
        
        # Who/whom (simplified)
        r'\bwhom\s+(is|are|was|were)\b': r'who \1',
        
        # Verb agreement with singular/plural
        r'\b(everyone|everybody|someone|somebody|anyone|anybody|no one|nobody)\s+are\b': r'\1 is',
        r'\b(everyone|everybody|someone|somebody|anyone|anybody|no one|nobody)\s+were\b': r'\1 was',
        
        # Common misspellings that affect grammar
        r'\balot\b': 'a lot',
        r'\bnoone\b': 'no one',
        r'\beveryday\s+(I|we|they|he|she)\b': r'every day \1',
    }
    
    # Verb conjugation patterns
    VERB_PATTERNS = {
        'present_to_past': {
            'go': 'went', 'run': 'ran', 'buy': 'bought', 'eat': 'ate',
            'drink': 'drank', 'see': 'saw', 'build': 'built', 'think': 'thought',
            'speak': 'spoke', 'write': 'wrote', 'read': 'read', 'bring': 'brought',
            'teach': 'taught', 'catch': 'caught', 'fight': 'fought', 'sell': 'sold',
            'tell': 'told', 'keep': 'kept', 'sleep': 'slept', 'feel': 'felt',
            'hold': 'held', 'make': 'made', 'take': 'took', 'get': 'got', 'give': 'gave'
        }
    }
    
    def __init__(self, model_name: str = "t5-small", use_model: bool = False):
        """
        Initialize grammar corrector.
        
        Args:
            model_name: T5 model name (only loaded if use_model=True)
            use_model: Whether to load T5 model (slow, disabled for speed)
        """
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
        # Compile regex patterns for performance
        self.compiled_rules = {
            re.compile(pattern, re.IGNORECASE): replacement
            for pattern, replacement in self.GRAMMAR_RULES.items()
        }
        
        logger.info(f"Grammar corrector initialized in FAST mode (enhanced rules, {len(self.compiled_rules)} patterns)")
        
        if use_model:
            logger.warning("T5 model loading is disabled for optimal performance")
    
    @lru_cache(maxsize=512)
    def correct(self, text: str) -> str:
        """
        Correct grammar in text - OPTIMIZED & ENHANCED.
        
        Args:
            text: Input text
            
        Returns:
            Grammar-corrected text
        """
        if not text or not text.strip():
            return text
        
        # Apply comprehensive rule-based corrections
        corrected = self._apply_grammar_rules(text)
        
        # Apply additional smart corrections
        corrected = self._fix_verb_agreement(corrected)
        corrected = self._fix_capitalization(corrected)
        corrected = self._fix_spacing(corrected)
        corrected = self._fix_punctuation(corrected)
        
        return corrected
    
    def _apply_grammar_rules(self, text: str) -> str:
        """Apply all compiled grammar rules."""
        corrected = text
        
        for pattern, replacement in self.compiled_rules.items():
            corrected = pattern.sub(replacement, corrected)
        
        return corrected
    
    def _fix_verb_agreement(self, text: str) -> str:
        """Fix subject-verb agreement issues."""
        # Handle common patterns
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            if i > 0:
                prev_word = words[i-1].lower()
                curr_word = word.lower()
                
                # Fix "I is/are" -> "I am"
                if prev_word == 'i' and curr_word in ['is', 'are']:
                    result.append('am')
                    continue
                
                # Fix "you is" -> "you are"
                elif prev_word == 'you' and curr_word == 'is':
                    result.append('are')
                    continue
                
                # Fix "they is" -> "they are"
                elif prev_word in ['they', 'we'] and curr_word == 'is':
                    result.append('are')
                    continue
            
            result.append(word)
        
        return ' '.join(result)
    
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
        
        # Capitalize common contractions with I
        text = re.sub(r"\bi'm\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi've\b", "I've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'll\b", "I'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'd\b", "I'd", text, flags=re.IGNORECASE)
        
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Fix spaces after punctuation
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
        
        # Fix spacing around quotes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        return text
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation issues."""
        # Remove multiple punctuation marks
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Fix comma usage
        text = re.sub(r',\s*,', ',', text)
        
        return text
    
    def get_stats(self) -> Dict[str, int]:
        """Get correction statistics."""
        return {
            'total_rules': len(self.compiled_rules),
            'cache_size': self.correct.cache_info().currsize,
            'cache_hits': self.correct.cache_info().hits,
            'cache_misses': self.correct.cache_info().misses
        }