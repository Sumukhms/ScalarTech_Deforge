"""Auto-formatting engine - ENHANCED with smart NLP."""
import re
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class AutoFormatter:
    """Auto-formatting with smart sentence detection."""
    
    # Common abbreviations that shouldn't trigger sentence breaks
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
        'etc', 'vs', 'inc', 'ltd', 'co', 'corp',
        'st', 'ave', 'blvd', 'dept', 'est',
        'i.e', 'e.g', 'a.m', 'p.m'
    }
    
    def __init__(self):
        """Initialize formatter."""
        # Compile regex patterns for performance
        self.sentence_endings = re.compile(r'[.!?]+\s*')
        self.capitalize_after = re.compile(r'([.!?]\s+)([a-z])')
        self.abbreviation_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(abbr) for abbr in self.ABBREVIATIONS) + r')\.',
            re.IGNORECASE
        )
    
    def format(self, text: str) -> str:
        """
        Apply comprehensive formatting.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text
        """
        if not text or not text.strip():
            return text
        
        # Apply formatting steps in order
        formatted = text
        
        # 1. Smart sentence segmentation
        formatted = self.segment_sentences(formatted)
        
        # 2. Fix capitalization
        formatted = self.fix_capitalization(formatted)
        
        # 3. Fix punctuation
        formatted = self.fix_punctuation(formatted)
        
        # 4. Add paragraphs if text is long
        if len(formatted) > 200:
            formatted = self.add_paragraphs(formatted)
        
        # 5. Final cleanup
        formatted = self.cleanup_spacing(formatted)
        
        return formatted.strip()
    
    def segment_sentences(self, text: str) -> str:
        """
        Smart sentence segmentation.
        
        Args:
            text: Input text
            
        Returns:
            Text with proper sentence breaks
        """
        # Handle missing periods at end
        if text and text[-1] not in '.!?':
            text = text.rstrip() + '.'
        
        # Normalize multiple punctuation
        text = re.sub(r'[.!?]{2,}', lambda m: m.group(0)[0], text)
        
        # Ensure space after sentence endings (but not abbreviations)
        # This is tricky - need to avoid breaking "Dr. Smith" etc.
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text
    
    def fix_capitalization(self, text: str) -> str:
        """
        Fix capitalization throughout text.
        
        Args:
            text: Input text
            
        Returns:
            Text with proper capitalization
        """
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        text = self.capitalize_after.sub(
            lambda m: m.group(1) + m.group(2).upper(),
            text
        )
        
        # Capitalize "I" as pronoun
        text = re.sub(r'\bi\b', 'I', text)
        
        # Capitalize common contractions with I
        text = re.sub(r"\bi'm\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi've\b", "I've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'll\b", "I'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'd\b", "I'd", text, flags=re.IGNORECASE)
        
        return text
    
    def fix_punctuation(self, text: str) -> str:
        """
        Fix punctuation issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed punctuation
        """
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix comma spacing
        text = re.sub(r',([^\s])', r', \1', text)
        
        # Remove space after opening quotes/parentheses
        text = re.sub(r'([("])\s+', r'\1', text)
        
        # Remove space before closing quotes/parentheses
        text = re.sub(r'\s+([)"])', r'\1', text)
        
        return text
    
    def add_paragraphs(self, text: str, max_sentences: int = 4) -> str:
        """
        Add paragraph breaks for readability.
        
        Args:
            text: Input text
            max_sentences: Max sentences per paragraph
            
        Returns:
            Text with paragraph breaks
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+\s*)', text)
        
        # Reconstruct sentences with punctuation
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    full_sentences.append(sentence.strip())
        
        if not full_sentences:
            return text
        
        # Group into paragraphs
        paragraphs = []
        current_para = []
        
        for i, sentence in enumerate(full_sentences):
            current_para.append(sentence)
            
            # Start new paragraph after max_sentences or at natural breaks
            if len(current_para) >= max_sentences:
                paragraphs.append(' '.join(current_para))
                current_para = []
        
        # Add remaining sentences
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Join paragraphs with double newline
        return '\n\n'.join(paragraphs)
    
    def cleanup_spacing(self, text: str) -> str:
        """
        Clean up spacing issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with cleaned spacing
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove spaces at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        
        # Normalize multiple newlines to double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def format_as_bullets(self, text: str) -> str:
        """
        Convert text to bullet list format.
        
        Args:
            text: Input text
            
        Returns:
            Bullet-formatted text
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        bullets = []
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if sentence and len(sentence) > 5:
                bullets.append(f"• {sentence}")
        
        return '\n'.join(bullets)
    
    def smart_capitalize_title(self, text: str) -> str:
        """
        Capitalize text as a title (skip small words).
        
        Args:
            text: Input text
            
        Returns:
            Title-cased text
        """
        # Words to not capitalize (unless first/last)
        small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for',
                      'in', 'of', 'on', 'or', 'the', 'to', 'with'}
        
        words = text.split()
        if not words:
            return text
        
        # Capitalize first and last word always
        capitalized = [words[0].capitalize()]
        
        for word in words[1:-1]:
            if word.lower() in small_words:
                capitalized.append(word.lower())
            else:
                capitalized.append(word.capitalize())
        
        if len(words) > 1:
            capitalized.append(words[-1].capitalize())
        
        return ' '.join(capitalized)