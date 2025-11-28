"""Fast configuration for <500ms latency."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class Config:
    """Fast configuration."""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")  # Less logging = faster
    
    # Use SMALL, FAST Vosk model
    VOSK_MODEL_PATH = os.getenv(
        "VOSK_MODEL_PATH", 
        str(MODELS_DIR / "vosk-model-small-en-us-0.15")
    )
    
    # NO grammar model for speed
    GRAMMAR_MODEL_NAME = None
    
    # Audio settings
    MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "60"))
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8000"))
    
    # Fast latency target
    TARGET_LATENCY_MS = int(os.getenv("TARGET_LATENCY_MS", "500"))
    PAUSE_DETECTION_MS = int(os.getenv("PAUSE_DETECTION_MS", "500"))
    
    # Processing
    MAX_TEXT_LENGTH = 5000
    BATCH_SIZE = 16

config = Config()
