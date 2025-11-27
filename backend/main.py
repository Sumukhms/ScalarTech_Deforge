"""FastAPI application - ENHANCED with better error handling and validation."""
import os
import sys
import tempfile
import warnings
import traceback
from pathlib import Path
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pipeline.processor import DictationProcessor
from modules.tone import ToneMode
from utils.logger import get_logger
from config import config

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Speech Dictation Engine",
    description="Real-time speech dictation with cleaning, grammar correction, and tone control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor (singleton)
processor: Optional[DictationProcessor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup."""
    global processor
    try:
        logger.info("=" * 60)
        logger.info("Starting Intelligent Speech Dictation Engine")
        logger.info("=" * 60)
        processor = DictationProcessor()
        logger.info("✓ Dictation engine ready")
        logger.info(f"✓ Target latency: ≤{config.TARGET_LATENCY_MS}ms")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Failed to initialize processor: {e}")
        traceback.print_exc()
        processor = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down dictation engine...")

# Request/Response models
class TextInput(BaseModel):
    text: str = Field(..., description="Input text to process", min_length=1)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long (max {config.MAX_TEXT_LENGTH} chars)")
        return v.strip()

class ToneInput(BaseModel):
    text: str = Field(..., description="Input text")
    mode: ToneMode = Field(default="neutral", description="Tone mode")

class ProcessResponse(BaseModel):
    original_text: str
    processed_text: str
    tone: Optional[str] = None
    improvement: Optional[dict] = None
    latency: Optional[dict] = None
    error: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Intelligent Speech Dictation Engine",
        "version": "1.0.0",
        "status": "running",
        "target_latency": f"≤{config.TARGET_LATENCY_MS}ms",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "voice_processing": "/voice/record-and-process",
            "text_processing": "/process/full",
            "individual_steps": [
                "/process/fillers",
                "/process/repetition",
                "/process/grammar",
                "/process/tone"
            ]
        }
    }

@app.get("/health")
async def health_check():
    """Health check with detailed status."""
    if processor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "processor_initialized": False,
                "error": "Processor not initialized"
            }
        )
    
    stats = processor.get_processing_stats()
    
    return {
        "status": "healthy",
        "processor_initialized": True,
        "stt_available": stats["stt_available"],
        "modules": stats["modules_loaded"],
        "target_latency_ms": stats["target_latency_ms"],
        "grammar_mode": stats["grammar_mode"]
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path
        }
    )

# === TEXT PROCESSING ENDPOINTS ===

@app.post("/process/full", response_model=ProcessResponse)
async def process_full_pipeline(
    text: str = Form(...),
    tone: ToneMode = Form(default="neutral")
):
    """
    Process text through full pipeline - MAIN ENDPOINT.
    
    Returns cleaned, formatted, tone-adjusted text with latency metrics.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        # Validate input
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(text) > config.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long (max {config.MAX_TEXT_LENGTH} chars)"
            )
        
        # Process through pipeline
        result = processor.process_full(text.strip(), tone=tone, track_latency=True)
        
        return ProcessResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/full-json")
async def process_full_pipeline_json(input_data: dict):
    """Process text through full pipeline (JSON body)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        text = input_data.get("text", "").strip()
        tone = input_data.get("tone", "neutral")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        result = processor.process_full(text, tone=tone, track_latency=True)
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# === INDIVIDUAL STEP ENDPOINTS ===

@app.post("/process/fillers", response_model=ProcessResponse)
async def remove_fillers(input_data: TextInput):
    """Remove filler words and disfluencies."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        processed = processor.process_step(input_data.text, "fillers")
        
        return ProcessResponse(
            original_text=input_data.text,
            processed_text=processed,
            improvement={
                "original_length": len(input_data.text),
                "processed_length": len(processed),
                "reduction_percent": round(
                    (len(input_data.text) - len(processed)) / len(input_data.text) * 100, 2
                ) if input_data.text else 0
            }
        )
    except Exception as e:
        logger.error(f"Filler removal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/repetition", response_model=ProcessResponse)
async def remove_repetition(input_data: TextInput):
    """Remove repeated phrases and segments."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        processed = processor.process_step(input_data.text, "repetition")
        
        return ProcessResponse(
            original_text=input_data.text,
            processed_text=processed,
            improvement={
                "original_length": len(input_data.text),
                "processed_length": len(processed),
                "reduction_percent": round(
                    (len(input_data.text) - len(processed)) / len(input_data.text) * 100, 2
                ) if input_data.text else 0
            }
        )
    except Exception as e:
        logger.error(f"Repetition removal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/grammar", response_model=ProcessResponse)
async def correct_grammar(input_data: TextInput):
    """Correct grammar and sentence structure."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        processed = processor.process_step(input_data.text, "grammar")
        
        return ProcessResponse(
            original_text=input_data.text,
            processed_text=processed,
            improvement={
                "original_length": len(input_data.text),
                "processed_length": len(processed)
            }
        )
    except Exception as e:
        logger.error(f"Grammar correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/tone", response_model=ProcessResponse)
async def transform_tone(input_data: ToneInput):
    """Transform text tone and style."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        processed = processor.process_step(input_data.text, "tone", mode=input_data.mode)
        
        return ProcessResponse(
            original_text=input_data.text,
            processed_text=processed,
            tone=input_data.mode
        )
    except Exception as e:
        logger.error(f"Tone transformation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === VOICE PROCESSING ENDPOINTS ===

@app.post("/voice/record-and-process")
async def voice_record_and_process(
    audio: UploadFile = File(...),
    tone: str = Form(default="neutral")
):
    """
    Record voice, transcribe, and process through full pipeline.
    
    This is the MAIN endpoint for voice input.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if processor.stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    tmp_path = None
    
    try:
        import time
        pipeline_start = time.perf_counter()
        
        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Get file extension
        file_extension = os.path.splitext(audio.filename)[1] or ".wav"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await audio.read()
            
            # Validate audio size
            if len(content) < 100:
                raise HTTPException(status_code=400, detail="Audio file too small")
            
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Step 1: Transcribe audio
        transcription_start = time.perf_counter()
        transcript = ""
        
        try:
            transcript = processor.transcribe_audio(tmp_path)
            transcription_latency = (time.perf_counter() - transcription_start) * 1000
            
            logger.info(f"Transcription: {transcription_latency:.1f}ms - '{transcript[:50]}...'")
            
            if not transcript or not transcript.strip():
                logger.warning("No speech detected in audio")
                transcript = ""
                
        except Exception as transcribe_error:
            logger.error(f"Transcription failed: {transcribe_error}", exc_info=True)
            
            # Check for format errors
            error_msg = str(transcribe_error)
            if "RIFF" in error_msg or "format" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Invalid audio format. Please use WAV format (16kHz, mono)."
                )
            
            # For other errors, continue with empty transcript
            transcript = ""
            transcription_latency = (time.perf_counter() - transcription_start) * 1000
        
        # Step 2: Process through pipeline
        try:
            result = processor.process_full(
                transcript if transcript else "",
                tone=tone,
                track_latency=True
            )
        except Exception as process_error:
            logger.error(f"Processing error: {process_error}", exc_info=True)
            result = {
                "original_text": transcript,
                "processed_text": transcript,
                "tone": tone,
                "improvement": {
                    "original_length": len(transcript),
                    "processed_length": len(transcript),
                    "reduction_percent": 0
                },
                "latency": {}
            }
        
        # Calculate total latency
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        
        # Add transcription info
        result["transcription"] = transcript
        result["source"] = "voice"
        
        # Update latency info
        if "latency" not in result:
            result["latency"] = {}
        
        result["latency"]["transcription_ms"] = round(transcription_latency, 2)
        result["latency"]["total_pipeline_ms"] = round(total_latency, 2)
        result["latency"]["meets_target"] = total_latency <= config.TARGET_LATENCY_MS
        
        # Handle empty transcript
        if not transcript:
            result["processed_text"] = ""
            result["improvement"] = {
                "original_length": 0,
                "processed_length": 0,
                "reduction_percent": 0
            }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

# === STREAMING ENDPOINT (Basic) ===

@app.post("/asr/stream", response_model=ProcessResponse)
async def stream_transcription(audio: UploadFile = File(...)):
    """
    Stream speech and get transcripts (simplified version).
    
    Note: True streaming requires WebSockets. This is a simplified version.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if processor.stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    tmp_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Transcribe
        transcript = processor.transcribe_audio(tmp_path)
        
        return ProcessResponse(
            original_text=transcript,
            processed_text=transcript,
            improvement={
                "original_length": len(transcript),
                "processed_length": len(transcript),
                "reduction_percent": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )