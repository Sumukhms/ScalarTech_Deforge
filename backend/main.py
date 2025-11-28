"""FastAPI application - ENHANCED with WebSocket Streaming for <500ms Latency."""
import os
import sys
import tempfile
import warnings
import traceback
import json
from pathlib import Path
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
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
    allow_origins=["*"],
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
            "websocket_stream": "/ws/transcribe",
            "text_processing": "/process/full"
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

# === WEBSOCKET STREAMING ENDPOINT ===

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time audio streaming endpoint.
    Receives raw PCM audio chunks, performs live STT, and runs full processing on completion.
    """
    await websocket.accept()
    
    if processor is None or processor.stt_engine is None:
        await websocket.close(code=1011, reason="STT Engine not available")
        return

    # Create a recognizer for this session (16kHz)
    try:
        rec = processor.stt_engine.create_recognizer(16000)
    except Exception as e:
        logger.error(f"Failed to create recognizer: {e}")
        await websocket.close(code=1011, reason="STT Init Failed")
        return

    full_transcript = []
    tone = "neutral"

    try:
        while True:
            # Receive message (bytes for audio, text for commands)
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                # Process audio chunk
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                    if text:
                        full_transcript.append(text)
                        # Send finalized sentence to client immediately
                        await websocket.send_json({"type": "final", "text": text})
                else:
                    # Send partial result for real-time feedback
                    partial = json.loads(rec.PartialResult())
                    if partial.get("partial"):
                        await websocket.send_json({"type": "partial", "text": partial["partial"]})

            elif "text" in message and message["text"]:
                # Handle control messages (e.g., stop recording with specific tone)
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "eof":
                        tone = data.get("tone", "neutral")
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    # Process final buffer
    final_res = json.loads(rec.FinalResult())
    if final_res.get("text"):
        full_transcript.append(final_res["text"])

    # Combine all parts
    original_text = " ".join(full_transcript).strip()
    logger.info(f"Full Transcript: {original_text[:50]}...")

    # Run the cleaning pipeline
    if original_text:
        processed_result = processor.process_full(original_text, tone=tone, track_latency=True)
        
        # Send complete result back to client
        await websocket.send_json({
            "type": "complete",
            "original_text": original_text,
            "processed_text": processed_result["processed_text"],
            "latency": processed_result.get("latency", {}),
            "improvement": processed_result.get("improvement", {})
        })
    else:
         await websocket.send_json({
            "type": "error",
            "message": "No speech detected"
        })

    await websocket.close()

# === TEXT PROCESSING ENDPOINTS ===

@app.post("/process/full", response_model=ProcessResponse)
async def process_full_pipeline(
    text: str = Form(...),
    tone: ToneMode = Form(default="neutral")
):
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    try:
        result = processor.process_full(text.strip(), tone=tone, track_latency=True)
        return ProcessResponse(**result)
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ... (Keep other individual step endpoints /process/fillers etc. if needed) ...

# === LEGACY VOICE ENDPOINT (Fallback) ===

@app.post("/voice/record-and-process")
async def voice_record_and_process(audio: UploadFile = File(...), tone: str = Form(default="neutral")):
    # Reuse existing logic for file uploads as a fallback
    if processor is None or processor.stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not available")
    
    tmp_path = None
    try:
        import time
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(await audio.read())
            tmp_path = tmp_file.name
        
        transcript = processor.transcribe_audio(tmp_path)
        result = processor.process_full(transcript, tone=tone, track_latency=True)
        
        # Add latency info
        total_ms = (time.perf_counter() - start) * 1000
        if "latency" not in result: result["latency"] = {}
        result["latency"]["total_pipeline_ms"] = total_ms
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")