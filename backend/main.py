# main.py — FastAPI Backend for Emotion Analytics
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from logic import EmotionAnalyzer


# =====================================================
#                  APP LIFECYCLE
# =====================================================

analyzer: EmotionAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, cleanup on shutdown."""
    global analyzer
    print("[FastAPI] Loading EmotionAnalyzer...")
    start = time.time()
    analyzer = EmotionAnalyzer()
    print(f"[FastAPI] Models loaded in {time.time() - start:.1f}s on {analyzer.device}")
    yield
    print("[FastAPI] Shutting down...")
    analyzer = None


app = FastAPI(
    title="Emotion Analytics API",
    description="Real-time emotion detection via WebSocket",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow Vercel frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
#                  REST ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    """Root endpoint for HF Spaces health check."""
    return {"status": "ok", "service": "Emotion Analytics API"}


@app.get("/api/health")
async def health():
    """Health check with model status and device info."""
    if analyzer is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Models are still loading..."},
        )
    return {
        "status": "ready",
        "device": str(analyzer.device),
        "labels": analyzer.labels,
    }


# =====================================================
#               WEBSOCKET ENDPOINT
# =====================================================

@app.websocket("/ws/emotion")
async def websocket_emotion(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion inference.

    Protocol:
    - Client sends: base64-encoded JPEG frame (text message)
    - Server responds: JSON with emotion, score, bbox, probabilities
    """
    await websocket.accept()
    print("[WS] Client connected")

    if analyzer is None:
        await websocket.send_json({"error": "Models not loaded yet"})
        await websocket.close()
        return

    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()

            # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,...")
            if "," in data:
                data = data.split(",", 1)[1]

            # Run inference in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(analyzer.process_base64_frame, data)

            # Send result back
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# =====================================================
#                  MAIN ENTRY
# =====================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
