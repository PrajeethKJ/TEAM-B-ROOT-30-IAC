"""
BioPrint: High-Speed Behavioral Biometric Authentication Server
Serves REST APIs, WebSocket real-time telemetry stream, and frontend web portal.
"""

import time
import hashlib
import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .biometrics import (
    FeatureExtractor,
    BotDetector,
    BiometricModel,
    AdaptiveProfileUpdater,
    BiometricExplainer
)
from .database import ProfileStore

app = FastAPI(
    title="BioPrint Behavioral Biometric Authentication API",
    description="Passwordless-proof identity verification powered by behavioral biometrics",
    version="1.0.0"
)

# Enable CORS for local web and Chrome extension content scripts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Instances
feature_extractor = FeatureExtractor()
bot_detector = BotDetector()
biometric_model = BiometricModel()
adaptive_updater = AdaptiveProfileUpdater()
explainer = BiometricExplainer()
profile_store = ProfileStore()

# WebSocket client manager for live visual telemetry
active_websockets: List[WebSocket] = []


async def broadcast_telemetry(data: Dict[str, Any]):
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


# Pydantic Schemas
class EnrollmentSession(BaseModel):
    keystrokes: List[Dict[str, Any]]
    mouse: List[Dict[str, Any]]
    client_metadata: Optional[Dict[str, Any]] = {}


class EnrollmentRequest(BaseModel):
    username: str
    password: str
    sessions: List[EnrollmentSession]


class AuthRequest(BaseModel):
    username: str
    password: str
    keystrokes: List[Dict[str, Any]]
    mouse: List[Dict[str, Any]]
    client_metadata: Optional[Dict[str, Any]] = {}


# API Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "engine": "BioPrint Hybrid Biometric Core",
        "timestamp": time.time()
    }


@app.get("/api/profiles")
async def list_profiles():
    profiles = profile_store.list_profiles()
    return {"profiles": profiles}


@app.get("/api/audit-logs")
async def get_audit_logs():
    logs = profile_store.get_audit_logs()
    return {"logs": logs}


@app.post("/api/reset")
async def reset_system():
    profile_store.reset_all()
    return {"status": "success", "message": "All demo profiles and audit logs have been reset."}


@app.post("/api/enroll")
async def enroll_user(req: EnrollmentRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    if len(req.sessions) < 2:
        raise HTTPException(status_code=400, detail="At least 2 calibration sessions are required for baseline enrollment")

    feature_vectors = []
    digraph_samples = []

    for s in req.sessions:
        k_feat = feature_extractor.extract_keystroke_features(s.keystrokes)
        m_feat = feature_extractor.extract_mouse_features(s.mouse)
        vec = feature_extractor.build_feature_vector(k_feat, m_feat)
        feature_vectors.append(vec)
        digraph_samples.append(k_feat.get("digraph_latencies", {}))

    # Fit baseline model
    profile_model = biometric_model.fit_profile(feature_vectors, digraph_samples)
    pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()

    profile_store.save_profile(req.username, pw_hash, profile_model)

    return {
        "status": "enrolled",
        "username": req.username,
        "sample_count": len(req.sessions),
        "baseline_summary": {
            "mean_dwell_ms": profile_model["mean"][0],
            "mean_flight_ms": profile_model["mean"][2],
            "cps": profile_model["mean"][4],
            "curvature_ratio": profile_model["mean"][5]
        }
    }


@app.post("/api/authenticate")
async def authenticate(req: AuthRequest):
    t_start = time.perf_counter()

    profile = profile_store.get_profile(req.username)
    if not profile:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "decision": "UNKNOWN_USER", "message": f"User '{req.username}' not enrolled."}
        )

    # 1. Password Verification (BioPrint verifies behavior EVEN IF password is valid)
    pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    if pw_hash != profile.get("password_hash"):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "decision": "INVALID_PASSWORD", "message": "Incorrect password credentials."}
        )

    bio_profile = profile.get("biometric_profile", {})

    # 2. Extract Features
    k_feat = feature_extractor.extract_keystroke_features(req.keystrokes)
    m_feat = feature_extractor.extract_mouse_features(req.mouse)
    attempt_vec = feature_extractor.build_feature_vector(k_feat, m_feat)
    attempt_digraphs = k_feat.get("digraph_latencies", {})

    # 3. Detect Bots & Scripted Fraud
    is_bot, bot_prob, bot_reasons = bot_detector.evaluate_telemetry(
        raw_keystrokes=req.keystrokes,
        raw_mouse=req.mouse,
        keystroke_features=k_feat,
        mouse_features=m_feat,
        client_metadata=req.client_metadata or {}
    )

    t_eval = time.perf_counter()

    if is_bot:
        decision = "BLOCKED_BOT"
        confidence_score = 0.0
        metrics = {"bot_probability": bot_prob, "reasons": bot_reasons}
    else:
        # 4. Behavioral Biometric Match
        is_genuine, confidence_score, metrics = biometric_model.evaluate_attempt(
            attempt_vector=attempt_vec,
            attempt_digraphs=attempt_digraphs,
            profile_data=bio_profile
        )

        if is_genuine:
            decision = "AUTHENTICATED"
            # 5. Adaptive Drift Update (Stretch Goal 1)
            updated_profile = adaptive_updater.update_profile(
                profile_data=bio_profile,
                confirmed_vector=attempt_vec,
                confirmed_digraphs=attempt_digraphs,
                confidence_score=confidence_score
            )
            profile_store.update_profile_biometrics(req.username, updated_profile)
        else:
            decision = "BLOCKED_IMPOSTOR"

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    # 6. Generate Explainability Report (Stretch Goal 4)
    explanation = explainer.explain_decision(
        decision=decision,
        confidence_score=confidence_score,
        bot_reasons=bot_reasons if is_bot else [],
        metrics=metrics,
        keystroke_features=k_feat,
        mouse_features=m_feat,
        profile_data=bio_profile
    )

    result_payload = {
        "status": "success",
        "decision": decision,
        "authenticated": (decision == "AUTHENTICATED"),
        "confidence_score": confidence_score,
        "latency_ms": latency_ms,
        "metrics": metrics,
        "explanation": explanation,
        "telemetry_summary": {
            "keystroke": k_feat,
            "mouse": m_feat
        }
    }

    # Log to audit store
    profile_store.log_attempt({
        "timestamp": time.time(),
        "username": req.username,
        "decision": decision,
        "confidence_score": confidence_score,
        "latency_ms": latency_ms,
        "headline": explanation["headline"],
        "badge_color": explanation["badge_color"]
    })

    # Broadcast to live visual HUD via WebSocket
    await broadcast_telemetry(result_payload)

    return result_payload


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # Keep-alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# Mount Static Files (Frontend Web Portal)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
