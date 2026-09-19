"""
BioPrint: High-Speed Behavioral Biometric Authentication Server (patched)
Serves REST APIs, WebSocket real-time telemetry stream, and frontend web portal.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .biometrics import (
    AdaptiveProfileUpdater,
    BiometricExplainer,
    BiometricModel,
    BotDetector,
    FeatureExtractor,
)
from .biometrics.bot_detector import ReplayGuard
from .database import ProfileStore

app = FastAPI(
    title="BioPrint Behavioral Biometric Authentication API",
    description="Passwordless-proof identity verification powered by behavioral biometrics",
    version="1.0.0",
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
replay_guard = ReplayGuard()

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
    nonce: Optional[str] = None


# API Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "engine": "BioPrint Hybrid Biometric Core",
        "timestamp": time.time(),
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


@app.post("/api/login-challenge")
async def login_challenge():
    return replay_guard.issue()


@app.post("/api/enroll")
async def enroll_user(req: EnrollmentRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    if len(req.sessions) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 calibration sessions are required for baseline enrollment",
        )

    feature_vectors, digraph_samples, masks, sequences = [], [], [], []
    for s in req.sessions:
        f = feature_extractor.extract_all(s.keystrokes, s.mouse)
        feature_vectors.append(f["vector"])
        digraph_samples.append(f["digraphs"])
        masks.append(f["mask"])
        sequences.append(f["sequences"])

    profile_model = biometric_model.fit_profile(
        username=req.username,
        feature_vectors=feature_vectors,
        digraph_samples=digraph_samples,
        masks=masks,
        sequences=sequences,
    )
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
            "curvature_ratio": profile_model["mean"][5],
        },
    }


@app.post("/api/authenticate")
async def authenticate(req: AuthRequest):
    t_start = time.perf_counter()

    profile = profile_store.get_profile(req.username)
    if not profile:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "decision": "UNKNOWN_USER", "message": f"User '{req.username}' not enrolled."},
        )

    # 1. Password Verification (BioPrint verifies behavior EVEN IF password is valid)
    pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    if pw_hash != profile.get("password_hash"):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "decision": "INVALID_PASSWORD", "message": "Incorrect password credentials."},
        )

    bio_profile = profile.get("biometric_profile", {})

    # 2. Extract Features
    f = feature_extractor.extract_all(req.keystrokes, req.mouse)

    # 3. Check Replay Nonce & Detect Bots
    replay = replay_guard.check(req.username, req.keystrokes, nonce=req.nonce, enforce_nonce=bool(req.nonce))

    is_bot, bot_prob, bot_reasons = bot_detector.evaluate_telemetry(
        raw_keystrokes=req.keystrokes,
        raw_mouse=req.mouse,
        keystroke_features=f["keystroke"],
        mouse_features=f["mouse"],
        client_metadata=req.client_metadata or {},
        replay_result=replay,
    )

    t_eval = time.perf_counter()

    if is_bot:
        decision = "BLOCKED_BOT"
        confidence_score = 0.0
        metrics = {"bot_probability": bot_prob, "reasons": bot_reasons}
    else:
        # 4. Behavioral Biometric Match
        is_genuine, confidence_score, metrics = biometric_model.evaluate_attempt(
            username=req.username,
            attempt_vector=f["vector"],
            attempt_digraphs=f["digraphs"],
            profile_data=bio_profile,
            mask=f["mask"],
            sequences=f["sequences"],
        )

        if is_genuine:
            decision = "AUTHENTICATED"
            # 5. Adaptive Drift Update (Stretch Goal 1)
            updated_profile = adaptive_updater.update_profile(
                profile_data=bio_profile,
                confirmed_vector=f["vector"],
                confirmed_digraphs=f["digraphs"],
                confidence_score=confidence_score,
                confirmed_mask=f["mask"],
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
        keystroke_features=f["keystroke"],
        mouse_features=f["mouse"],
        profile_data=bio_profile,
    )

    # Full payload for audit log and WebSocket live dashboard
    dashboard_payload = {
        "status": "success",
        "decision": decision,
        "authenticated": (decision == "AUTHENTICATED"),
        "confidence_score": confidence_score,
        "latency_ms": latency_ms,
        "metrics": metrics,
        "explanation": explanation,
        "telemetry_summary": {
            "keystroke": f["keystroke"],
            "mouse": f["mouse"],
        },
    }

    # Log to audit store
    profile_store.log_attempt({
        "timestamp": time.time(),
        "username": req.username,
        "decision": decision,
        "confidence_score": confidence_score,
        "latency_ms": latency_ms,
        "headline": explanation["headline"],
        "badge_color": explanation["badge_color"],
    })

    # Broadcast to live visual HUD via WebSocket
    await broadcast_telemetry(dashboard_payload)

    # Public payload for HTTP client (stop leaking metrics on blocked responses)
    public = biometric_model.public_metrics(metrics, decision == "AUTHENTICATED")
    result_payload = {
        "status": "success",
        "decision": decision,
        "authenticated": (decision == "AUTHENTICATED"),
        "confidence_score": confidence_score if decision == "AUTHENTICATED" else None,
        "latency_ms": latency_ms,
        "metrics": public,
        "explanation": explanation,
    }
    if decision == "AUTHENTICATED":
        result_payload["telemetry_summary"] = {
            "keystroke": f["keystroke"],
            "mouse": f["mouse"],
        }

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
