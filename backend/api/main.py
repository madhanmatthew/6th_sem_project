"""
AI Multi-Agent Emergency Coordination System
FastAPI Backend
"""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents import ambulance_agent, coordinator, hospital_agent, police_agent, traffic_agent
from backend.ml.eta_model import MODEL_FEATURES, TRAINING_SAMPLES, train_eta_model
from backend.ml.ml_models import apply_severity_model, train_ml_models
from backend.data.report_parser import parse_report

# ── ORS API KEY ─────────────────────────────────────────────────────────────
# Paste your key from openrouteservice.org here OR set env var ORS_API_KEY
# Get free key at: https://openrouteservice.org/dev/#/signup
os.environ.setdefault("ORS_API_KEY", "PASTE_YOUR_ORS_KEY_HEREeyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImM1YTdjZmQzZjljNjQzZmJhMzQ4YjAyNjQ2YTkwMzU2IiwiaCI6Im11cm11cjY0In0=")

app = FastAPI(title="AI Emergency Coordination System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("[Startup] Training XGBoost ETA model...")
eta_model, model_rmse = train_eta_model()
print("[Startup] Model ready.")

print("[Startup] Training AIML agent models...")
agent_ml_models, agent_ml_metrics = train_ml_models()
print("[Startup] AIML agent models ready.")


def _open_browser():
    time.sleep(1.4)
    url = "http://localhost:8000"
    try:
        if sys.platform == "win32":
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
    except Exception:
        pass

threading.Thread(target=_open_browser, daemon=True).start()


class ReportRequest(BaseModel):
    text: str


def run_dispatch(text: str) -> dict:
    parsed       = apply_severity_model(parse_report(text), agent_ml_models["severity"])
    ambulance    = ambulance_agent(parsed, eta_model, model_rmse)
    traffic      = traffic_agent(parsed, agent_ml_models["traffic"])
    police       = police_agent(parsed)
    hospital     = hospital_agent(parsed, agent_ml_models["hospital"])
    coordination = coordinator(ambulance, traffic, police, hospital)
    return {
        "parsed": parsed,
        "agents": {"ambulance": ambulance, "traffic": traffic, "police": police, "hospital": hospital},
        "coordination": coordination,
    }


@app.post("/dispatch")
def dispatch(req: ReportRequest):
    return run_dispatch(req.text)

@app.get("/dispatch")
def dispatch_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static", "index.html"))

@app.get("/map")
def map_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static", "map.html"))

@app.get("/results")
def results_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static", "results.html"))

@app.get("/model-info")
def model_info():
    return {
        "model": "GradientBoostingRegressor",
        "training_samples": TRAINING_SAMPLES,
        "rmse_minutes": round(model_rmse, 2),
        "features": MODEL_FEATURES,
        "agent_models": agent_ml_metrics,
    }

@app.websocket("/ws/dispatch")
async def ws_dispatch(ws: WebSocket):
    await ws.accept()
    try:
        data    = await ws.receive_text()
        payload = json.loads(data)
        text    = payload.get("text", "")
        parsed  = apply_severity_model(parse_report(text), agent_ml_models["severity"])
        await ws.send_text(json.dumps({"event": "parsed", "data": parsed}))
        await asyncio.sleep(0.3)
        ambulance = ambulance_agent(parsed, eta_model, model_rmse)
        await ws.send_text(json.dumps({"event": "agent", "data": ambulance}))
        await asyncio.sleep(0.5)
        traffic = traffic_agent(parsed, agent_ml_models["traffic"])
        await ws.send_text(json.dumps({"event": "agent", "data": traffic}))
        await asyncio.sleep(0.5)
        police = police_agent(parsed)
        await ws.send_text(json.dumps({"event": "agent", "data": police}))
        await asyncio.sleep(0.5)
        hospital = hospital_agent(parsed, agent_ml_models["hospital"])
        await ws.send_text(json.dumps({"event": "agent", "data": hospital}))
        await asyncio.sleep(0.7)
        coordination = coordinator(ambulance, traffic, police, hospital)
        await ws.send_text(json.dumps({"event": "coordination", "data": coordination}))
        await ws.send_text(json.dumps({"event": "done"}))
    except WebSocketDisconnect:
        pass

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static", "home.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
