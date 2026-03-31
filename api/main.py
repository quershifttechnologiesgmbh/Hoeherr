"""
hoeherr API - REST API for video upload and match analysis.
Also serves the web frontend.
"""
import json
import uuid
import shutil
import asyncio
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
ROOT_PATH = os.environ.get("ROOT_PATH", "")
app = FastAPI(title="hoeherr - Football Analysis", version="0.1.0", root_path=ROOT_PATH)

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.middleware("http")
async def add_base_path(request: Request, call_next):
    """Make root_path available to all templates."""
    request.state.base = request.scope.get("root_path", "")
    response = await call_next(request)
    return response

# In-memory job tracking
jobs = {}


# ─── Web UI Routes ───

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page — upload form + job list."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "jobs": jobs,
    })


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    """Job detail page with results."""
    if job_id not in jobs:
        return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

    job = jobs[job_id]
    report = None
    if job["status"] == "completed":
        report_path = RESULTS_DIR / job_id / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

    return templates.TemplateResponse("job_detail.html", {
        "request": request,
        "job_id": job_id,
        "job": job,
        "report": report,
    })


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """Upload a video and start analysis."""
    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": video.filename,
        "created_at": datetime.now().isoformat(),
        "video_path": str(video_path),
        "error": None,
    }

    background_tasks.add_task(process_video_task, job_id, str(video_path))
    return RedirectResponse(url=f"{ROOT_PATH}/jobs/{job_id}", status_code=303)


# ─── API Routes ───

@app.post("/api/v1/analyze")
async def api_analyze(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """API: Upload video and start analysis."""
    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": video.filename,
        "created_at": datetime.now().isoformat(),
        "video_path": str(video_path),
        "error": None,
    }

    background_tasks.add_task(process_video_task, job_id, str(video_path))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/status/{job_id}")
async def api_status(job_id: str):
    """API: Get job status."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "progress": jobs[job_id]["progress"],
        "error": jobs[job_id].get("error"),
    }


@app.get("/api/v1/results/{job_id}")
async def api_results(job_id: str):
    """API: Get analysis results."""
    result_path = RESULTS_DIR / job_id / "report.json"
    if not result_path.exists():
        return JSONResponse(status_code=404, content={"error": "Results not ready"})
    with open(result_path) as f:
        return json.load(f)


@app.get("/api/v1/jobs")
async def api_list_jobs():
    """API: List all jobs."""
    return {
        job_id: {
            "status": j["status"],
            "progress": j["progress"],
            "filename": j["filename"],
            "created_at": j["created_at"],
        }
        for job_id, j in jobs.items()
    }


# ─── Background Processing ───

def process_video_task(job_id: str, video_path: str):
    """Background task: run the analysis pipeline."""
    import sys, os
    # Add project root to path so src imports work
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 10

    try:
        from src.pipeline import HoeherrPipeline

        # Check if a trained model exists, otherwise use pretrained
        model_path = Path(project_root) / "models" / "detection" / "best.pt"
        if not model_path.exists():
            # Fallback: use base yolov8n for demo purposes
            model_path = "yolov8n.pt"

        # Use GPU if available
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        pipeline = HoeherrPipeline(
            model_path=str(model_path),
            tracker_config=str(Path(project_root) / "configs" / "botsort.yaml"),
            device=device,
        )

        output_dir = RESULTS_DIR / job_id
        jobs[job_id]["progress"] = 30

        report = pipeline.process_match(video_path, str(output_dir))
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        import traceback
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        traceback.print_exc()
