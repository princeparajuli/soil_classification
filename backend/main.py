from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io
import os
import sys

# Add backend folder to sys.path for imports
sys.path.append(os.path.dirname(__file__))

import soil_engine_core  # now it will always find the module

app = FastAPI(title="AI Soil Classification API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend folder
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.exists(frontend_dir):
    raise RuntimeError(f"Frontend folder does not exist: {frontend_dir}")

app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
def home():
    return {"status": "API running"}

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = soil_engine_core.classify_soil(img)

        # Standardize output for HTML JS
        if result.get("error"):
            return JSONResponse({
                "error": True,
                "message": result.get("message", "Unknown error")
            })
        else:
            return JSONResponse({
                "error": False,
                "soil": result.get("soil", "unknown"),
                "confidence": result.get("confidence", 0)
            })
    except Exception as e:
        return JSONResponse({
            "error": True,
            "message": f"Failed to process image: {str(e)}"
        })
