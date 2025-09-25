"""
FastAPI Face Recognition Server
Main server file for face recognition application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
from PIL import Image
import io
import os
from face_recognition import FaceRecognition

# Initialize FastAPI app
app = FastAPI(title="Face Recognition API", version="1.0.0")

# Initialize face recognition
face_recognition = FaceRecognition()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page with file upload form"""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Run server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,
        reload=True
    )
