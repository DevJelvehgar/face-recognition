from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image
import io
from face_recognition import FaceRecognition

app = FastAPI()

# Initialize face recognition
face_recognition = FaceRecognition()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/recognize")
async def recognize_face_endpoint(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform face recognition
        result = face_recognition.recognize_face(image)
        
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)