# Face Recognition System

A powerful face recognition system built with FastAPI, PyTorch, and FaceNet. This application can detect faces in images, extract facial embeddings, and find similar faces from a pre-built database.

## Features

- **Face Detection**: Uses MTCNN for robust face detection
- **Face Recognition**: Leverages FaceNet (InceptionResnetV1) for face embedding extraction
- **Similarity Matching**: Cosine similarity-based face matching
- **Web API**: FastAPI-based REST API for easy integration
- **Database Management**: Automatic face database building from image collections
- **GPU Support**: CUDA acceleration for faster processing

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Deep Learning**: PyTorch, FaceNet-PyTorch
- **Computer Vision**: OpenCV, PIL
- **Face Detection**: MTCNN
- **Similarity**: scikit-learn
- **Web Server**: Uvicorn

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd face_recognition
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Building the Face Database

First, you need to build a face database from your image collection:

```bash
python build_database.py
```

This script will:
- Process all images in the `temp/` folder
- Extract face embeddings using FaceNet
- Save the database to `face_database.pkl`

**Note**: Place your reference images in the `temp/` folder before running this command.

### 2. Running the Web Server

Start the FastAPI server:

```bash
python main.py
```

The server will start at `http://127.0.0.1:5000`

### 3. Using the API

#### Upload and Recognize Faces

Send a POST request to the root endpoint with an image file:

```bash
curl -X POST "http://127.0.0.1:5000/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

#### API Response

```json
{
  "success": true,
  "results": [
    {
      "filename": "person1.jpg",
      "path": "temp/person1.jpg",
      "similarity": 0.95,
      "similarity_percentage": 95.0
    }
  ]
}
```

## Project Structure

```
face_recognition/
├── main.py                 # FastAPI server
├── face_recognition.py     # Core face recognition logic
├── build_database.py       # Database builder script
├── requirements.txt        # Python dependencies
├── face_database.pkl       # Face embeddings database
├── samples/               # Sample images
├── temp/                  # Reference images (ignored by git)
└── README.md              # This file
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Web interface for file upload |
| POST   | `/`      | Upload image for face recognition |

## Configuration

### Face Recognition Parameters

You can modify the face detection parameters in `face_recognition.py`:

```python
self.mtcnn = MTCNN(
    image_size=160,           # Face image size
    margin=0,                 # Face margin
    min_face_size=20,         # Minimum face size
    thresholds=[0.6, 0.7, 0.7], # Detection thresholds
    factor=0.709,             # Scale factor
    post_process=True,        # Post-processing
    device=self.device        # GPU/CPU device
)
```

### Server Configuration

Modify server settings in `main.py`:

```python
uvicorn.run(
    "main:app",
    host="127.0.0.1",    # Server host
    port=5000,           # Server port
    reload=True          # Auto-reload on changes
)
```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for faster processing
2. **Image Quality**: Higher quality images generally yield better recognition results
3. **Database Size**: Larger databases may take longer to search but provide more accurate results
4. **Face Size**: Ensure faces in images are clearly visible and not too small

## Troubleshooting

### Common Issues

1. **No face detected**: Ensure the image contains a clear, well-lit face
2. **CUDA errors**: Install PyTorch with CUDA support or use CPU-only version
3. **Database not found**: Run `build_database.py` first to create the face database
4. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

### System Requirements

- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **Storage**: 2GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) for the face recognition model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Support

If you encounter any issues or have questions, please open an issue on GitHub or contact the maintainers.
