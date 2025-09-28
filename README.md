# Face Recognition System

A powerful AI-powered face recognition system built with FastAPI, PyTorch, MTCNN, and FaceNet. This application can detect faces in images, extract facial embeddings, and find similar faces from a pre-built database with high accuracy.

## 🚀 Features

* **Advanced Face Detection**: Uses MTCNN (Multi-task CNN) for robust face detection
* **Deep Learning Recognition**: Leverages FaceNet (InceptionResnetV1) for precise face embedding extraction
* **Smart Similarity Matching**: Cosine similarity-based face matching algorithm
* **Clean Web Interface**: Simple FastAPI-based interface with file upload functionality
* **Recursive Directory Search**: Automatically searches through nested folders using regex and glob patterns
* **Configurable Directory**: Easy directory configuration for different image sources
* **Top 15 Results**: Returns the 15 most similar faces with names and similarity percentages
* **GPU Support**: CUDA acceleration for faster processing

## 🛠 Technology Stack

* **Backend**: FastAPI, Python 3.8+
* **Deep Learning**: PyTorch, FaceNet-PyTorch
* **Computer Vision**: OpenCV, PIL
* **Face Detection**: MTCNN
* **Pattern Matching**: Regex and Glob for file search
* **Similarity**: scikit-learn cosine similarity
* **Web Server**: Uvicorn

## 📦 Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/DevJelvehgar/face-recognition.git
cd face-recognition
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

## 🗂 Directory Configuration

### Configuring Image Source Directory

You can easily change the source directory for your image database by modifying **line 19** in `build_database.py`:

```python
def __init__(self, images_folder=r"E:\images", database_file="face_database.pkl"):
```

**Examples of different directory configurations:**

```python
# Windows paths
images_folder=r"C:\Users\YourName\Pictures"
images_folder=r"D:\Photos\FaceDatabase"

# Linux/Mac paths  
images_folder="/home/username/images"
images_folder="/Users/username/Pictures"

# Relative paths
images_folder="./my_images"
images_folder="../photos"
```

### 🔍 Advanced File Search with Regex and Glob

The system now includes powerful file search capabilities:

#### **Regex Pattern Matching (Line 78)**
```python
image_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp|tiff|gif|webp)$', re.IGNORECASE)
```

**Supported image formats:**
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images  
- `.bmp` - Bitmap images
- `.tiff` - TIFF images
- `.gif` - GIF images
- `.webp` - WebP images

#### **Recursive Glob Search (Line 89)**
```python
all_files = glob.glob(os.path.join(self.images_folder, '**', '*'), recursive=True)
```

**Search behavior:**
- `**` enables recursive search through all subdirectories
- Finds images in nested folders of any depth
- Example structure it can handle:
```
E:\images\
├── photo1.jpg
├── family\
│   ├── wedding.png
│   └── vacation\
│       └── beach.jpeg
├── work\
│   ├── meeting.jpg
│   └── events\
│       ├── conference.png
│       └── team\
│           └── group_photo.jpg
```

## 🚀 Usage

### 1. Building the Face Database

First, configure your image directory and build the face database:

```bash
python build_database.py
```

**This script will:**
- Process all images in your configured folder using regex patterns
- Recursively search through subdirectories using glob
- Extract face embeddings using FaceNet
- Save the database to `face_database.pkl`

**Output example:**
```
Found 150 images in E:\images and its subdirectories
Processing: E:\images\person1.jpg
✓ Successfully processed: person1.jpg
Processing: E:\images\family\person2.png
✓ Successfully processed: person2.png
...
Database building completed!
Total images processed: 150
Successfully processed: 142
```

### 2. Running the Web Server

Start the FastAPI server:

```bash
python main.py
```

The server will start at `http://127.0.0.1:8000`

### 3. Using the Web Interface

1. Open your browser and go to `http://127.0.0.1:8000`
2. Click "Choose Files" to select an image
3. Click "Show Similar Images" to find matches
4. View the top 15 similar faces with names and similarity percentages

### 4. API Response Format

```json
{
  "success": true,
  "results": [
    {
      "filename": "person1.jpg",
      "path": "E:/images/family/person1.jpg",
      "similarity": 0.9532,
      "similarity_percentage": 95.32
    },
    {
      "filename": "person2.png", 
      "path": "E:/images/work/person2.png",
      "similarity": 0.8745,
      "similarity_percentage": 87.45
    }
  ]
}
```

## 📁 Project Structure

```
face-recognition/
├── main.py                 # FastAPI server (simplified, clean code)
├── face_recognition.py     # Core face recognition logic
├── build_database.py       # Database builder with regex/glob search
├── index.html             # Simple web interface (no CSS)
├── requirements.txt       # Python dependencies
├── face_database.pkl      # Face embeddings database
├── samples/              # Sample images
└── README.md             # This documentation
```

## ⚙️ Configuration

### Face Detection Parameters

Modify detection parameters in `face_recognition.py`:

```python
self.mtcnn = MTCNN(
    image_size=160,           # Face image size for FaceNet
    margin=0,                 # Face margin pixels
    min_face_size=20,         # Minimum detectable face size
    thresholds=[0.6, 0.7, 0.7], # [detection, refinement, landmarks]
    factor=0.709,             # Scale factor between sizes
    post_process=True,        # Enable post-processing
    device=self.device        # GPU/CPU device
)
```

### Directory Search Configuration

Customize the regex pattern in `build_database.py` for different file types:

```python
# Current pattern (line 78)
image_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp|tiff|gif|webp)$', re.IGNORECASE)

# Custom examples:
# Only JPEG files
image_pattern = re.compile(r'.*\.(jpg|jpeg)$', re.IGNORECASE)

# Include RAW formats
image_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp|tiff|gif|webp|raw|cr2|nef)$', re.IGNORECASE)
```

### Server Configuration

Modify server settings in `main.py`:

```python
uvicorn.run(app, host="127.0.0.1", port=8000)
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /        | Web interface for file upload |
| POST   | /recognize | Upload image for face recognition |

## 📊 Performance Tips

1. **GPU Acceleration**: Install PyTorch with CUDA for 3-5x faster processing
2. **Image Quality**: Use high-resolution, well-lit images for better accuracy
3. **Database Organization**: Organize images in logical subdirectories
4. **Face Size**: Ensure faces are at least 20x20 pixels
5. **File Formats**: PNG and JPEG typically give best results

## 🐛 Troubleshooting

### Common Issues

1. **"No face detected"**: Ensure images contain clear, well-lit faces
2. **"Directory not found"**: Check the path in line 19 of `build_database.py`
3. **CUDA errors**: Install PyTorch with CUDA support or use CPU version
4. **Import errors**: Run `pip install -r requirements.txt`
5. **No images found**: Verify regex pattern matches your file extensions

### System Requirements

* **Minimum RAM**: 4GB
* **Recommended RAM**: 8GB+
* **Storage**: 2GB free space
* **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## 🔄 Recent Updates

### Version 1.3 Features:
- ✅ **Regex Pattern Matching**: Advanced file filtering with customizable patterns
- ✅ **Glob Recursive Search**: Deep directory traversal for nested folders  
- ✅ **Configurable Directory**: Easy path configuration in line 19
- ✅ **Simplified Interface**: Clean UI with just file chooser and results
- ✅ **Top 15 Results**: Shows exactly 15 most similar faces
- ✅ **Clean Output**: Only filename and similarity percentage (no image display)
- ✅ **English Interface**: All text in English for better accessibility

## 📝 Usage Examples

### Example 1: Family Photo Organization
```
E:\family_photos\
├── 2023\
│   ├── wedding\
│   │   ├── bride.jpg
│   │   └── groom.png
│   └── vacation\
│       └── kids.jpeg
├── 2024\
│   └── reunion\
│       ├── grandma.jpg
│       └── cousins.png
```

### Example 2: Employee Database  
```
C:\company\employees\
├── management\
│   ├── ceo.jpg
│   └── directors\
│       ├── john_doe.png
│       └── jane_smith.jpg
├── engineering\
│   ├── developers\
│   │   ├── alice.jpg
│   │   └── bob.png
│   └── qa\
│       └── charlie.jpeg
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

* [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) for the face recognition model
* [FastAPI](https://fastapi.tiangolo.com/) for the web framework
* [PyTorch](https://pytorch.org/) for the deep learning framework
* [MTCNN](https://github.com/ipazc/mtcnn) for face detection

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on [GitHub](https://github.com/DevJelvehgar/face-recognition/issues)
- Check the troubleshooting section above
- Review the configuration examples

## 🌟 Star this Repository

If you find this project helpful, please consider giving it a star ⭐ on GitHub!

---

**Made with ❤️ by DevJelvehgar**
