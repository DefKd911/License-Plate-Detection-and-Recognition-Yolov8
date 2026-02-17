# 🚗 Car License Plate Recognition using YOLOv8 + Streamlit

This project implements **real-time car license plate detection and recognition** using the **YOLOv8 object detection model** with an interactive **Streamlit interface**.

Users can upload images or videos, and the app detects license plates, extracts their text using OCR, and displays results in real time.

![License Plate Detection Demo](https://img.shields.io/badge/YOLOv8-License%20Plate%20Detection-blue?style=for-the-badge&logo=python)

---

## 📌 Features

- 🔍 **YOLOv8-powered license plate detection** with high accuracy
- 📝 **OCR integration** (EasyOCR) for text extraction - better accuracy for license plates
- 🎥 Upload **images or video files** for processing
- 🌐 Simple and interactive **Streamlit web app**
- ⚡ **Real-time processing** with optimized inference
- 📊 **Confidence scoring** for detections
- 🎯 **Bounding box visualization** with extracted text overlay

### 📚 Detailed documentation (FastAPI + Docker)

See `DOCUMENTATION.md`.

### 🚀 Applications
- Traffic monitoring and law enforcement
- Automated toll collection systems
- Smart parking management
- Vehicle tracking and access control
- Security and surveillance systems

---

## 🧠 What is YOLOv8?

**YOLOv8 (You Only Look Once version 8)** is the latest iteration of the popular YOLO object detection family, developed by Ultralytics. It's a state-of-the-art, real-time object detection model that offers significant improvements over previous versions.

### 🔑 Key Features of YOLOv8:

1. **Architecture Improvements**:
   - Enhanced backbone network with better feature extraction
   - Improved neck design for better feature fusion
   - Anchor-free detection head for faster inference

2. **Performance Benefits**:
   - Higher accuracy with lower computational cost
   - Faster training convergence
   - Better small object detection (perfect for license plates)
   - Improved robustness in various lighting conditions

3. **Model Variants**:
   - **YOLOv8n** (Nano): Ultra-fast, lightweight
   - **YOLOv8s** (Small): Balanced speed and accuracy
   - **YOLOv8m** (Medium): Higher accuracy
   - **YOLOv8l** (Large): Maximum accuracy
   - **YOLOv8x** (Extra Large): Best performance

### 🎯 Why YOLOv8 for License Plate Detection?

- **Single-stage detection**: Processes entire image in one pass
- **Real-time performance**: Suitable for video processing
- **High precision**: Excellent at detecting small, rectangular objects
- **Transfer learning**: Easy to fine-tune on custom datasets
- **Flexible input sizes**: Adapts to various image resolutions

---

## 🏗️ Project Structure

```
cars-license-plate-yolov8/
├── .venv/                              # Virtual environment
├── output/                             # Processed results and outputs
├── temp/                               # Temporary files during processing
├── models/                             # Model weights and configurations
│   └── kbest.pt                       # Trained YOLOv8 model weights
├── data/                              # Dataset and sample files
│   └── demo.mp4                       # Sample video for testing
├── src/                               # Source code
│   ├── yoloapplication.py             # Main Streamlit application
│   ├── detection.py                   # Detection utilities
│   └── ocr_utils.py                   # OCR processing functions
├── notebooks/                         # Jupyter notebooks
│   └── cars-license-plate-yolov8.ipynb # Training & experiments
├── requirements.txt                    # Project dependencies
├── config.yaml                        # Configuration settings
├── README.md                          # Project documentation
└── LICENSE                           # Project license
```

---

## ⚙️ Installation & Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/cars-license-plate-yolov8.git
cd cars-license-plate-yolov8
```

### 2. **Create Virtual Environment** (Recommended)
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Download Pre-trained Model** (if not included)
```bash
# The kbest.pt file should be in the models/ directory
# If missing, you can train your own or download from releases
```

---

## 🚀 Usage

### ▶️ **Run the Streamlit Web Application**
```bash
streamlit run yoloapplication.py
```

1. Open your browser and navigate to `http://localhost:8501`
2. Upload an image or video file using the file uploader
3. Adjust detection confidence threshold if needed
4. Click "Process" to detect license plates
5. View results with bounding boxes and extracted text

---

## 🔌 FastAPI (REST API)

This repo now also includes a **FastAPI backend** so you can integrate plate detection into any app (web/mobile/IoT).

### Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Health check: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### Example request (Windows PowerShell)

```powershell
curl.exe -X POST "http://localhost:8000/predict?conf=0.25" -F "file=@your_image.jpg"
```

### Get annotated image (with boxes/text)

```powershell
curl.exe -X POST "http://localhost:8000/predict/annotated?conf=0.25" -F "file=@your_image.jpg" --output out.jpg
```

---

## 🐳 Docker (deploy anywhere)

### Build

```bash
docker build -t plate-api .
```

### Run

```bash
docker run --rm -p 8000:8000 plate-api
```

Then open `http://localhost:8000/docs`.

### Using your own weights (optional)

If you want to keep weights outside the image:

```bash
docker run --rm -p 8000:8000 ^
  -e MODEL_PATH=/models/kbest.pt ^
  -v "%cd%\\models:/models" ^
  plate-api
```

### ▶️ **Training Custom Model** (Advanced)
```bash
# Open and execute the training notebook
jupyter notebook notebooks/cars-license-plate-yolov8.ipynb
```



## 📊 Model Performance

### Training Details:
- **Dataset**:
      len(train) = 345
      len(val) = 44
      len(test) = 44
- **Architecture**: YOLOv8n (nano variant)
- **Training Epochs**: 100
- **Image Size**: 320 X 320
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Learning Rate**: 0.01

  
### Performance Metrics:
- **mAP@0.5**: 91.7%
- **mAP@0.5:0.95**: 53.9%
- **Precision**: 86.3%
- **Recall**: 86.2 %
- Speed: 0.8ms preprocess, 46.4ms inference, 0.7ms postprocess per image at shape (1, 3, 224, 320)
- **Inference time**: 50 ms
- **Inference Speed**: ~  1000/45 ~ 23 FPS 

---

## 📈 Example Results

### Input Image:
![image](https://github.com/user-attachments/assets/b05455f2-4424-43c5-92fa-5742d7448710)


### Detection Output:
- **Detected Plates**: 1
- **Confidence Scores**: [0.87]
- **Extracted Text**: 
  - Plate : "DL8CAF5030"
  
## 🛠️ Technical Details

### YOLOv8 Architecture Components:

1. **Backbone**: Enhanced CSPDarknet with improved feature extraction
2. **Neck**: Path Aggregation Network (PANet) for better feature fusion
3. **Head**: Anchor-free detection head with separate classification and localization branches

### Detection Pipeline:
1. **Image Preprocessing**: Resize, normalize, and format for YOLOv8
2. **Inference**: Forward pass through YOLOv8 model
3. **Post-processing**: Non-Maximum Suppression (NMS) to filter detections
4. **OCR Processing**: Extract text from detected license plate regions
5. **Results Formatting**: Combine detection and OCR results

### OCR Integration:
- **EasyOCR**: Deep learning-based OCR with better accuracy for license plates, handles preprocessing automatically


## 🎯 Future Enhancements

- [ ] **Multi-language Support**: Extend OCR for international license plates
- [ ] **Real-time Webcam Processing**: Live camera feed integration
- [ ] **Database Integration**: Store and manage detection results
- [ ] **API Development**: REST API for integration with other systems
- [ ] **Mobile Deployment**: Convert to mobile-friendly format
- [ ] **Edge Deployment**: Optimize for Raspberry Pi and Jetson Nano
- [ ] **Advanced OCR**: Custom OCR model trained specifically for license plates
- [ ] **Vehicle Make/Model Recognition**: Extend to identify vehicle details
- [ ] **Night Vision Enhancement**: Improve detection in low-light conditions
- [ ] **Batch Processing**: Handle multiple files simultaneously

---

## 📝 Requirements

```txt
# Core dependencies
ultralytics>=8.0.196
streamlit>=1.28.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# OCR dependencies
easyocr>=1.7.0

# Additional utilities
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

# Optional: GPU support
torch>=2.0.0
torchvision>=0.15.0
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙌 Acknowledgements

- **[Ultralytics](https://ultralytics.com/)** - For the amazing YOLOv8 framework
- **[Streamlit](https://streamlit.io/)** - For the interactive web app frameworks
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** - For deep learning-based OCR support
- **[OpenCV](https://opencv.org/)** - For computer vision utilities
- **Community Contributors** - For dataset annotations and testing

---
