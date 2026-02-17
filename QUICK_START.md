# Quick Start: Running FastAPI Locally (No Docker)

This guide shows you how to run and test your FastAPI application locally.

---

## Step 1: Install Dependencies

Open PowerShell in the project directory:

```powershell
cd d:\Car_license\License-Plate-Detection-and-Recognition-Yolov8
```

### Option A: Use your existing virtual environment

If you have `.venv` already:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Install globally (or create new venv)

```powershell
pip install -r requirements.txt
```

**Important**: EasyOCR will download models (~500MB) on first use:

- Models are automatically downloaded when you make your first OCR request
- First OCR call will take 2-3 minutes (downloading models)
- Models are cached in `~/.EasyOCR/model/` for future use
- No system binaries needed - everything installs via pip!

---

## Step 2: Verify Model File Exists

Make sure `kbest.pt` is in the project root:

```powershell
dir kbest.pt
```

If it's missing, the server will fail to start (by design - fail fast).

---

## Step 3: Start the FastAPI Server

```powershell
uvicorn app.main:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Keep this terminal open!** The server runs here.

---

## Step 4: Test the API

### Method 1: Use the Interactive Swagger UI (Easiest!)

Open your browser and go to:

```
http://localhost:8000/docs
```

This is **Swagger UI** - FastAPI automatically generates this!

**To test:**

1. Click on `POST /predict` → **Try it out**
2. Click **Choose File** and select an image
3. Adjust `conf` parameter if needed (default: 0.25)
4. Click **Execute**
5. See the JSON response with detected plates!

To test annotated image:
- Click on `POST /predict/annotated` → same process → download the annotated image

---

### Method 2: Use PowerShell curl

**Test health endpoint:**

```powershell
curl.exe http://localhost:8000/health
```

**Test prediction (JSON):**

```powershell
curl.exe -X POST "http://localhost:8000/predict?conf=0.25" -F "file=@your_image.jpg"
```

**Test prediction (annotated image):**

```powershell
curl.exe -X POST "http://localhost:8000/predict/annotated?conf=0.25" -F "file=@your_image.jpg" --output output.jpg
```

Replace `your_image.jpg` with an actual image path!

---

### Method 3: Use the Python Test Script

I've created `test_api.py` for you:

```powershell
# In a NEW terminal (keep server running in first terminal)
python test_api.py
```

Follow the prompts!

---

## Step 5: Check Predictions

### Expected JSON Response (`/predict`):

```json
{
  "count": 1,
  "detections": [
    {
      "bbox_xyxy": [100, 200, 300, 250],
      "confidence": 0.87,
      "text": "DL8CAF5030"
    }
  ],
  "texts": ["DL8CAF5030"]
}
```

### What each field means:

- **`count`**: Number of license plates detected
- **`detections`**: Array of detection objects
  - **`bbox_xyxy`**: Bounding box coordinates `[x1, y1, x2, y2]`
  - **`confidence`**: YOLO detection confidence (0.0 to 1.0)
  - **`text`**: OCR-extracted plate text
- **`texts`**: Simple array of all detected texts

---

## Troubleshooting

### Error: "Model weights not found"

- Make sure `kbest.pt` exists in project root
- Or set environment variable: `$env:MODEL_PATH="path/to/kbest.pt"`

### Error: "EasyOCR not found" or OCR fails

- Run: `pip install easyocr`
- Check OCR status: `curl.exe http://localhost:8000/health/ocr`
- First OCR call downloads models (~500MB) - be patient!
- If models are corrupted, clear cache: `Remove-Item -Path "$env:USERPROFILE\.EasyOCR\model\*" -Recurse -Force`

### Error: "ModuleNotFoundError: No module named 'fastapi'"

- Run: `pip install -r requirements.txt`

### Server won't start on port 8000

- Port might be in use: `uvicorn app.main:app --reload --port 8001`
- Then use `http://localhost:8001` instead

### Predictions return empty text

- First OCR call downloads models - wait 2-3 minutes
- Check `/health/ocr` endpoint to verify EasyOCR is working
- This is normal if OCR can't read the plate (blurry, angle, etc.)
- Try adjusting `conf` parameter (lower = more detections, but may include false positives)

---

## Next Steps

- Try different images
- Adjust confidence threshold (`conf` parameter)
- Check the annotated images to see bounding boxes
- Read `DOCUMENTATION.md` for deeper understanding

---

## Stop the Server

Press `Ctrl+C` in the terminal where `uvicorn` is running.
