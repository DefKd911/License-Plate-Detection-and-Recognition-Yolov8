# Docker Guide: Containerizing Your License Plate Detection API

This guide explains **Docker concepts** and how to containerize your FastAPI application.

---

## 🎯 What is Docker? (Simple Explanation)

Think of Docker like a **shipping container** for software:

- **Without Docker**: Your app works on your mchine, but might break on others (different Python versions, missing libraries, etc.)
- **With Docker**: Your app runs the same way everywhere - your laptop, server, cloud, anywhere!

### Key Concepts

1. **Docker Image**: A snapshot/blueprint containing:
   - Operating system (Linux)
   - Python runtime
   - Your code
   - All dependencies (pip packages)
   - Configuration

2. **Docker Container**: A running instance of an image
   - Like running a program from an executable
   - Isolated from your host system
   - Can start/stop/delete easily

3. **Dockerfile**: Instructions to build an image
   - Step-by-step recipe
   - "Start with Python 3.10, install these packages, copy my code, run this command"

---

## 📋 Understanding Your Dockerfile (Line by Line)

Let's break down your `Dockerfile`:

```dockerfile
FROM python:3.10-slim
```
**What it does**: Start with a base image (Python 3.10 on Debian Linux)
- `slim` = smaller image size (no unnecessary tools)

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
```
**What it does**: Set environment variables
- `PYTHONDONTWRITEBYTECODE=1`: Don't create `.pyc` files (saves space)
- `PYTHONUNBUFFERED=1`: Print output immediately (better for logs)

```dockerfile
WORKDIR /app
```
**What it does**: Set working directory inside container
- All commands run from `/app` directory

```dockerfile
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgl1 \
     libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
```
**What it does**: Install system libraries needed by OpenCV
- `libgl1`, `libglib2.0-0`: Graphics libraries OpenCV needs
- `--no-install-recommends`: Don't install optional packages (smaller image)
- `rm -rf /var/lib/apt/lists/*`: Clean up package cache (smaller image)

```dockerfile
COPY requirements.txt /app/requirements.txt
```
**What it does**: Copy your requirements file into the image

```dockerfile
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r /app/requirements.txt
```
**What it does**: Install Python dependencies
- Install PyTorch (CPU version) first (Ultralytics needs it)
- Then install everything from `requirements.txt`
- `--no-cache-dir`: Don't save pip cache (smaller image)

```dockerfile
COPY app /app/app
COPY kbest.pt /app/kbest.pt
```
**What it does**: Copy your code and model weights into the image

```dockerfile
EXPOSE 8000
```
**What it does**: Document that the app uses port 8000
- Doesn't actually open the port - just documentation

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
**What it does**: Command to run when container starts
- `0.0.0.0` = listen on all network interfaces (not just localhost)

---

## 🚀 Building Your Docker Image

### Step 1: Ensure Docker Desktop is Running

On Windows, make sure **Docker Desktop** is started.

Check:
```powershell
docker version
```

You should see both **Client** and **Server** sections.

### Step 2: Build the Image

```powershell
cd d:\Car_license\License-Plate-Detection-and-Recognition-Yolov8
docker build -t plate-api .
```

**What happens**:
- `docker build`: Build command
- `-t plate-api`: Tag/name the image as "plate-api"
- `.`: Build context (current directory)

**This will take 5-10 minutes** the first time (downloading base image, installing packages).

### Step 3: Verify Image was Created

```powershell
docker images
```

You should see `plate-api` in the list.

---

## 🏃 Running Your Container

### Basic Run

```powershell
docker run --rm -p 8000:8000 plate-api
```

**What each part means**:
- `docker run`: Start a container
- `--rm`: Automatically delete container when it stops
- `-p 8000:8000`: Map port 8000 (host) → 8000 (container)
- `plate-api`: Image name

**Test it**: Open `http://localhost:8000/docs` in your browser!

### Run in Background (Detached Mode)

```powershell
docker run -d --name plate-api-container -p 8000:8000 plate-api
```

- `-d`: Run in background (detached)
- `--name`: Give container a name

**Check if running**:
```powershell
docker ps
```

**Stop it**:
```powershell
docker stop plate-api-container
```

**Remove it**:
```powershell
docker rm plate-api-container
```

---

## 🐳 Using Docker Compose (Easier!)

Docker Compose lets you define everything in a file (`docker-compose.yml`).

### Your docker-compose.yml Explained

```yaml
services:
  plate-api:
    build: .                    # Build image from current directory
    ports:
      - "8000:8000"            # Host port : Container port
    environment:
      YOLO_DEVICE: "cpu"       # Set environment variables
      YOLO_CONF: "0.25"
      MODEL_PATH: "kbest.pt"
```

### Run with Compose

```powershell
docker compose up --build
```

**What it does**:
- Builds image if needed (`--build`)
- Starts container
- Shows logs in terminal

**Stop**:
```powershell
docker compose down
```

**Run in background**:
```powershell
docker compose up -d --build
```

---

## 🔍 Useful Docker Commands

### List Images
```powershell
docker images
```

### List Running Containers
```powershell
docker ps
```

### List All Containers (including stopped)
```powershell
docker ps -a
```

### View Container Logs
```powershell
docker logs plate-api-container
```

### Execute Command Inside Container
```powershell
docker exec -it plate-api-container bash
```
Opens a bash shell inside the running container.

### Remove Image
```powershell
docker rmi plate-api
```

### Remove All Stopped Containers
```powershell
docker container prune
```

### Remove Unused Images
```powershell
docker image prune
```

---

## 🎓 Docker Concepts Summary

| Concept | Real-World Analogy | What It Is |
|---------|-------------------|------------|
| **Image** | Blueprint/Recipe | Snapshot with OS + code + dependencies |
| **Container** | Running Instance | Active process running from the image |
| **Dockerfile** | Recipe Instructions | Step-by-step build instructions |
| **docker-compose.yml** | Configuration File | Defines services, ports, env vars |

### Image vs Container

- **Image**: Template (like a class in programming)
- **Container**: Instance (like an object created from a class)

You can run **multiple containers** from the **same image**!

---

## 🧪 Testing Your Dockerized API

### 1. Build and Run

```powershell
docker compose up --build
```

### 2. Test Health Endpoint

In a new terminal:
```powershell
curl.exe http://localhost:8000/health
```

### 3. Test Prediction

```powershell
curl.exe -X POST "http://localhost:8000/predict?conf=0.25" -F "file=@demo_frame.jpg"
```

### 4. Check Logs

```powershell
docker compose logs
```

---

## 🐛 Common Docker Issues

### Issue 1: "Cannot connect to Docker daemon"

**Fix**: Start Docker Desktop

### Issue 2: Port Already in Use

**Error**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Fix**: 
- Stop the container using port 8000: `docker stop <container-name>`
- Or change port in `docker-compose.yml`: `"8001:8000"`

### Issue 3: Image Build Fails

**Check**:
- Is `requirements.txt` correct?
- Is `kbest.pt` in the project root?
- Check build logs: `docker build -t plate-api .` (look for errors)

### Issue 4: Container Starts but API Doesn't Work

**Check logs**:
```powershell
docker logs <container-name>
```

Common causes:
- Model file missing
- Port not mapped correctly
- Code errors

---

## 🚀 Next Steps

1. **Test locally**: Build and run your container
2. **Push to registry**: Upload to Docker Hub (optional)
3. **Deploy**: Run on cloud (AWS, Azure, GCP) or your server

---

## 💡 Why Docker is Powerful

1. **Consistency**: Works the same everywhere
2. **Isolation**: Doesn't mess with your host system
3. **Portability**: Run on any machine with Docker installed
4. **Scalability**: Easy to run multiple instances
5. **Version Control**: Tag images with versions

---

## 📚 Quick Reference

```powershell
# Build image
docker build -t plate-api .

# Run container
docker run --rm -p 8000:8000 plate-api

# Run with compose
docker compose up --build

# Stop compose
docker compose down

# View logs
docker compose logs -f

# Check status
docker ps
```

---

Ready to build and test? Let's do it! 🎉
