# Docker Quick Start Guide

## 🚨 First: Start Docker Desktop

**The error you're seeing means Docker Desktop is not running.**

### Step 1: Start Docker Desktop

1. **Open Docker Desktop** from Start Menu
   - Search for "Docker Desktop" in Windows Start Menu
   - Click to launch

2. **Wait for it to start** (30-60 seconds)
   - Look for the Docker whale icon in your system tray (bottom right)
   - Icon should be steady (not animated) when ready

3. **Verify it's running**

```powershell
docker version
```

You should see both **Client** and **Server** sections. If you only see Client, Docker Desktop is still starting.

---

## ✅ Once Docker is Running

### Step 1: Build Your Docker Image

```powershell
cd d:\Car_license\License-Plate-Detection-and-Recognition-Yolov8
docker build -t plate-api .
```

**What happens**:
- Downloads base Python image (~100MB)
- Installs system libraries
- Installs Python packages (PyTorch, FastAPI, EasyOCR, etc.)
- Copies your code and model

**Time**: 5-10 minutes first time, faster after (cached layers)

### Step 2: Verify Image Created

```powershell
docker images
```

You should see `plate-api` in the list.

### Step 3: Run Container

**Option A: Simple run**
```powershell
docker run --rm -p 8000:8000 plate-api
```

**Option B: Using Docker Compose (recommended)**
```powershell
docker compose up --build
```

### Step 4: Test Your Containerized API

Open browser: `http://localhost:8000/docs`

Or test with curl:
```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST "http://localhost:8000/predict?conf=0.25" -F "file=@demo_frame.jpg"
```

---

## 🐛 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Fix**: Start Docker Desktop and wait for it to fully start.

### Error: "Port 8000 already in use"

**Fix**: Stop the container using port 8000:
```powershell
docker ps                    # Find container name/ID
docker stop <container-id>   # Stop it
```

Or change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 on host
```

### Error: "Build failed"

**Check**:
- Is `requirements.txt` correct?
- Is `kbest.pt` in project root?
- Check build logs for specific errors

### Container starts but API doesn't work

**Check logs**:
```powershell
docker compose logs
```

Or if running without compose:
```powershell
docker logs <container-name>
```

---

## 📋 Quick Command Reference

```powershell
# Check Docker status
docker version

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

# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container-name>

# Remove container
docker rm <container-name>

# Remove image
docker rmi plate-api
```

---

## 🎯 What You'll Learn

After building and running:
- ✅ How Docker packages your entire app
- ✅ How containers isolate your app
- ✅ How to deploy anywhere Docker runs
- ✅ How Docker Compose simplifies management

---

**Next Step**: Start Docker Desktop, then run `docker build -t plate-api .` 🚀
