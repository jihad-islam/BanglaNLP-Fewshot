# Setup Guide

Complete installation instructions for Arch Linux and Windows.

---

## System Requirements

- **CPU:** 2+ cores
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 5GB free space
- **OS:** Arch Linux or Windows 10/11

---

## Table of Contents

1. [Arch Linux Setup](#arch-linux-setup)
2. [Windows Setup](#windows-setup)
3. [Model Download](#model-download)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)

---

## Arch Linux Setup

### Step 1: Install Prerequisites

```bash
# Update system
sudo pacman -Syu

# Install Python, Node.js, and Git
sudo pacman -S python python-pip nodejs npm git
```

### Step 2: Clone Repository

```bash
git clone https://github.com/jihad-islam/BanglaNLP-Fewshot.git
cd BanglaNLP-Fewshot
```

### Step 3: Download Models

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download models (1.5GB)
huggingface-cli download Jihad07/bangla-nlp-models --repo-type=model --local-dir sources/models
```

**This will take 5-10 minutes depending on your internet speed.**

### Step 4: Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (this takes 5-10 minutes)
pip install -r requirements.txt
```

### Step 5: Setup Frontend

```bash
# Open a new terminal
cd frontend

# Install dependencies (this takes 3-5 minutes)
npm install
```

### Step 6: Verify Installation

```bash
# Check backend files
ls backend/
# Should see: main.py, model_loader.py, requirements.txt, venv/

# Check frontend files
ls frontend/
# Should see: src/, public/, package.json, node_modules/

# Check models
ls sources/models/BanglaBert/
ls sources/models/MetaLearning/
```

---

## Windows Setup

### Step 1: Install Prerequisites

#### Install Python

1. Go to https://www.python.org/downloads/
2. Download Python 3.8 or higher
3. Run installer
4. **IMPORTANT:** Check "Add Python to PATH"
5. Click "Install Now"
6. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Install Node.js

1. Go to https://nodejs.org/
2. Download LTS version
3. Run installer with default settings
4. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

#### Install Git (Optional)

1. Go to https://git-scm.com/download/win
2. Download and run installer
3. Use default settings

### Step 2: Download Project

**Option A: Using Git**

```cmd
git clone https://github.com/jihad-islam/BanglaNLP-Fewshot.git
cd BanglaNLP-Fewshot
```

**Option B: Download ZIP**

1. Go to https://github.com/jihad-islam/BanglaNLP-Fewshot
2. Click "Code" → "Download ZIP"
3. Extract to `C:\Users\YourName\Documents\BanglaNLP-Fewshot`
4. Open Command Prompt:
   ```cmd
   cd C:\Users\YourName\Documents\BanglaNLP-Fewshot
   ```

### Step 3: Download Models

```cmd
# Install Hugging Face CLI
pip install huggingface_hub

# Download models (1.5GB)
huggingface-cli download Jihad07/bangla-nlp-models --repo-type=model --local-dir sources/models
```

**This will take 5-10 minutes.**

### Step 4: Setup Backend

Open **Command Prompt** or **PowerShell**:

```cmd
cd backend

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt
```

**Note:** PyTorch installation may take longer on Windows.

### Step 5: Setup Frontend

Open **new Command Prompt**:

```cmd
cd frontend

:: Install dependencies (takes 3-5 minutes)
npm install
```

### Step 6: Verify Installation

```cmd
:: Check backend
dir backend
:: Should see: main.py, model_loader.py, requirements.txt, venv\

:: Check frontend
dir frontend
:: Should see: src\, public\, package.json, node_modules\

:: Check models
dir sources\models\BanglaBert
dir sources\models\MetaLearning
```

---

## Model Download

### What Gets Downloaded

```
sources/models/
├── BanglaBert/
│   ├── BanglaBERT Hate Speech Detection/
│   ├── BanglaBERT Sentiment Analysis/
│   └── BanglaBERT Topic Classification/
└── MetaLearning/
    ├── Meta Learning Hate Speech Detection/
    ├── Meta Learning Sentiment Analysis/
    └── Meta Learning Topic Classification/
```

**Total size:** ~1.5GB

### Manual Download (Alternative)

If CLI download fails:

1. Go to https://huggingface.co/Jihad07/bangla-nlp-models
2. Click "Files and versions"
3. Download all folders
4. Extract to `sources/models/`

---

## Running the Application

### Arch Linux

**Terminal 1 - Backend:**

```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm start
```

### Windows

**Command Prompt 1 - Backend:**

```cmd
cd backend
venv\Scripts\activate
python main.py
```

**Command Prompt 2 - Frontend:**

```cmd
cd frontend
npm start
```

### Access URLs

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Expected Output

**Backend:**

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Initializing models...
INFO:     ✓ Loaded Baseline model for hate
INFO:     ✓ Loaded Baseline model for sentiment
INFO:     ✓ Loaded Baseline model for topic
INFO:     ✓ Loaded ProtoNet model for hate
INFO:     ✓ Loaded ProtoNet model for sentiment
INFO:     ✓ Loaded ProtoNet model for topic
INFO:     Total models loaded: 6
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Frontend:**

```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.x:3000
```

---

## Troubleshooting

### Arch Linux Issues

**Port Already in Use:**

```bash
# Find process using port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill process
kill -9 <PID>
```

**Module Not Found:**

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

**Permission Denied:**

```bash
chmod +x backend/main.py
```

### Windows Issues

**Python not recognized:**

1. Reinstall Python
2. Check "Add to PATH" during installation
3. Restart Command Prompt
4. Verify: `python --version`

**Port Already in Use:**

```cmd
:: Find process
netstat -ano | findstr :8000

:: Kill process (replace <PID>)
taskkill /PID <PID> /F
```

**Module Not Found:**

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

**PowerShell Execution Policy:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Common Issues (Both OS)

**Models Not Loading:**

```bash
# Verify models exist
ls sources/models/BanglaBert/  # Arch
dir sources\models\BanglaBert  # Windows

# Re-download if missing
huggingface-cli download Jihad07/bangla-nlp-models --repo-type=model --local-dir sources/models
```

**Slow First Prediction:**

- Normal behavior
- First prediction: 10-20 seconds (models loading into memory)
- Subsequent predictions: 2-5 seconds

**Out of Memory:**

- Close other applications
- Minimum 4GB RAM required
- 8GB recommended

**Frontend Won't Compile:**

```bash
cd frontend
rm -rf node_modules package-lock.json  # Arch
npm install

# Windows
cd frontend
rmdir /s /q node_modules
del package-lock.json
npm install
```

---

## Stopping the Application

### Arch Linux

Press `Ctrl+C` in each terminal

### Windows

Press `Ctrl+C` in each Command Prompt window

---

## Installation Time

- **Prerequisites:** 2-5 minutes
- **Model Download:** 5-10 minutes (depends on internet)
- **Backend Setup:** 5-10 minutes
- **Frontend Setup:** 3-5 minutes
- **Total:** 15-30 minutes

---

## Installation Size

- Backend dependencies: ~2.5GB
- Frontend dependencies: ~300MB
- Models: ~1.5GB
- **Total:** ~5GB

---

## Next Steps

After successful installation:

1. Test the health endpoint:

   ```bash
   curl http://localhost:8000/health
   ```

2. Open the web UI:

   ```
   http://localhost:3000
   ```

3. Try sentiment analysis with Bangla text

4. Explore API documentation:
   ```
   http://localhost:8000/docs
   ```

---

## Support

For issues:

- Check this guide
- Review error messages
- Ensure all prerequisites installed
- Verify Python 3.8+ and Node.js 16+

---

**Last Updated:** December 9, 2025  
**Tested On:** Arch Linux, Windows 10/11
