# Setup Guide

Complete installation and setup instructions for the Bangla NLP Few-Shot Learning System.

---

## System Requirements

### Minimum

- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended)
- **Storage**: 5GB free space
- **OS**: Linux, Windows, macOS

### Software

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **npm**: 7 or higher
- **Git**: For cloning repository

---

## Installation

### 1. Install Prerequisites

#### Linux (Arch/Manjaro)

```bash
sudo pacman -S python python-pip nodejs npm git
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv nodejs npm git
```

#### Linux (Fedora)

```bash
sudo dnf install python3 python3-pip nodejs npm git
```

#### Windows

1. Download Python from https://www.python.org/downloads/
2. Download Node.js from https://nodejs.org/
3. Install Git from https://git-scm.com/download/win
4. **Important**: Check "Add to PATH" during installation

#### macOS

```bash
# Install Homebrew first (https://brew.sh/)
brew install python node git
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/bangla-nlp-few-shot.git
cd bangla-nlp-few-shot
```

### 3. Run Setup

#### Linux/macOS

```bash
./setup-only.sh
```

#### Windows

```cmd
# Open Command Prompt or PowerShell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# New terminal
cd frontend
npm install
```

---

## Running the Application

### Linux/macOS

#### Start Backend (Terminal 1)

```bash
./start-backend.sh
```

#### Start Frontend (Terminal 2)

```bash
./start-frontend.sh
```

### Windows

#### Start Backend (Terminal 1)

```cmd
cd backend
venv\Scripts\activate
python main.py
```

#### Start Frontend (Terminal 2)

```cmd
cd frontend
npm start
```

---

## Stopping the Application

### Linux/macOS

```bash
./stop-backend.sh
./stop-frontend.sh
```

Or press `Ctrl+C` in each terminal.

### Windows

Press `Ctrl+C` in each Command Prompt window.

---

## Verification

### Check Backend Health

**Linux/macOS:**

```bash
curl http://localhost:8000/health
```

**Windows (PowerShell):**

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

**Expected Response:**

```json
{
  "status": "healthy",
  "models_loaded": 6,
  "available_tasks": ["hate", "sentiment", "topic"]
}
```

### Check Frontend

Open http://localhost:3000 in your browser.

### Test Prediction

**Linux/macOS:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task": "sentiment",
    "text": "এটি একটি দুর্দান্ত সিনেমা",
    "mode": "single",
    "model_type": "baseline"
  }'
```

**Windows (PowerShell):**

```powershell
$body = @{
    task = "sentiment"
    text = "এটি একটি দুর্দান্ত সিনেমা"
    mode = "single"
    model_type = "baseline"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```

---

## Troubleshooting

### Port Already in Use

**Linux/macOS:**

```bash
# Check what's using the port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill process
./stop-backend.sh
./stop-frontend.sh
```

**Windows:**

```cmd
# Find process
netstat -ano | findstr :8000

# Kill process (replace PID with actual number)
taskkill /PID <PID> /F
```

### Virtual Environment Issues

**Linux/macOS:**

```bash
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**

```cmd
cd backend
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Module Not Found

Ensure virtual environment is activated:

**Linux/macOS:**

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Won't Compile

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Models Not Loading

Verify models directory exists:

```bash
ls sources/models/BanglaBert/
ls sources/models/MetaLearning/
```

Ensure you have:

- `BanglaBERT Hate Speech Detection/`
- `BanglaBERT Sentiment Analysis/`
- `BanglaBERT Topic Classification/`
- `Meta Learning Hate Speech Detection/`
- `Meta Learning Sentiment Analysis/`
- `Meta Learning Topic Classification/`

### Slow First Prediction

First prediction takes 10-20 seconds (models loading into memory). Subsequent predictions are faster (2-5 seconds).

---

## Installation Size

- Backend dependencies: ~2.5GB (PyTorch is large)
- Frontend dependencies: ~300MB
- Model files: ~1.5GB
- **Total**: ~5GB

## Installation Time

- Fast internet (50+ Mbps): 5-10 minutes
- Moderate internet (10-50 Mbps): 10-15 minutes
- Slow internet (<10 Mbps): 15-30 minutes

---

## Script Reference

| Script              | Purpose                           |
| ------------------- | --------------------------------- |
| `setup-only.sh`     | Install all dependencies          |
| `start-backend.sh`  | Start backend server (port 8000)  |
| `start-frontend.sh` | Start frontend server (port 3000) |
| `stop-backend.sh`   | Stop backend server               |
| `stop-frontend.sh`  | Stop frontend server              |

---

## Support

For issues and questions:

- Check this setup guide
- Review error messages in terminal
- Ensure all prerequisites are installed
- Verify Python and Node.js versions

---

**Last Updated**: December 9, 2025  
**Tested On**: Arch Linux, Ubuntu 22.04, Windows 10/11, macOS
