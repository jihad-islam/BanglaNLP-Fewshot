# Bangla NLP Few-Shot Learning System

A web-based platform for Bangla text analysis using transformer models and meta-learning approaches.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)

---

## Features

### Supported Tasks

- **Sentiment Analysis** - Positive, Neutral, Negative
- **Hate Speech Detection** - Hate, Non-Hate
- **Topic Classification** - Politics, Sports, Entertainment, Economy

### Model Architecture

- **BanglaBERT (Baseline)** - Fine-tuned transformer model
- **ProtoNet (Meta-Learning)** - 10-shot learning with prototypical networks

### Interface

- Interactive web UI with real-time predictions
- RESTful API with automatic documentation
- Side-by-side model comparison

---

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- 4GB RAM minimum (8GB recommended)
- 5GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bangla-nlp-few-shot.git
cd bangla-nlp-few-shot

# Download models from Hugging Face (required)
pip install huggingface_hub
huggingface-cli download Jihad07/bangla-nlp-models --repo-type=model --local-dir sources/models

# Install dependencies
./setup-only.sh
```

**Note:** Models are hosted on Hugging Face due to size constraints. Visit: https://huggingface.co/Jihad07/bangla-nlp-models

### Running the Application

```bash
# Terminal 1 - Start backend
./start-backend.sh

# Terminal 2 - Start frontend
./start-frontend.sh
```

### Stopping the Application

```bash
./stop-backend.sh
./stop-frontend.sh
```

Or press `Ctrl+C` in each terminal.

---

## Access URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Project Structure

```
bangla-nlp-few-shot/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── model_loader.py  # Model loading logic
│   └── requirements.txt # Python dependencies
├── frontend/            # React frontend
│   ├── src/
│   │   ├── App.js      # Main component
│   │   └── index.js    # Entry point
│   └── package.json    # Node dependencies
├── sources/
│   └── models/         # Pre-trained models
│       ├── BanglaBert/ # Baseline models
│       └── MetaLearning/ # ProtoNet models
├── setup-only.sh       # Install dependencies
├── start-backend.sh    # Start backend
├── start-frontend.sh   # Start frontend
├── stop-backend.sh     # Stop backend
├── stop-frontend.sh    # Stop frontend
├── README.md           # This file
└── SETUP.md            # Detailed setup guide
```

---

## API Usage

### Sentiment Analysis

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

### Model Comparison

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task": "hate",
    "text": "আমরা একসাথে কাজ করি",
    "mode": "comparison"
  }'
```

---

## Technology Stack

### Backend

- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- Transformers (HuggingFace) - Pre-trained models
- Uvicorn - ASGI server

### Frontend

- React 18 - UI framework
- Axios - HTTP client
- Tailwind CSS - Styling

### Models

- BanglaBERT - csebuetnlp/banglabert
- ProtoNet - Prototypical Networks for few-shot learning

---

## Development

### Backend Only

```bash
cd backend
source venv/bin/activate
python main.py
```

### Frontend Only

```bash
cd frontend
npm start
```

---

## Troubleshooting

### Port Already in Use

```bash
./stop-backend.sh   # For port 8000
./stop-frontend.sh  # For port 3000
```

### Module Not Found

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Won't Compile

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

For detailed troubleshooting, see [SETUP.md](SETUP.md).

---

## License

MIT License

---

## Acknowledgments

- BanglaBERT team at BUET for the pre-trained model
- HuggingFace for the Transformers library
- FastAPI and React communities
