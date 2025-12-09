# Bangla NLP Few-Shot Learning System

Web-based platform for Bangla text analysis using transformer models and meta-learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)

---

## Features

- **Sentiment Analysis** - Classify text as Positive, Neutral, or Negative
- **Hate Speech Detection** - Identify hateful or offensive content
- **Topic Classification** - Categorize text into Politics, Sports, Entertainment, or Economy
- **Dual Models** - Compare BanglaBERT (baseline) vs ProtoNet (meta-learning)
- **Interactive UI** - Real-time predictions with confidence scores
- **REST API** - Complete API documentation with Swagger UI

---

## Technology Stack

**Backend**

- FastAPI - Modern Python web framework
- PyTorch - Deep learning
- Transformers - HuggingFace pre-trained models
- Uvicorn - ASGI server

**Frontend**

- React 18 - UI framework
- Axios - HTTP client
- Tailwind CSS - Styling

**Models**

- BanglaBERT - Fine-tuned transformer (csebuetnlp/banglabert)
- ProtoNet - 10-shot prototypical networks

---

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- 4GB RAM (8GB recommended)
- 5GB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/jihad-islam/BanglaNLP-Fewshot.git
cd BanglaNLP-Fewshot

# Download models from Hugging Face
pip install huggingface_hub
huggingface-cli download Jihad07/bangla-nlp-models --repo-type=model --local-dir sources/models

# Install backend dependencies
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

## Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

Backend runs at: http://localhost:8000

### Start Frontend (Terminal 2)

```bash
cd frontend
npm start
```

Frontend opens at: http://localhost:3000

---

## Usage

### Web Interface

1. Open http://localhost:3000
2. Select task: Sentiment, Hate Speech, or Topic
3. Choose model: Baseline (BanglaBERT) or Meta-Learning (ProtoNet)
4. Enter Bangla text
5. Click "Analyze Text"
6. View predictions with confidence scores

### API

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Sentiment Analysis:**

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

**API Documentation:** http://localhost:8000/docs

---

## Project Structure

```
BanglaNLP-Fewshot/
├── backend/
│   ├── main.py              # API endpoints
│   ├── model_loader.py      # Model management
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── index.js        # Entry point
│   ├── public/
│   └── package.json        # Node dependencies
├── sources/
│   ├── models/             # Pre-trained models (download separately)
│   └── code.txt            # Training scripts
├── README.md               # This file
└── SETUP.md                # Detailed setup guide
```

---

## Models

Pre-trained models (~1.5GB) are hosted on Hugging Face:

**Download:** https://huggingface.co/Jihad07/bangla-nlp-models

Models are not included in the repository due to size constraints.

---

## API Endpoints

| Endpoint   | Method | Description     |
| ---------- | ------ | --------------- |
| `/health`  | GET    | Health check    |
| `/tasks`   | GET    | Available tasks |
| `/predict` | POST   | Make prediction |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License

---

## Acknowledgments

- BanglaBERT team at BUET
- HuggingFace Transformers
- FastAPI and React communities

---

## Links

- **GitHub:** https://github.com/jihad-islam/BanglaNLP-Fewshot
- **Models:** https://huggingface.co/Jihad07/bangla-nlp-models
- **Live Demo:** [Coming Soon]
