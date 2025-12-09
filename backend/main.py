"""
FastAPI Backend for Bangla NLP System
Provides inference endpoints for both Baseline and ProtoNet models
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from model_loader import initialize_models, model_registry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Initializing models...")
    initialize_models()
    logger.info(f"Available tasks: {model_registry.get_available_tasks()}")
    yield
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Bangla NLP API",
    description="Few-Shot & Meta-Learning System for Bangla Text Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    task: str  # sentiment, hate, topic
    text: str
    mode: str = "single"  # single, comparison
    model_type: Optional[str] = None  # baseline, protonet (for single mode)


class PredictionResponse(BaseModel):
    task: str
    text: str
    mode: str
    baseline: Optional[Dict] = None
    protonet: Optional[Dict] = None
    error: Optional[str] = None


def predict_baseline(model_data: Dict, text: str) -> Dict:
    """Run inference with baseline HuggingFace model"""
    try:
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        task = model_data["task"]
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
            
            # Get label names
            labels = model_registry.get_labels(task)
            predicted_label = labels[predicted_class] if predicted_class < len(labels) else f"Class {predicted_class}"
            
            # Get all class probabilities
            all_probs = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
            
        return {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": all_probs,
            "model_type": "baseline"
        }
    except Exception as e:
        logger.error(f"Baseline prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Baseline prediction failed: {str(e)}")


def predict_protonet(model_data: Dict, text: str) -> Dict:
    """
    Run inference with ProtoNet using prototype-based distance calculation.
    
    This implements Prototypical Networks inference:
    1. Compute query embedding
    2. Calculate Euclidean distances to class prototypes
    3. Predict nearest prototype (minimum distance)
    """
    try:
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        task = model_data["task"]
        num_labels = model_data["num_labels"]
        
        # Tokenize input (query)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # ProtoNet inference: Distance-based classification
        with torch.no_grad():
            # Step 1: Get query embedding (256-dim)
            query_embedding = model(inputs["input_ids"], inputs["attention_mask"])  # [1, 256]
            
            # Step 2: Compute Euclidean distances to each class prototype
            prototypes = model.prototypes  # [num_labels, 256]
            
            # Compute distances: ||query - prototype||^2
            distances = torch.cdist(query_embedding, prototypes, p=2)[0]  # [num_labels]
            
            # Step 3: Convert distances to probabilities
            # Predicted class = nearest prototype (minimum distance)
            # Use negative distances as logits (closer = higher probability)
            negative_distances = -distances
            probs = F.softmax(negative_distances, dim=-1)  # Softmax over negative distances
            
            # Get prediction
            predicted_class = torch.argmin(distances).item()  # Nearest prototype
            confidence = probs[predicted_class].item()
            
            # Get label names
            labels = model_registry.get_labels(task)
            predicted_label = labels[predicted_class] if predicted_class < len(labels) else f"Class {predicted_class}"
            
            # Get all class probabilities
            all_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}
            
        return {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": all_probs,
            "model_type": "protonet"
        }
    except Exception as e:
        logger.error(f"ProtoNet prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"ProtoNet prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bangla NLP API",
        "version": "1.0.0",
        "endpoints": ["/health", "/tasks", "/predict"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_registry.models),
        "available_tasks": model_registry.get_available_tasks()
    }


@app.get("/tasks")
async def get_tasks():
    """Get available tasks and their labels"""
    tasks = model_registry.get_available_tasks()
    result = {}
    for task in tasks:
        result[task] = {
            "labels": model_registry.get_labels(task),
            "has_baseline": model_registry.get_model(task, "baseline") is not None,
            "has_protonet": model_registry.get_model(task, "protonet") is not None
        }
    return result


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint
    Supports both single model and comparison mode
    """
    try:
        # Validate task
        available_tasks = model_registry.get_available_tasks()
        if request.task not in available_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task. Available tasks: {available_tasks}"
            )
        
        # Validate text
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        response = PredictionResponse(
            task=request.task,
            text=request.text,
            mode=request.mode
        )
        
        # Get models
        baseline_model = model_registry.get_model(request.task, "baseline")
        protonet_model = model_registry.get_model(request.task, "protonet")
        
        # Run predictions based on mode
        if request.mode == "comparison":
            # Run both models
            if baseline_model:
                response.baseline = predict_baseline(baseline_model, request.text)
            else:
                response.error = f"Baseline model not available for {request.task}"
            
            if protonet_model:
                response.protonet = predict_protonet(protonet_model, request.text)
            else:
                if response.error:
                    response.error += f" | ProtoNet model not available for {request.task}"
                else:
                    response.error = f"ProtoNet model not available for {request.task}"
        
        else:  # single mode
            # Check if specific model type is requested
            if request.model_type == "baseline":
                if baseline_model:
                    response.baseline = predict_baseline(baseline_model, request.text)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Baseline model not available for task: {request.task}"
                    )
            elif request.model_type == "protonet":
                if protonet_model:
                    response.protonet = predict_protonet(protonet_model, request.text)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"ProtoNet model not available for task: {request.task}"
                    )
            else:
                # Default: prefer baseline, fallback to protonet
                if baseline_model:
                    response.baseline = predict_baseline(baseline_model, request.text)
                elif protonet_model:
                    response.protonet = predict_protonet(protonet_model, request.text)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No models available for task: {request.task}"
                    )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
