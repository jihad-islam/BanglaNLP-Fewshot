"""
Dynamic Model Loader for Bangla NLP System
Automatically scans and loads both Baseline (HuggingFace) and ProtoNet models
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtoNet(nn.Module):
    """ProtoNet architecture for meta-learning models"""
    def __init__(self, num_labels=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("csebuetnlp/banglabert")
        self.head = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls_token))


class ModelRegistry:
    """Registry to manage all loaded models"""
    
    def __init__(self, models_dir: str = None):
        # Auto-detect models directory relative to project root
        if models_dir is None:
            # Get the project root (parent of backend directory)
            backend_dir = Path(__file__).parent
            project_root = backend_dir.parent
            models_dir = project_root / "sources" / "models"
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.task_labels = {
            "sentiment": ["Negative", "Neutral", "Positive"],
            "hate": ["Non-Hate", "Hate"],
            "topic": ["Politics", "Sports", "Entertainment", "Economy"]
        }
        
    def _extract_task_name(self, folder_name: str) -> str:
        """Extract task name from folder name"""
        folder_lower = folder_name.lower()
        if "sentiment" in folder_lower:
            return "sentiment"
        elif "hate" in folder_lower:
            return "hate"
        elif "topic" in folder_lower:
            return "topic"
        return None
    
    def _load_baseline_model(self, model_path: Path, task: str):
        """Load HuggingFace baseline model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            model.eval()
            
            logger.info(f"✓ Loaded Baseline model for {task} from {model_path}")
            return {
                "model": model,
                "tokenizer": tokenizer,
                "type": "baseline",
                "task": task,
                "path": str(model_path)
            }
        except Exception as e:
            logger.error(f"✗ Failed to load baseline model from {model_path}: {e}")
            return None
    
    def _load_protonet_model(self, model_path: Path, task: str):
        """Load PyTorch ProtoNet model from legacy format"""
        try:
            # Get number of labels for this task
            num_labels = len(self.task_labels.get(task, []))
            
            # Initialize ProtoNet
            protonet = ProtoNet(num_labels=num_labels)
            
            # Load state dict from PyTorch legacy save format (directory-based)
            # This format requires custom unpickling to load storage references
            import pickle
            
            data_pkl_path = model_path / 'data.pkl'
            if not data_pkl_path.exists():
                raise FileNotFoundError(f"data.pkl not found in {model_path}")
            
            class StorageContext:
                """Custom unpickler context for legacy PyTorch format"""
                def __init__(self, data_dir):
                    self.data_dir = data_dir
                    
                def persistent_load(self, pid):
                    """Load persistent storage references from data/ directory"""
                    if pid[0] == 'storage':
                        storage_type, key, location, size = pid[1:]
                        tensor_file = self.data_dir / 'data' / str(key)
                        if tensor_file.exists():
                            storage = torch.FloatStorage.from_file(str(tensor_file), False, size)
                            return storage
                    return None
            
            # Load state dict with custom unpickler
            ctx = StorageContext(model_path)
            with open(data_pkl_path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                unpickler.persistent_load = ctx.persistent_load
                state_dict = unpickler.load()
            
            protonet.load_state_dict(state_dict)
            protonet.eval()
            
            # Load tokenizer (using base BanglaBERT tokenizer)
            tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
            
            logger.info(f"✓ Loaded ProtoNet model for {task} from {model_path}")
            return {
                "model": protonet,
                "tokenizer": tokenizer,
                "type": "protonet",
                "task": task,
                "path": str(model_path),
                "num_labels": num_labels
            }
        except Exception as e:
            logger.error(f"✗ Failed to load ProtoNet model from {model_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def scan_and_load(self):
        """Scan models directory and load all models"""
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return
        
        logger.info(f"Scanning models directory: {self.models_dir}")
        
        # Scan BanglaBert folder for baseline models
        banglabert_dir = self.models_dir / "BanglaBert"
        if banglabert_dir.exists():
            for model_folder in banglabert_dir.iterdir():
                if model_folder.is_dir():
                    config_path = model_folder / "config.json"
                    if config_path.exists():
                        task = self._extract_task_name(model_folder.name)
                        if task:
                            model_data = self._load_baseline_model(model_folder, task)
                            if model_data:
                                key = f"{task}_baseline"
                                self.models[key] = model_data
        
        # Scan MetaLearning folder for ProtoNet models
        metalearning_dir = self.models_dir / "MetaLearning"
        if metalearning_dir.exists():
            for task_folder in metalearning_dir.iterdir():
                if task_folder.is_dir():
                    task = self._extract_task_name(task_folder.name)
                    if task:
                        # Look for model folders (e.g., model_10shot, protonet_model_10shot)
                        for model_folder in task_folder.iterdir():
                            if model_folder.is_dir() and ("model" in model_folder.name.lower()):
                                # This is a PyTorch save directory
                                model_data = self._load_protonet_model(model_folder, task)
                                if model_data:
                                    key = f"{task}_protonet"
                                    self.models[key] = model_data
                                    break  # Only load first model found
        
        logger.info(f"Total models loaded: {len(self.models)}")
        logger.info(f"Available models: {list(self.models.keys())}")
    
    def get_model(self, task: str, model_type: str = "baseline"):
        """Get a specific model by task and type"""
        key = f"{task}_{model_type}"
        return self.models.get(key)
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks"""
        tasks = set()
        for key in self.models.keys():
            task = key.split("_")[0]
            tasks.add(task)
        return sorted(list(tasks))
    
    def get_labels(self, task: str) -> List[str]:
        """Get label names for a task"""
        return self.task_labels.get(task, [])


# Global registry instance
model_registry = ModelRegistry()


def initialize_models():
    """Initialize all models at startup"""
    model_registry.scan_and_load()
    return model_registry
