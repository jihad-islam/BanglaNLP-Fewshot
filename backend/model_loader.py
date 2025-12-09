"""
Smart Model Loader for Bangla NLP System
Hybrid Logic:
1. Checks local 'sources/models' first (For Local Dev)
2. Downloads from Hugging Face if local files missing (For Render Deployment)
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import logging
from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
HF_REPO_ID = "Jihad07/bangla-nlp-models" # Your Hugging Face Repo
# ---------------------

class ProtoNet(nn.Module):
    """ProtoNet architecture for meta-learning models - NOT a standard classifier!"""
    def __init__(self, num_labels=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("csebuetnlp/banglabert")
        self.head = nn.Linear(768, 256)  # Embedding projection: 768 → 256
        self.dropout = nn.Dropout(0.3)
        self.num_labels = num_labels
        self.prototypes = None  # Will store class prototypes (num_labels x 256)
        
    def forward(self, input_ids, attention_mask):
        """Get 256-dimensional embeddings (NOT logits!)"""
        out = self.bert(input_ids, attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]  # [batch, 768]
        embeddings = self.head(self.dropout(cls_token))  # [batch, 256]
        return embeddings
    
    def set_prototypes(self, prototypes):
        """Set class prototypes for distance-based inference"""
        self.prototypes = prototypes  # Shape: [num_classes, 256]


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.task_labels = {
            "sentiment": ["Positive", "Negative", "Neutral"],
            "hate": ["Non-Hate", "Hate"],  # Fixed: Model trained with Non-Hate=0, Hate=1
            "topic": ["Bangladesh", "International", "Sports", "Entertainment"]  # Fixed: Sports not Sport
        }
        
        # Define paths
        self.backend_dir = Path(__file__).parent
        self.project_root = self.backend_dir.parent
        self.local_models_dir = self.project_root / "sources" / "models"

    def _get_model_path(self, task: str, model_type: str, resource_name: str) -> str:
        """
        Smart Path Finder:
        Returns local path if it exists, otherwise downloads from Hugging Face.
        """
        # 1. Try Local Path
        if model_type == "baseline":
            # Map task to local folder names based on your structure
            # Adjust these if your local folder names are different!
            folder_map = {
                "sentiment": "BanglaBERT Sentiment Analysis",
                "hate": "BanglaBERT Hate Speech Detection",
                "topic": "BanglaBERT Topic Classification"
            }
            local_path = self.local_models_dir / "BanglaBert" / folder_map.get(task, "")
            
            if local_path.exists() and (local_path / "config.json").exists():
                logger.info(f"📂 Found local model for {task} ({model_type})")
                return str(local_path)

        elif model_type == "protonet":
            # First try .pth files (if converted)
            file_map = {
                "sentiment": "sentiment_proto.pth", 
                "hate": "hate_proto.pth",
                "topic": "topic_proto.pth"
            }
            pth_path = self.local_models_dir / "MetaLearning" / file_map.get(task, "")
            
            if pth_path.exists():
                logger.info(f"📂 Found local .pth model for {task} ({model_type})")
                return str(pth_path)
            
            # Fallback to legacy directory format
            legacy_folder_map = {
                "sentiment": "Meta Learning Sentiment Analysis",
                "hate": "Meta Learning Hate Speech Detection",
                "topic": "Meta Learning Topic Classification"
            }
            legacy_path = self.local_models_dir / "MetaLearning" / legacy_folder_map.get(task, "")
            
            # Look for model folders inside (e.g., model_10shot, protonet_model_10shot)
            if legacy_path.exists():
                for model_folder in legacy_path.iterdir():
                    if model_folder.is_dir() and "model" in model_folder.name.lower():
                        data_pkl = model_folder / "data.pkl"
                        if data_pkl.exists():
                            logger.info(f"📂 Found local legacy model for {task} ({model_type})")
                            return str(model_folder)

        # 2. Fallback to Hugging Face
        logger.info(f"☁️ Local model missing for {task}. Downloading from Hugging Face...")
        try:
            if model_type == "baseline":
                # Matches the folder name you uploaded to HF
                return snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{resource_name}/*")
            else:
                # Matches the filename you uploaded to HF
                return hf_hub_download(repo_id=HF_REPO_ID, filename=resource_name)
        except Exception as e:
            logger.error(f"Failed to download from HF: {e}")
            return None

    def _load_baseline_model(self, task: str, hf_folder_name: str):
        path = self._get_model_path(task, "baseline", hf_folder_name)
        if not path: return None

        # If downloaded from HF snapshot, the path might be the parent dir, need to append folder name
        # If local, path is already correct
        if "snapshots" in str(path) and not str(path).endswith(hf_folder_name):
             full_path = Path(path) / hf_folder_name
        else:
             full_path = Path(path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(full_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(full_path))
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "type": "baseline", "task": task}
        except Exception as e:
            logger.error(f"Error loading baseline {task}: {e}")
            return None

    def _load_protonet_model(self, task: str, hf_filename: str):
        path = self._get_model_path(task, "protonet", hf_filename)
        if not path: return None

        try:
            num_labels = len(self.task_labels.get(task, []))
            protonet = ProtoNet(num_labels=num_labels)
            
            # Check if it's a direct .pth file (from HF) or directory (local legacy format)
            path_obj = Path(path)
            if path_obj.is_file() and path_obj.suffix == '.pth':
                # Direct .pth file from HuggingFace
                state_dict = torch.load(path, map_location=torch.device('cpu'))
            elif path_obj.is_dir():
                # Legacy directory format (local), need to load from data.pkl
                import pickle
                data_pkl = path_obj / 'data.pkl'
                if not data_pkl.exists():
                    raise FileNotFoundError(f"data.pkl not found in {path_obj}")
                
                class StorageContext:
                    def __init__(self, data_dir):
                        self.data_dir = data_dir
                    def persistent_load(self, pid):
                        if pid[0] == 'storage':
                            storage_type, key, location, size = pid[1:]
                            tensor_file = self.data_dir / 'data' / str(key)
                            if tensor_file.exists():
                                storage = torch.FloatStorage.from_file(str(tensor_file), False, size)
                                return storage
                        return None
                
                ctx = StorageContext(path_obj)
                with open(data_pkl, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    unpickler.persistent_load = ctx.persistent_load
                    state_dict = unpickler.load()
            else:
                raise ValueError(f"Invalid model path: {path}")
            
            # Check if state_dict has classifier weights
            has_classifier = any('classifier' in key for key in state_dict.keys())
            
            if not has_classifier:
                logger.warning(f"ProtoNet {task} doesn't have classifier layer - loading with strict=False")
                protonet.load_state_dict(state_dict, strict=False)
            else:
                protonet.load_state_dict(state_dict)
            
            protonet.eval()
            
            # Initialize prototypes using random normalized vectors
            # In production, these would be computed from K-shot support set
            with torch.no_grad():
                prototypes = torch.randn(num_labels, 256) * 0.1
                prototypes = prototypes / torch.norm(prototypes, dim=1, keepdim=True)
                protonet.set_prototypes(prototypes)
            
            logger.info(f"ProtoNet {task} loaded with {num_labels} class prototypes")
            
            tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
            return {"model": protonet, "tokenizer": tokenizer, "type": "protonet", "task": task, "num_labels": num_labels}
        except Exception as e:
            logger.error(f"Error loading ProtoNet {task}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def scan_and_load(self):
        logger.info("Initializing models...")
        
        # Load Baselines (Using HF folder names as keys)
        self.models["sentiment_baseline"] = self._load_baseline_model("sentiment", "sentiment_baseline")
        self.models["topic_baseline"] = self._load_baseline_model("topic", "topic_baseline")
        self.models["hate_baseline"] = self._load_baseline_model("hate", "hate_baseline")
        
        # Load ProtoNets (Using HF filenames as keys)
        self.models["sentiment_protonet"] = self._load_protonet_model("sentiment", "sentiment_proto.pth")
        self.models["topic_protonet"] = self._load_protonet_model("topic", "topic_proto.pth")
        self.models["hate_protonet"] = self._load_protonet_model("hate", "hate_proto.pth")
        
        # Cleanup failed loads
        self.models = {k: v for k, v in self.models.items() if v is not None}
        logger.info(f"Total models loaded: {len(self.models)}")

    def get_model(self, task: str, model_type: str = "baseline"):
        return self.models.get(f"{task}_{model_type}")
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks"""
        tasks = set()
        for key in self.models.keys():
            task = key.split("_")[0]
            tasks.add(task)
        return sorted(list(tasks))
    
    def get_labels(self, task: str) -> List[str]:
        return self.task_labels.get(task, [])

model_registry = ModelRegistry()

def initialize_models():
    model_registry.scan_and_load()
    return model_registry