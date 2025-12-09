"""
Compute class prototypes from K-shot support sets for ProtoNet models.
This script should be run once to generate prototypes from your training data.
"""
import torch
import pandas as pd
from pathlib import Path
from model_loader import ModelRegistry, ProtoNet
from transformers import AutoTokenizer
import json

def compute_prototypes_from_csv(csv_path: str, text_column: str, label_column: str, 
                                 model: ProtoNet, tokenizer, k_shot: int = 10):
    """
    Compute class prototypes from K examples per class.
    
    Args:
        csv_path: Path to training CSV
        text_column: Name of text column
        label_column: Name of label column  
        model: ProtoNet model
        tokenizer: Tokenizer
        k_shot: Number of examples per class
    
    Returns:
        prototypes: Tensor of shape [num_classes, embedding_dim]
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Get unique classes
    unique_labels = df[label_column].unique()
    num_classes = len(unique_labels)
    
    print(f"Found {num_classes} classes: {unique_labels}")
    
    # Store embeddings per class
    class_embeddings = {label: [] for label in unique_labels}
    
    # Sample K examples per class
    for label in unique_labels:
        label_data = df[df[label_column] == label]
        samples = label_data.sample(min(k_shot, len(label_data)))
        
        print(f"\nProcessing {len(samples)} examples for class '{label}'...")
        
        for text in samples[text_column]:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get embedding
            with torch.no_grad():
                embedding = model(inputs["input_ids"], inputs["attention_mask"])
                class_embeddings[label].append(embedding[0])  # Remove batch dim
    
    # Compute prototypes (mean embedding per class)
    prototypes = []
    label_order = []
    
    for label in sorted(unique_labels):  # Sort for consistency
        embeddings = torch.stack(class_embeddings[label])
        prototype = embeddings.mean(dim=0)  # Average across examples
        prototypes.append(prototype)
        label_order.append(label)
        print(f"Prototype for '{label}': shape {prototype.shape}")
    
    prototypes = torch.stack(prototypes)  # [num_classes, embedding_dim]
    print(f"\nFinal prototypes shape: {prototypes.shape}")
    
    return prototypes, label_order


def main():
    """Compute and save prototypes for all tasks"""
    
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "sources" / "trained dataset"
    
    # Initialize model registry
    registry = ModelRegistry()
    registry.scan_and_load()
    
    # Task configurations
    tasks = {
        "sentiment": {
            "csv": dataset_dir / "Sentiment Analysis.csv",
            "text_col": "text",
            "label_col": "label"
        },
        "hate": {
            "csv": dataset_dir / "Hate Speech Detection.csv",
            "text_col": "text",
            "label_col": "label"
        },
        "topic": {
            "csv": dataset_dir / "Topic Classification.csv",
            "text_col": "text",
            "label_col": "label"
        }
    }
    
    # Compute prototypes for each task
    for task_name, config in tasks.items():
        print(f"\n{'='*60}")
        print(f"Computing prototypes for {task_name.upper()}")
        print(f"{'='*60}")
        
        # Get ProtoNet model
        model_key = f"{task_name}_protonet"
        if model_key not in registry.models:
            print(f"❌ ProtoNet model not found for {task_name}")
            continue
        
        model_data = registry.models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Compute prototypes
        try:
            prototypes, label_order = compute_prototypes_from_csv(
                str(config["csv"]),
                config["text_col"],
                config["label_col"],
                model,
                tokenizer,
                k_shot=10
            )
            
            # Save prototypes
            output_file = project_root / "backend" / f"{task_name}_prototypes.pt"
            torch.save({
                "prototypes": prototypes,
                "label_order": label_order
            }, output_file)
            
            print(f"\n✅ Saved prototypes to {output_file}")
            
        except Exception as e:
            print(f"\n❌ Error computing prototypes for {task_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
