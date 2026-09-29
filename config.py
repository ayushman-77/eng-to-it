
from pathlib import Path
from typing import Dict, Any, Optional

def get_config() -> Dict[str, Any]:
    return {
        # Hardware / Speed Optimizations (CPU-friendly defaults)
        "batch_size": 32,            # Increased batch size for better parallelization
        "num_epochs": 20,
        "lr": 3 * 10**-4,
        "seq_len": 80,               # 80 tokens covers >90% of sentences & speeds up attention by 20x!
        "d_model": 256,              # 256 dim is lightweight and fast for CPU
        "d_ff": 512,                 # Feed-forward hidden dimension
        "N": 3,                      # 3 encoder & 3 decoder layers (instead of 6)
        "h": 8,                      # 8 attention heads
        "dropout": 0.1,
        
        # Dataset & Paths
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config: Dict[str, Any], epoch: Optional[str] = None) -> Optional[str]:
    model_folder = str(config['model_folder'])
    model_basename = str(config['model_basename'])
    if epoch is None:
        return latest_weights_file_path(config)
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config: Dict[str, Any]) -> Optional[str]:
    model_folder = str(config['model_folder'])
    model_basename = str(config['model_basename'])
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])