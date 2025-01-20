import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") 
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import (AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer)


from utils import seed_everything, get_model, get_repoeval


MODEL_CONFIGS = {
    "CodeLlama-7B": {
        "path": "CodeLlama-7B", 
    },
    "Qwen2.5-Coder-1.5B": {
        "path": "Qwen2.5-Coder-1.5B",
    },
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-1.5B")
    parser.add_argument("--max_length", type=int, default=16000)
    parser.add_argument('--dataset', type=str, default="ToolEval")
    parser.add_argument("--dataset_root", type=str, default="/home/qikahh/projects/Structured_Code_Context/RepoCoder")
    parser.add_argument('--mode', default="window", help="KV mode")
    args = parser.parse_args()
    print("max_length: ", args.max_length)
    return args

# nohup python -u main.py > output.log 2>&1 &
if __name__ == '__main__':
    seed_everything(42)
    args = get_args()
    world_size = torch.cuda.device_count()
    
    if args.dataset == "RepoEval":
        dataset = get_repoeval(args.dataset_root)
    
    elif args.dataset == "ToolEval":
        
    
    model_config = MODEL_CONFIGS[args.model_name]
    model, tokenizer = get_model(model_config)
    
    for data in dataset:
        data_context = make_context(data[''])
    
    