import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM


MODELSCOPE_PATH = "/home/qikahh/models"

def get_model(model_config, model_root=MODELSCOPE_PATH, device=None):
    model_path = os.path.join(model_root, model_config['path'])
    if model_config['path'][:4] == "Qwen":
        config = AutoConfig.from_pretrained(model_path)
        config._attn_implementation = "eager"
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, **kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    torch_dtype="auto",
                                                    device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.eval()
    if device is not None:
        model = model.to(device)
    return model, tokenizer

def get_tokenizer(model_config, model_root=MODELSCOPE_PATH):
    model_path = os.path.join(model_root, model_config['path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    model_config = {
        "path": "Qwen2.5-Coder-1.5B",
    }   
    model, tokenizer = get_model(model_config)

    

    input_text = "计算后续公式得到结果：A=5+B,B=7-3,B为"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=100)

    print(tokenizer.decode(output[0]))

    qika = 1