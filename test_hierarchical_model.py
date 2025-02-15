import logging
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, Qwen2ForCausalLM


from utils.model_utils import get_model, seed_everything
from utils.hierarchical_context import get_tooleval, initial_context, initial_input
from utils.hierarchical_context import ContextNode
from utils.hierarchical_model import HierarchicalModel

# 初始化
seed_everything(42)
model_config = {
    "path": "Qwen2.5-Coder-3B-Instruct",
}
model, tokenizer = get_model(model_config)

# 构建分层模型
hierarchical_model = HierarchicalModel(model.model, model.lm_head, tokenizer)

# 加载数据集
source_code_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/Source_Code"
data_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/data.jsonl"
result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/RepoContext"

# 读取jsonl文件
dataset = []
with open(data_path, "r") as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data)
        
from utils.dataset_utils import make_input_string
for data in dataset:
    context_dict_path = os.path.join(result_path, data["project_path"], "ContextTree.pth")
    # 首先检查文件是否存在
    if not os.path.exists(context_dict_path):
        print(f"文件不存在: {context_dict_path}")
        continue
        
    context_dict = torch.load(context_dict_path) 
    
    # 测试生成
    root = context_dict[context_dict['']]
    
    target_namespace = root.name+'.'+data["namespace"]
    
    # 获取初始上下文
    init_context = initial_context(context_dict, target_namespace)
    # 获取输入信息
    try:
        data_dict:ContextNode = context_dict[target_namespace]
        input_nodes = initial_input(context_dict, target_namespace)
        input_head = context_dict[data_dict.children[0]].content
    except:
        print(f"字典不存在: {target_namespace} 尝试直接获取函数头")
        continue
    
    input_prefix = [context_dict[node] for node in input_nodes]
    init_input = "".join([context_dict[node].content for node in input_nodes]) 
    init_input = ""
    
    init_instruct = f"""# Complete the functional code directly based on the provided code prefix without generating any additional content! The code prefix consists of two parts. Before this paragraph, the first part contains contextual information about the project, such as folder names starting with "# folder:", file names starting with "# file:", and code snippets within the file that follow the file name. The second part is the current function header and comments provided later, and the generated result needs to be aligned with the indentation of the function header. After generating the function, output the "<|im_end|>" tag as the end\n"""
    
    input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "ToolEval")
    
    input_ids = tokenizer.encode(input_string, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    
    # 逐token生成
    generated = input_ids
    init_context = init_context["cross_file_nodes"] + init_context["in_file_nodes"] + init_context["in_class_nodes"]
    
    for _ in range(hierarchical_model.max_length):
        next_token, curr_context, context_dict, past_key_values = hierarchical_model.generate_step(
                target_namespace=target_namespace,
                input_ids=generated,  
                past_key_values=None,
                context_dict=context_dict, 
                init_context_nodes=init_context)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # 将更新后的上下文字典更新到文件中
        print(tokenizer.decode(next_token[0]))
        # torch.save(context_dict, context_dict_path)
        
        # 如果生成了结束标记则停止
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    print("生成结果:")
    print(tokenizer.decode(generated[0]))