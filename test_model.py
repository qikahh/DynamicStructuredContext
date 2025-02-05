import logging
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pickle

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, Qwen2ForCausalLM


from utils.model_utils import get_model, seed_everything
from utils.hierarchical_context import get_tooleval, initial_context, initial_input
from utils.hierarchical_context import ContextNode
from utils.hierarchical_model import HierarchicalModel

def generate_step(
        model,
        input_ids: torch.Tensor, 
        past_key_values, 
    ):
    """
    执行一步生成,返回生成的token_id
    """
    # 获取输入的嵌入表示
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    # 编码位置信息
    position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    
    # 获取注意力掩码
    causal_mask = model.model._update_causal_mask(
        None, inputs_embeds, position_ids, past_key_values, output_attentions=True
    )
    
    # 创建位置编码
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    
    # 初始化隐藏状态
    hidden_states = inputs_embeds
    
    # 逐层处理
    layer_outputs = []
    for layer_idx, layer in enumerate(model.model.layers):
        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=True,
            use_cache=True,
            cache_position=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
    
    # 最后进行归一化
    outputs = model.model.norm(hidden_states)
    outputs = (outputs,) + layer_outputs[1:]
    
    # 获取logits
    logits = model.lm_head(outputs[0])
    
    # 获取最后一个时间步的logits
    next_token_logits = logits[:, -1, :]
    
    # 采样获取下一个token
    next_token = torch.argmax(next_token_logits, dim=-1)
    
    return next_token

# 初始化
seed_everything(42)
model_config = {
    "path": "Qwen2.5-Coder-3B-Instruct",
}
model, tokenizer = get_model(model_config)


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
    
    init_instruct = f"""# Complete the function code directly based on the provided code prefix, without generating any additional content! The code prefix consists of two parts. The first part contains contextual information within the project, such as other folder names starting with "# folder:", other file names starting with "# file:", and other code snippets. The second part is the function header and comments of the current function, and the generated result needs to be aligned with the indentation of the function header. After the function is generated, output the '<|im_end|>' tag as the end\n"""
    input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "ToolEval")
    
    instruct_ids = tokenizer.encode(init_instruct, add_special_tokens=False, return_tensors="pt")
    instruct_ids = instruct_ids.to(model.device)
    
    input_ids = tokenizer.encode(input_string, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    # 逐token生成
    generated = torch.cat([instruct_ids, input_ids], dim=1)
    begin_length = generated.shape[1]
    
    # 使用模型进行生成
    for _ in range(2048):
        next_token = generate_step(
                model = model,
                input_ids=generated,  
                past_key_values=None)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # 将更新后的上下文字典更新到文件中
        # print(tokenizer.decode(next_token[0]))
        # torch.save(context_dict, context_dict_path)
        
        # 如果生成了结束标记则停止
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    print("生成结果:")
    print(tokenizer.decode(generated[0]))
    
    pass