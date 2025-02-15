
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


import logging
logging.basicConfig(level=logging.INFO)
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, Qwen2ForCausalLM

from utils.model_utils import get_model, seed_everything
from utils.dataset_utils import make_input_string
from utils.hierarchical_context import get_tooleval, initial_context, initial_input
from utils.hierarchical_context import ContextNode
from utils.hierarchical_model import HierarchicalModel

if __name__ == "__main__":

    # 初始化
    seed_everything(42)
    config = {
        "path": "Qwen2.5-Coder-3B-Instruct",
        "context": "File", # None or Structure or File
    }


    # 加载数据集
    source_code_path = "/home/lijiaa/DevEval-main/Source_Code"
    data_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/data.jsonl"
    context_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/RepoContext"
    result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval"

    # 读取jsonl文件
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    # qika 第502个样本开始有上下文依赖
    begin_idx = 502
    end_idx = None

    if end_idx is None:
        end_idx = len(dataset)
    dataset = dataset[begin_idx: end_idx]

    # 打开输出文件夹
    if config["context"] == "None":
        result_file = os.path.join(result_path, config["path"], "result_noc_2.json")
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    elif config["context"] == "Structure":
        result_file = os.path.join(result_path, config["path"], "result_st.json")
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    elif config["context"] == "File":
        result_file = os.path.join(result_path, config["path"], "result_fi_2.json")
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    generation_result = json.load(open(result_file, "r"))

    model, tokenizer = get_model(config)

    # 构建分层模型
    hierarchical_model = HierarchicalModel(model.model, model.lm_head, tokenizer)


    for idx, data in enumerate(dataset):
        idx += begin_idx
        
        logging.info("--------data {}--------".format(idx))
        
        context_dict_path = os.path.join(context_path, data["project_path"], "ContextTree.pth")
        # 首先检查文件是否存在
        if not os.path.exists(context_dict_path):
            logging.info(f"文件不存在: {context_dict_path}")
            continue
        context_dict = torch.load(context_dict_path) 
        target_namespace = data["namespace"]
        
        # 获取初始上下文
        init_context = []
        input_head = ""
        try:
            data_dict:ContextNode = context_dict[target_namespace]
            if config["context"] == "File":
                init_context = initial_input(context_dict, target_namespace)
                cut_pos = len(init_context)-1
                all_length = 0
                while cut_pos >= 0:
                    node_length = min(hierarchical_model.max_node_length, len(tokenizer.encode(context_dict[init_context[cut_pos]].content)))
                    all_length += node_length
                    if all_length > hierarchical_model.max_context_length:
                        logging.info("file too long {}, cut {}/{} nodes".format(target_namespace, cut_pos, len(init_context)))
                        init_context = init_context[cut_pos+1:]
                        break
                    cut_pos -= 1
                pass
            elif config["context"] == "Structure":
                init_context = initial_context(context_dict, target_namespace)
                init_context = init_context["cross_file_nodes"] + init_context["in_file_nodes"] + init_context["in_class_nodes"]
            else:
                init_context = []
            input_head = context_dict[data_dict.children[0]].content
        except:
            logging.info(f"字典不存在: {target_namespace}")
            continue
        
        if config["context"] in ["Structure"]:
            init_input = ""
            init_instruct = """# Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}, prohibit cyclic calling the current function!\n""".format(data["completion_path"][len(data["project_path"])+1:])
        elif config["context"] in ["File"]:
            init_input = ""
            init_instruct = """# Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}, prohibit cyclic calling the current function!\n""".format(data["completion_path"][len(data["project_path"])+1:])
            hierarchical_model.past_layers = 128
        elif config["context"] == "None":
            init_input = ""
            init_instruct = """# Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}, prohibit cyclic calling the current function!\n""".format(data["completion_path"][len(data["project_path"])+1:])
        
        input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
        init_input_ids = tokenizer.encode(input_string, add_special_tokens=False, return_tensors="pt")
        init_input_ids = init_input_ids.to(model.device)
        
        # 逐token生成
        next_token = ""
        generated = ""
        new_line = ""
        input_ids = init_input_ids
        prefix_kv = None
        result_context = []
        newline_flag = False
        lspace_level = len(input_head) - len(input_head.lstrip())
        
        with tqdm(total=hierarchical_model.max_length, desc="data {}".format(idx)) as pbar:
            for step in range(hierarchical_model.max_length):
                logging.debug("*** step {} ***".format(step))
                if (step+1)%10 == 0:
                    pbar.update(10)
                
                next_idx, seen_context, context_dict, prefix_kv = hierarchical_model.generate_step(
                        target_namespace=target_namespace,
                        input_ids=input_ids,  
                        prefix_kv=prefix_kv,
                        context_dict=context_dict, 
                        init_context_nodes=init_context)
                
                next_token = tokenizer.decode(next_idx[0])
                # 输出当前生成的token
                logging.debug(f"Generated: {next_token}")
                
                
                # 统计用到的上下文节点
                for node in seen_context:
                    if node.type == "code":
                        node = context_dict[node.parent]
                    if node.type in ["function", "class"] and node not in result_context:
                        result_context.append(node.namespace)
                
                # 清理显存
                torch.cuda.empty_cache()
                
                # 如果生成了结束标记则停止
                if next_idx.item() == tokenizer.eos_token_id or next_token == "<|endoftext|>":
                    break
                elif newline_flag and next_token[0] not in [" ", "\n"]:
                    break
                if next_token[-1] == '\n':
                    newline_flag = True
                else:
                    newline_flag = False
                    
                generated += next_token
                if next_token[-1] == '\n':
                    new_line = ""
                else:
                    new_line += next_token
                if len(new_line.lstrip()) and (len(new_line) - len(new_line.lstrip()) <= lspace_level):
                    break
                
                input_ids = next_idx.unsqueeze(0)
                # init_context = [node.namespace for node in seen_context]
                
                # input_ids = generated
                # prefix_kv = None
                pass
                
        # 将更新后的上下文字典更新到文件中
        # torch.save(context_dict, context_dict_path)
        
        result = generated.split("|endoftext|")[0]
        
        result = result.split("\n")
        real_result = []
        for line in result:
            if len(line) and (len(line) - len(line.lstrip()) <= lspace_level):
                break
            real_result.append(line)
        result = "\n".join(real_result) + '\n'
        
        # 将生成的结果保存到文件中
        if str(idx) not in generation_result:
            generation_result[str(idx)] = []
            
        generation_result[str(idx)].append({
            "result": result,
            "context": sorted(result_context),
        })
        
        logging.info("生成结果:")
        logging.info(input_string[len(init_input):]+result)
        
        if False and len(result_context):
            logging.info("上下文:")
            logging.info(result_context)
        
        pass
        
        if (idx+1)%16 == 0:
            logging.info(f"已生成{idx}条结果")
            # 在生成16条结果后，按key为generation_result字典排序并保存结果
            generation_result = dict(sorted(generation_result.items(), key=lambda x: int(x[0])))
            json.dump(generation_result, open(result_file, "w"), indent=4)
            
    generation_result = dict(sorted(generation_result.items(), key=lambda x: x[0]))
    json.dump(generation_result, open(result_file, "w"), indent=4)