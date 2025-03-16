
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import json
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
import cProfile
import yappi
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, Qwen2ForCausalLM

from utils.model_utils import get_model, seed_everything
from utils.dataset_utils import make_input_string
from utils.hierarchical_context import get_tooleval, initial_context, initial_input, initial_bm25, get_namespace
from utils.hierarchical_context import ContextNode
from utils.hierarchical_model import HierarchicalModel
from utils.visualize import visualize_line

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--context', type=str, default="Structure")
    parser.add_argument('--info', type=str, default=None)
    parser.add_argument('--data_id', type=str, default="-1")
    parser.add_argument('--gen_num', type=int, default=1)
    parser.add_argument('--data_start', type=int, default=515)
    parser.add_argument('--data_end', type=int, default=None)
    return parser.parse_args()

def main():
    # 用随机数初始化
    random_seed = np.random.randint(0, 512)
    seed_everything(random_seed)
    args = get_parser()
    config = {
        "path": "Qwen2.5-Coder-3B-Instruct",
        "context": args.context, # None or Structure or File or Oracle or BM25
    }

    # 输出当前信息 时间 pid 生成方式
    pid = os.getpid()
    logging.info("time: {}".format(os.popen("date").read()))
    logging.info("pid: {}".format(pid))
    logging.info("seed: {}".format(random_seed))
    logging.info("model: {}".format(config["path"]))
    logging.info("generate: {}".format(config["context"]))


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
    # begin_idx = 629
    begin_idx = args.data_start
    end_idx = args.data_end

    if end_idx is None:
        end_idx = len(dataset)
    dataset = dataset[begin_idx: end_idx]

    # 打开输出文件夹
    if config["context"] == "None":
        result_file = os.path.join(result_path, config["path"], "result_noc_{}.json".format(args.data_id))
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    elif config["context"] == "Structure":
        file_name = "result_st_{}.json".format(args.data_id)
        if args.info != None:
            file_name = "result_st_{}_{}.json".format(args.info, args.data_id)
        result_file = os.path.join(result_path, config["path"], file_name)
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    elif config["context"] == "File":
        result_file = os.path.join(result_path, config["path"], "result_fi_{}.json".format(args.data_id))
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)
                
    elif config["context"] == "BM25":
        result_file = os.path.join(result_path, config["path"], "result_bm_{}.json".format(args.data_id))
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)
    
    elif config["context"] == "Oracle":
        result_file = os.path.join(result_path, config["path"], "result_oc_{}.json".format(args.data_id))
        if not os.path.exists(result_file):
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 创建json文件
            
            with open(result_file, "w") as f:
                json.dump({}, f)

    logging.info("result_path: {}".format(result_file))
    generation_result = {} # json.load(open(result_file, "r"))

    
    model, tokenizer = get_model(config)

    # 构建分层模型
    hierarchical_model = HierarchicalModel(model.model, model.lm_head, tokenizer)
    data_type_attn = []

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
        except:
            logging.info(f"字典不存在: {target_namespace}")
            continue
        
        input_head = context_dict[data_dict.children[0]].content
        target_nodes = data_dict.dfs_leave(context_dict)
        input_body = "".join([node.content for node in target_nodes[1:]])
        target_length = len(tokenizer.tokenize(input_body))
        max_gen_length = min(max(2*target_length, 128), hierarchical_model.max_length)
        
        if config["context"] in ["Structure"]:
            init_input = ""
            init_instruct = """# Instruction: Implement the function body based on the provided in-file contents and other cross-file contents in the current project, without generating any additional content! The current function is located in file {}, you can use modules from the current project and you MUST write import statements FIRST to call modules from other files.\n""".format(data["completion_path"][len(data["project_path"])+1:])
            input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
            init_context = initial_context(context_dict, target_namespace)
            file_head = init_context["in_file_nodes"][0]
            context_dict[file_head].content = "\n## Here are in-file contents from {}. We simplified the code by removing some code blocks.\n\n".format(context_dict[file_head].content[2:-1])
            
            infile_context = init_context["in_file_nodes"] + init_context["in_class_nodes"]
            cut_pos = len(infile_context)-1
            all_length = 0
            while cut_pos >= 0:
                node_length = min(hierarchical_model.max_node_length, len(tokenizer.encode(context_dict[infile_context[cut_pos]].content)))
                all_length += node_length
                if all_length > hierarchical_model.max_context_length:
                    logging.info("file too long {}, cut {}/{} nodes".format(target_namespace, cut_pos, len(infile_context)))
                    infile_context = infile_context[cut_pos+1:]
                    break
                cut_pos -= 1
            
            init_context = init_context["cross_file_nodes"] + infile_context
            ini_prefix_kv = None
            ini_prefix_pos = None
            
        elif config["context"] in ["File"]:
            init_input = ""
            init_instruct = """# Instruction: Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}.\n""".format(data["completion_path"][len(data["project_path"])+1:])
            input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
            hierarchical_model.begin_layer = 128
            hierarchical_model.end_layer = 128
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
            ini_prefix_kv = None
            ini_prefix_pos = None
            pass
            
        elif config["context"] == "None":
            init_input = ""
            init_instruct = """# Instruction: Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}.\n""".format(data["completion_path"][len(data["project_path"])+1:])
            input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
            hierarchical_model.begin_layer = 128
            hierarchical_model.end_layer = 128
            ini_prefix_kv = None
            ini_prefix_pos = None
            
        elif config["context"] == "BM25":
            init_input = ""
            init_instruct = """# Instruction: Implement the function body based on the provided code prefix and other resources in the current project, without generating any additional content! The current function is located in file {}, you can use resources from current file and other files and you MUST write import statements FIRST to call resources from other files.\n""".format(data["completion_path"][len(data["project_path"])+1:])
            input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
            hierarchical_model.begin_layer = 128
            hierarchical_model.end_layer = 128
            init_context = initial_input(context_dict, target_namespace)
            cross_list, infile_list = initial_bm25(context_dict, target_namespace, input_string, tokenizer=tokenizer, init_context=init_context, max_cross_num=hierarchical_model.max_crossfile_node_num, max_file_length=hierarchical_model.max_context_length, max_infile_num=hierarchical_model.max_infile_node_num)
            infile_parts = hierarchical_model.cluster_brothers([context_dict[ns] for ns in infile_list], context_dict[target_namespace], context_dict)
            infile_str = ""
            for part in infile_parts:
                infile_str += "".join([node.content for node in part[1:]])
            input_string = infile_str + input_string
            input_tokens = tokenizer.tokenize(input_string, add_special_tokens=False)
            input_tokens = input_tokens[-hierarchical_model.max_context_length:]
            input_string = tokenizer.convert_tokens_to_string(input_tokens)
            input_string = "\n## Here are in-file contents from {}. We simplified the code by removing some code blocks.\n\n".format(infile_parts[0][0].content[2:-1])+input_string

            ini_prefix_kv=None
            ini_prefix_pos = None
            init_context = cross_list
            
            
        elif config["context"] == "Oracle":
            init_input = ""
            init_instruct = """# Implement the function body based on the provided code prefix, without generating any additional content! The current function is located in file {}, prohibit cyclic calling the current function!\n""".format(data["completion_path"][len(data["project_path"])+1:])
            input_string = make_input_string(init_input, init_instruct, input_head, data["requirement"], "DevEval")
            init_context = []
            depen_list = data["dependency"]["cross_file"] + data["dependency"]["intra_file"] + data["dependency"]["intra_class"]
            for depen in depen_list:
                depen_ns = get_namespace(depen, context_dict)
                if depen_ns is None:
                    depen = depen.rsplit(".", 1)[0]
                    depen_ns = get_namespace(depen, context_dict)
                if depen_ns is not None and depen_ns not in init_context:
                    init_context.append(depen_ns)
            ini_prefix_kv = None
            
        hierarchical_model.context_dict = context_dict
        hierarchical_model.target_node = context_dict[target_namespace]
            
        init_input_ids = tokenizer.encode(input_string, add_special_tokens=False, return_tensors="pt")
        init_input_ids = init_input_ids.to(model.device)
        
        
        # 逐token生成, 生成次数为gen_num, 并重新设置随机种子
        for time in range(args.gen_num):
            seed_everything(random_seed+time)
            next_token = ""
            generated = ""
            new_line = input_string.split("\n")[-1]
            input_ids = init_input_ids
            prefix_kv = ini_prefix_kv
            prefix_pos = None
            result_context = []
            newline_flag = False
            lspace_level = len(input_head) - len(input_head.lstrip())
            max_position = 0
            max_length = 0
            
            
            """# 设置时钟类型（CPU时间或挂钟时间）
            yappi.set_clock_type("wall")  # 可选"wall"模式

            # 启动分析器（支持线程级控制）
            yappi.start(builtins=True, profile_threads=False)"""
            
            with torch.no_grad(), tqdm(total=max_gen_length, desc="data {}".format(idx)) as pbar:
                
                for step in range(max_gen_length):
                    logging.debug("*** step {} ***".format(step))
                    if (step+1)%10 == 0:
                        pbar.update(10)
                    
                    
                    info_dict = hierarchical_model.generate_step(target_namespace=target_namespace,input_ids=input_ids, prefix_kv=prefix_kv, prefix_pos=prefix_pos, init_context_nodes=init_context)

                    
                    next_idx = info_dict["next_token"]
                    seen_context = info_dict["seen_context"]
                    curr_context = info_dict["curr_context"]
                    prefix_kv = info_dict["prefix_kv"]
                    prefix_pos = info_dict["prefix_pos"]
                    max_position = max(max_position, info_dict["position"])
                    max_length = max(max_length, info_dict["length"])
                    
                    step_type_attn = info_dict["type_attn_by_layer"]
                    data_type_attn.append(step_type_attn)
                    
                    next_token = tokenizer.decode(next_idx[0])
                    
                    # 输出当前生成的token
                    logging.debug(f"Generated: {next_token}")
                    
                    
                    # 统计用到的上下文节点
                    seen_context = [node for node in curr_context]
                    for node in seen_context:
                        if node.type == "code":
                            node = context_dict[node.parent]
                        if node.type in ["function", "class"] and node.namespace not in result_context:
                            result_context.append(node.namespace)
                    
                    # 清理显存
                    # torch.cuda.empty_cache()
                    
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
            
            if time == 0:
                generation_result[str(idx)].append({
                    "result": result,
                    "context": sorted(result_context),
                })
            else:
                generation_result[str(idx)].append({
                    "result": result,
                })
            
            logging.info("函数头: \n{}".format(input_head))
            logging.info("目标函数体: \n{}".format(input_body))
            logging.info("生成结果: \n{}".format(result))
            logging.info("-"*40)
            logging.info("kv length: {}".format(max_length))
            logging.info("max position: {}".format(max_position))
            logging.info("context_num: {}".format(len(result_context)))
            logging.info("target_namespace: {}".format(target_namespace))
            logging.info("time: {}".format(os.popen("date").read()))
            logging.info("*"*40)
            
            if False and len(result_context):
                logging.info("上下文:")
                logging.info(result_context)
                
            # 为attn绘制折线图，包含均值和标准差
            """attn_value = torch.stack(data_type_attn, dim=0)
            # 交换
            attn_value = attn_value.permute(2, 0, 1)
            attn_value[1:,:,:] /= attn_value[:1,:,:]
            visualize_line(attn_value[1:,:,:], label=["in_file", "cross_file"])
            data_type_attn = []"""
            
            
            """yappi.stop()
            # 获取函数级别统计数据
            stats = yappi.get_func_stats()
            stats.print_all(columns={ 0: ("name", 128),1: ("ncall", 5),2: ("tsub", 8),3: ("ttot", 8),4: ("tavg", 8) })"""
            pass
        
        if (idx+1)%2 == 0:
            logging.info(f"已生成{idx}条结果")
            
            # 在生成足够结果后，按key为generation_result字典排序并保存结果
            old_gen_result = json.load(open(result_file, "r"))
            # 合并两个dict
            for key in generation_result:
                if key in old_gen_result:
                    old_gen_result[key] += generation_result[key]
                else:
                    old_gen_result[key] = generation_result[key]
            all_result = dict(sorted(old_gen_result.items(), key=lambda x: int(x[0])))
            json.dump(all_result, open(result_file, "w"), indent=4)
            generation_result = {}
    
    
    old_gen_result = json.load(open(result_file, "r"))
    # 合并两个dict
    for key in generation_result:
        if key in old_gen_result:
            old_gen_result[key] += generation_result[key]
        else:
            old_gen_result[key] = generation_result[key]
    generation_result = dict(sorted(old_gen_result.items(), key=lambda x: int(x[0])))
    json.dump(generation_result, open(result_file, "w"), indent=4)
    
    
if __name__ == "__main__":
    main()
    