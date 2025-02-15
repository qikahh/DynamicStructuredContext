import os
import json
import logging
import torch
from tqdm import tqdm

from utils.hierarchical_context import ContextNode

if __name__ == "__main__":
    # 加载数据集
    source_code_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/Source_Code"
    data_path = "/home/lijiaa/DevEval-main/data.jsonl"
    context_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/RepoContext"
    result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval"
    
    # 读取jsonl文件
    dataset = []
    new_dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    no_data_num = 0
    for data in tqdm(dataset):
        context_dict_path = os.path.join(context_path, data["project_path"], "ContextTree.pth")
        # 首先检查文件是否存在
        if not os.path.exists(context_dict_path):
            logging.info(f"文件不存在: {context_dict_path}")
            continue
            
        context_dict = torch.load(context_dict_path) 
        
        # 测试生成
        root = context_dict[context_dict['']]
        
        target_name_list = data["namespace"].split(".")
        
        try_namespace = root.name
        suffix_list = ["<file>", "<folder>", "<class>", "<func>"]
        def dfs_ns(i, try_namespace):
            target_name = target_name_list[i]
            if i == len(target_name_list)-1:
                new_try_namespace = try_namespace+"."+target_name+"<func>"
                if new_try_namespace in context_dict[try_namespace].children:
                    return new_try_namespace
            else:
                for suffix in suffix_list:
                    new_try_namespace = try_namespace+'.'+target_name+suffix
                    if new_try_namespace in context_dict[try_namespace].children:
                        new_try_namespace = dfs_ns(i+1, new_try_namespace)
                        if new_try_namespace is not None:
                            return new_try_namespace
            for suffix in ["src<folder>", "__init__<file>", "install<folder>", "utils<folder>"]:
                new_try_namespace = try_namespace+'.'+suffix
                if new_try_namespace in context_dict[try_namespace].children:
                    new_try_namespace = dfs_ns(i, new_try_namespace)
                    if new_try_namespace is not None:
                        return new_try_namespace
            
            return None
        
        
        target_namespace = dfs_ns(0, root.name)
        if target_namespace is None:
            logging.info("字典不存在: {}.{}".format(data['project_path'], data["namespace"]))
            no_data_num += 1
            continue
            
        else:
            data["namespace"] = target_namespace
            new_dataset.append(data)
    
    logging.info("no_data_num: {}".format(no_data_num))
    # 保存数据到result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # 清除原本的文件
    if os.path.exists(os.path.join(result_path, "data.jsonl")):
        os.remove(os.path.join(result_path, "data.jsonl"))
    with open(os.path.join(result_path, "data.jsonl"), "w") as f:
        for data in new_dataset:
            f.write(json.dumps(data)+"\n")