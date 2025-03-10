import os
import ujson as json
import logging
import torch
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi

from utils.model_utils import get_tokenizer, seed_everything
from utils.hierarchical_context import ContextNode
from utils.dataset_utils import make_input_string


source_code_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/Source_Code"
context_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/RepoContext"
root_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval"
rawdata_path = "/home/lijiaa/DevEval-main/data.jsonl"

seed_everything(42) # 42 18 63
config = {
    "path": "Qwen2.5-Coder-3B-Instruct",
    "context": "Structure", # None or Structure or File or Oracle
}

def remove_extend():
    result_path = os.path.join(root_path, config["path"], "result_st_2.json")
    
    
    with open(result_path, "r") as f:
        dataset = json.load(f)
    
    for i, key in tqdm(enumerate(dataset)):
        dataset[key] = dataset[key][:1]
        
    # 重新保存
    with open(result_path, "w") as f:
        json.dump(dataset, f, indent=4)


def make_result_deveval():
    result_path = os.path.join(root_path, config["path"], "result_noc.json")
    dataset_path = os.path.join(root_path,  "data.jsonl")
    
    with open(result_path, "r") as f:
        results = json.load(f)
    
    # jsonl文件读取
    dataset = []
    for line in tqdm(open(dataset_path, "r")):
        data = json.loads(line)
        dataset.append(data)
    
    rawdataset = []
    for line in tqdm(open(rawdata_path, "r")):
        data = json.loads(line)
        rawdataset.append(data)
    
    final_result = []
    
    idx = 0
    for i, data in tqdm(enumerate(dataset)):
        while idx<len(rawdataset) and (rawdataset[idx]["completion_path"] != data["completion_path"] or rawdataset[idx]["requirement"]["Functionality"] != data["requirement"]["Functionality"]):
            idx += 1
        
        if idx >= len(rawdataset):
            logging.info("No rawdata found")
            break
        
        namespace = rawdataset[idx]["namespace"]
        result_list = results[str(i)]
        
        for result in result_list:
            final_result.append({
                "namespace": namespace,
                "completion": result["result"],
            })
    
    with open(os.path.join(root_path, config["path"], "final_result_noc.jsonl"), "w") as f:
        for result in final_result:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    make_result_deveval()
    