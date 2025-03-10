import os
import json
import logging
import torch
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi

from utils.model_utils import get_tokenizer, seed_everything
from utils.hierarchical_context import ContextNode
from utils.dataset_utils import make_input_string


source_code_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/Source_Code"
data_path = "/home/lijiaa/DevEval-main/data.jsonl"
context_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/RepoContext"
result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval"

seed_everything(42) # 42 18 63
config = {
    "path": "Qwen2.5-Coder-3B-Instruct",
    "context": "Structure", # None or Structure or File or Oracle
}

def update_namespace(context_dict, namespace):
    root = context_dict[context_dict['']]
    target_name_list = namespace.split(".")
    
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
    return target_namespace
    
def update_dataset_namespace():

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

def update_dataset_bodypos():
    dataset_path = os.path.join(result_path, "data.jsonl")
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    for idx, data in tqdm(enumerate(dataset)):
        context_dict_path = os.path.join(context_path, data["project_path"], "ContextTree.pth")
        context_dict = torch.load(context_dict_path) 
        target_namespace = data["namespace"]
        data_dict = context_dict[target_namespace]
        first_child = data_dict.children[0]
        last_child = data_dict.children[-1]
        begin_pos = context_dict[first_child].end_line
        end_pos = context_dict[last_child].end_line
        if data["body_position"][0] != begin_pos or data["body_position"][1] != end_pos:
            logging.info("{} {} {} {}".format(data["project_path"], data["namespace"], data["body_position"], [begin_pos, end_pos]))
            data["body_position"] = [begin_pos, end_pos]
            dataset[idx] = data
            
    # 保存回去dataset_path
    with open(dataset_path, "w") as f:
        for data in dataset:
            f.write(json.dumps(data)+"\n")
    
    

def get_leaf_nodes(context_dict, root):
    leaf_nodes = []
    def dfs(node):
        if len(node.children) == 0:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                dfs(context_dict[child])
    dfs(root)
    return leaf_nodes
def statis_codefile_length():
    tokenizer = get_tokenizer(config)
    
    length_by_project = []
    file_num_by_project = []
    for project_type in tqdm(os.listdir(context_path)):
        for project_name in os.listdir(os.path.join(context_path, project_type)):
            code_lenth = 0
            file_num = 0
            context_dict = torch.load(os.path.join(context_path, project_type, project_name, "ContextTree.pth"))
            
            for node in tqdm(context_dict.values()):
                if isinstance(node, ContextNode) and node.type == "file":
                    leaf_nodes = get_leaf_nodes(context_dict, node)
                    for leaf_node in leaf_nodes[1:]:
                        code_lenth += len(tokenizer.encode(leaf_node.content))
                    file_num += 1
            
            length_by_project.append(code_lenth/file_num)
            file_num_by_project.append(file_num)
    
    print("mean length: {}".format(sum(length_by_project)/len(length_by_project)))
    print("mean file num: {}".format(sum(file_num_by_project)/len(file_num_by_project)))
         

def statis_func_dependency():
    context_path = "/home/lijiaa/DevEval/DevEval-main/Dependency_Data"
    
    cross_num_by_project = []
    infile_num_by_project = []
    depen_num_by_project = []
    func_num_by_project = []
    for project_type in tqdm(os.listdir(context_path)):
        for project_name in os.listdir(os.path.join(context_path, project_type)):
            cross_num = 0
            infile_num = 0
            depen_num = 0
            func_num = 0
            context_dict_path = os.path.join(context_path, project_type, project_name, "all_call_info.json")
            context_dict = json.load(open(context_dict_path, "r"))
            
            for file_data in tqdm(context_dict.values()):
                for data in file_data.values():
                    if data["type"] == "method" and "test" not in data["name"]:
                        func_num += 1
                        if len(data["in_object"]) > 0:
                            cross_num += 1
                        if len(data["in_file"]+data["in_class"])>0:
                            infile_num += 1
                        if len(data["in_object"]+data["in_file"]+data["in_class"])>0:
                            depen_num += 1
            
            cross_num_by_project.append(cross_num)
            infile_num_by_project.append(infile_num)
            depen_num_by_project.append(depen_num)
            func_num_by_project.append(func_num)
            
    
    print("mean cross: {}".format(sum(cross_num_by_project)/sum(func_num_by_project)))
    print("mean infile: {}".format(sum(infile_num_by_project)/sum(func_num_by_project)))
    print("mean depen: {}".format(sum(depen_num_by_project)/sum(func_num_by_project)))
    pass
    
def make_BM25_top():
    data_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/data.jsonl"
    # 读取jsonl文件
    dataset = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    data = dataset[901]
    
    context_dict_path = os.path.join(context_path, data["project_path"], "ContextTree.pth")
    context_dict = torch.load(context_dict_path) 
    target_namespace = data["namespace"]
    
    data_dict:ContextNode = context_dict[target_namespace]
    
    input_head = context_dict[data_dict.children[0]].content
    
    input_string = """
    def gather_named_configs(
        self,
    ) -> Generator[Tuple[str, Union[ConfigScope, ConfigDict, str]], None, None]:
        \"\"\"Collect all named configs from this ingredient and its sub-ingredients.

        Yields
        ------
        config_name
            The full (dotted) name of the named config.
        config
            The corresponding named config.
        \"\"\"
    """
    
    # 根据imput_string从context_dict中查询BM25最大的节点
    
    
    all_func = [node for node in context_dict.values() if isinstance(node, ContextNode) and node.type == "function"]
    all_func_content = ["".join([context_dict[child].content for child in node.children]) for node in all_func]
    
    tokenizer = get_tokenizer(config)
    
    tokenized_corpus = [tokenizer.tokenize(content) for content in all_func_content]
    
    BM25_model = BM25Okapi(tokenized_corpus)
    
    scores = BM25_model.get_scores(tokenizer.tokenize(input_string))
    
    top_index = np.argsort(scores)[::-1]
    
    for index in top_index[:10]:
        print(scores[index])
        print(all_func[index].namespace)
        print(all_func_content[index])
    
    print("*"*20)
    depen_list = [value for value_list in data["dependency"].values() for value in value_list]
    
    for depen in depen_list:
        depen_namespace = update_namespace(context_dict, depen)
        if depen_namespace not in context_dict:
            continue
        print("*"*20)
        all_children = context_dict[depen_namespace].children
        for idx, node in enumerate(all_func):
            if node.namespace == depen_namespace:
                print(scores[idx])
                print(node.namespace)
                print(all_func_content[idx])
                for top, pos in enumerate(top_index):
                    if pos == idx:
                        print("top: ", top)
    
    pass
    
    

if __name__ == "__main__":
    update_dataset_bodypos()
    
    
    