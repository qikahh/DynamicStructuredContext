import os
import pickle

from .dataset_utils import load_from_jsonl, traversal_files

def get_repo_info(database_root, repo_name):
    """
    从repo中抽取结构化上下文表示 组织成字典
    表中每个元素为项目中某一层级的内容(文件夹, py文件, 类, 函数, 代码行)
    """

        
    repo_database = []
    begin_path = os.path.join(database_root, repo_name)
    repo_database = traversal_files(database_root, repo_name, in_code=False)
    

def get_repoeval(root, level='api', length='2k'):
    """
    从RepoCoder项目中提取RepoEval数据集相关信息
    root: RepoCoder项目路径 以RepoCoder结尾
    level: api/line/function
    """
    repo_path = os.path.join(root, "repositories")
    
    assert level in ['api', 'line', 'function']
    dataname = "{}_level_completion_{}_context_codex.test".format(level,length)+".jsonl"
    datapath = os.path.join(root, "datasets", dataname)
    raw_dataset = load_from_jsonl(datapath)
    dataset = []
    repo_database = {}
    for raw_data in raw_dataset:
        file_path = raw_data['metadata']['fpath_tuple']
        data = {
            "task_id": raw_data['metadata']['task_id'],
            "fpath_tuple": file_path,
            "line_no": raw_data['metadata']['line_no'],
            "ground_truth": raw_data['metadata']['ground_truth'],
            "repo_root": repo_path,
        }
        dataset.append(data)
    return dataset






class RepoCache:
    def __init__(self, repo_root, repo_name, model_name, cache_root="/home/qikahh/projects/Structured_Code_Context/KVcache") -> None:
        self.repo_root = repo_root
        self.repo_name = repo_name
        self.model_name = model_name
        self.cache_root = cache_root
        self.database_path = os.path.join(cache_root, model_name, repo_name)
        self.database = None
        
        assert repo_root != None and model_name != None
        if not os.path.exists(self.database_path):
            repo_database, root_namespaces = traversal_files(repo_root, repo_name, in_code=False)
            self.save_database(repo_database, repo_name, self.database_path)
            self.database = repo_database
        else:
            self.database = self.load_database(self.database_path, self.repo_name)
    
    def load_database(self, path, name):
        """
        获取repo下namespace对应的实体的信息
        """
        with open(os.path.join(path, name+".pkl"), 'rb') as file:
            database = pickle.load(file)
        return database
        
    def save_database(self, database, database_name, path):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, database_name+".pkl")
        with open(full_path, "wb") as file:
            pickle.dump(database, file)
        
    def get_tree_siblings(self, namespace):
        siblings_list = []
        sub_namespace = []
        for name in namespace.split("."):
            sub_namespace.append(name)
            if not self.database or ".".join(sub_namespace) not in self.database:
                print("no database or no namespace")
                break
            siblings_list.append(".".join(sub_namespace))
            data = self.database[".".join(sub_namespace)]
            for sibling in data["children"]:
                if sibling not in namespace:
                    siblings_list.append(sibling)
        return siblings_list
    
    def get_children(self, namespace):
        if not self.database or namespace not in self.database:
            print("no database or no namespace")
        return self.database[namespace]["children"]
    
    def get_tokens(self, namespaces):
        """
        将文件级上下文和代码级上下文分别组织成文本序列
        """
        
    
    def get_kv_cache(self, namespace, layer):
        """
        获取repo下namespace对应的实体在模型中编码成KV对的表示 layer为提取第几层表示
        """
        pass

    
if __name__ == "__main__":
    root = "/home/qikahh/projects/Structured_Code_Context/RepoCoder"
    dataset = get_repoeval(root)
    for data in dataset:
        repo_root = data['repo_root']
        repo_name = data['fpath_tuple'][0]
        repo_cache = RepoCache(repo_root=repo_root, repo_name=repo_name, model_name="CodeLlama-7B")
        data_file_namespace = ".".join(data['fpath_tuple']).rsplit(".", 1)[0]
        
        tree_sblings = repo_cache.get_tree_siblings(data_file_namespace)
        
        qika = 1
        
        
    
    