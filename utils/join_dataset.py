import os
import json

if __name__ == "__main__":
    # 加载数据集
    result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/DevEval/Qwen2.5-Coder-3B-Instruct"
    
    result_name = "result_st"
    
    datasets = []
    
    for i in range(0, 11):
        dataset_path = f"{result_path}/{result_name}_{i}.json"
        if os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                data = json.load(f)
                datasets.append(data)
    
    new_dataset = {}
    for dataset in datasets:
        for key, value in dataset.items():
            if key in new_dataset:
                new_dataset[key].extend(value)
            else:
                new_dataset[key] = value
    
    for key, value in new_dataset.items():
        print(key+" num: "+str(len(value)))
        for data in value:
            if "context" in data:
                data["context"] = list(set(data["context"]))
        new_dataset[key] = value
    
    new_dataset = {k: v for k, v in sorted(new_dataset.items(), key=lambda item: int(item[0]))}
    
    with open(f"{result_path}/{result_name}_all.json", "w") as f:
        json.dump(new_dataset, f, indent=4)
        
    
    
    
    
    
    
    