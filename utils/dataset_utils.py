
import logging
import os
import json
import jsonlines

import torch
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

logging.basicConfig(level=logging.INFO)
def load_from_jsonl(file_name):
    with open(file_name, 'r') as f:
        data = jsonlines.Reader(f)
        data = list(data)
    return data

def traversal_node(rela_path, namespace, node):
    sub_database = {}
    if node.type == "class_definition" or node.type == "function_definition":
        name_node = node.children[1]
        class_name = name_node.text.decode()
        namespace = namespace + "." + class_name
        node_data = {
            "namespace": namespace,
            "name": class_name,
            "path": rela_path,
            "type": node.type.split("_")[0],
            "text": "",
            "children": []
        }
        class_body_node = node.children[-1]
        class_head_begin_line = node.start_point.row
        class_head_end_line = class_body_node.start_point.row
        node_data["begin_line"] = class_head_begin_line
        node_data["middle_line"] = class_head_end_line
        node_data["end_line"] = node.end_point.row
        
        sub_database[namespace] = node_data
        
        if class_body_node.type == "block":
            for child in class_body_node.children:
                children_database, children_namespaces = traversal_node(rela_path, namespace, child)
                sub_database.update(children_database)
                sub_database[namespace]["children"]+=children_namespaces
        else:
            children_database, children_namespaces = traversal_node(rela_path, namespace, class_body_node)
            sub_database.update(children_database)
            sub_database[namespace]["children"] += children_namespaces
        
        return sub_database, [namespace]
                
    elif node.type == "block":
        sub_namespace = []
        for child in node.children:
            children_database, children_namespaces = traversal_node(rela_path, namespace, child)
            sub_database.update(children_database)
            sub_namespace.extend(children_namespaces)
        return sub_database, sub_namespace

    else:
        line_number = node.start_point.row
        namespace = namespace + "." + str(line_number)
        node_data = {
            "namespace": namespace,
            "name": line_number,
            "path": rela_path,
            "type": "line",
            "text": "",
            "begin_line": line_number,
            "middle_line": node.end_point.row,
            "end_line": node.end_point.row,
            "children": []
        }
        sub_database[namespace] = node_data
        return sub_database, [namespace]
    
    return None, None
    
def traversal_code(database_root, rela_path, namespace = None):
    """
    从python文件中抽取结构化信息 分为 类 函数 行 三个级别
    """
    code_namespaces = []
    
    element_path = os.path.join(database_root, rela_path)
    if not os.path.exists(element_path):
        print(f"{element_path} not exists")
        return None, None
    if not (os.path.isfile(element_path) and element_path.endswith(".py")):
        print(f"{element_path} is not a python file")
        return None, None
    
    if not namespace:
        namespace = database_root.replace("/", ".").rsplit(".",1)[0]
    
    code_database = {}
    # 读取文件内容
    with open(element_path, 'r') as f:
        code_lines = f.readlines()
    
    # 使用tree-sitter解析
    tree = parser.parse(bytes("\n".join(code_lines), encoding='utf8'))
    
    for child in tree.root_node.children: 
        child_data, child_namespaces = traversal_node(rela_path, namespace, child)
        if child_data:
            code_database.update(child_data)
            code_namespaces.extend(child_namespaces)
    
    return code_database, code_namespaces
    
def traversal_files(database_root, rela_path, in_code, level=-1):
    """
    递归遍历此路径下所有子项目
    in_code控制是否解析到文件内容级别
    level控制解析多少层 -1表示无限制
    """
    if level == 0:
        return None, None
    elif level > 0:
        level -= 1
    now_database = {}
    root_namespaces = []
    element_path = os.path.join(database_root, rela_path)
    if not os.path.exists(element_path):
        print(f"{element_path} not exists")
        return None, None
    
    if os.path.isdir(element_path):
        namespace = rela_path.replace("/", ".")
        now_data = {
            "namespace": namespace,
            "name": rela_path.split("/")[-1],
            "path": rela_path,
            "type": "folder",
            "children": []
        }
        root_namespaces.append(namespace)
        now_database[now_data["namespace"]] = now_data
        for child in os.listdir(element_path):
            child_path = os.path.join(rela_path, child)
            child_database, child_namespaces = traversal_files(database_root, child_path, in_code, level)
            if child_database:
                now_database.update(child_database)
                now_database[now_data["namespace"]]["children"] += child_namespaces
        
        # 删除内部无python代码的文件夹
        if len(now_data["children"]) == 0:
            return None, None

        return now_database, root_namespaces
    
    elif os.path.isfile(element_path):
        if element_path.endswith(".py"):
            namespace = rela_path.replace("/", ".").rsplit(".",1)[0]
            name = rela_path.split("/")[-1].rsplit(".",1)[0]
            root_namespaces.append(namespace)
            now_data = {
                "namespace": namespace,
                "name": name,
                "path": rela_path,
                "type": "file",
                "children": []
            }
            now_database[now_data["namespace"]] = now_data
            if in_code:
                child_database, child_namespaces = traversal_code(database_root, rela_path, namespace)
                if child_database:
                    now_database.update(child_database)
                    now_database[now_data["namespace"]]["children"] += child_namespaces
        else:
            return None, None

    return now_database, root_namespaces


def make_input_string(prefix, instruct, head, requirement, dataset):
    if dataset in ["ToolEval", "DevEval"]:
        input_string = ""
        rspace_level = len(head) - len(head.rstrip()) - 1
        lspace_level = len(head) - len(head.lstrip())
        input_string += prefix
        input_string += " "*lspace_level + "\n"
        if instruct and len(instruct):
            input_string += " "*lspace_level + instruct
        input_string += head
        requirement_list = requirement["Functionality"].split("\n") + requirement["Arguments"].split("\n")
        requirement_string = " "*lspace_level+"    "+"\"\"\"\n" 
        for requirement in requirement_list:
            requirement_string += " "*lspace_level+"    "+f"{requirement}\n"
        requirement_string += " "*lspace_level+"    "+"\"\"\"\n"
        
        input_string += requirement_string
        
        # input_string += " "*lspace_level + "    " + "import"
    
    return input_string
        


