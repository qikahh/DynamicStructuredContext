import ast
import os
import json
import pickle
import random
import torch

from tqdm import tqdm


import tree_sitter_python as tspython
from tree_sitter import Language, Parser
# 初始化tree-sitter解析器
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class ContextNode(object):
    """
    层次化上下文节点类，表示代码结构中的一个实体（文件夹、文件、类、函数、代码块）
    每个节点包含:
    - namespace: 命名空间，表示该节点在项目中的完整路径
    - name: 节点名称
    - type: 节点类型，可以是 folder/file/class/function/code
    - children: 子节点列表
    - parent: 父节点
    - content: 节点内容，对于代码块是具体代码，对于其他类型为对应的压缩信息
    - vectors: 节点向量，用于表示节点对于特定模型被编码后的的语义特征KV对 可以直接参与注意力计算
    """
    def __init__(self, namespace, name, type, content=None):
        self.namespace = namespace  # 节点的完整命名空间路径
        self.name = name  # 节点名称
        self.type = type  # 节点类型
        self.children = []  # 子节点namespace列表
        self.parent = None  # 父节点namespace
        
        # 代码对应的位置
        self.file_path = None
        self.file_ns = None
        self.byte_begin = None
        self.byte_end = None
        
        self.content = content  # 节点内容
        
        self.model = None # 编码所用的模型 
        self.length = 0 # 所占的token数量
        # 节点向量, 元素为tensor的unit类型，分别为（key，value，hidden），不同元素对应不同模型层
        # kv和hidden都是和节点的兄弟节点一起编码的
        self.vectors = [] 
        self.vectors_pos = [] # 计算节点向量时对应的起始位置
        self.children_vectors = [] # 子节点池化后的向量 为tensor的unit类型，分别为（key，value）， 不同元素对应不同模型层
        self.children_vectors_pos = [] # 计算节点向量时对应的起始位置
    
    def __str__(self) -> str:
        return f"{self.type}:{self.namespace} {len(self.children)}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 对于无法pickle的属性，可以在这里删除或处理
        return state
    def __getitem__(self, key):
        return getattr(self, key)

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    

def build_blocks(code, parent_node, begin_pos, cut = True):
    """
    将一段代码按空行分割成多个代码块,每个代码块创建一个ContextNode节点
    并将这些节点作为parent_node的子节点
        code: str - 需要处理的代码文本
        parent_node: ContextNode - 父节点，这些代码块将作为其子节点
        begin_pos: int - 代码块的起始位置(按字节记录)
        cut = True则将代码块分割为多个独立代码块 否则整体作为一个块
    """
    # 将代码按空行分割成块
    all_nodes = []
    if cut:
        # 使用正则表达式进行分割，保留分隔符
        import re
        blocks = re.split(r'(\n\n+)', code)
        # 合并相邻的块和分隔符
        merged_blocks = []

        for i in range(0, len(blocks), 2):
            current_block = blocks[i]
            if i + 1 < len(blocks):
                current_block += blocks[i + 1]
            if len(merged_blocks) != 0 and merged_blocks[-1].strip('\n') == "":
                merged_blocks[-1] += current_block
            else:
                merged_blocks.append(current_block)
        # 只有当最后一个块完全是空白字符时才删除
        if len(merged_blocks)>1 and len(merged_blocks[-1])==0:
            merged_blocks.pop()
        blocks = merged_blocks
    else:
        blocks = [code]
    current_pos = begin_pos
    
    for i, block in enumerate(blocks):
    
        # 计算当前块的字节数
        block_size = len(block)
        
        # 创建代码块节点
        block_node = ContextNode(
            namespace=parent_node.namespace + f".block_{current_pos}",
            name=f"block_{current_pos}",
            type="code",
            content=block
        )
        block_node.file_path = parent_node.file_path
        block_node.file_ns = parent_node.file_ns
        block_node.byte_begin = current_pos
        # 更新当前行号
        current_pos += block_size
        block_node.byte_end = current_pos
        
        # 设置父子关系
        block_node.parent = parent_node.namespace
        parent_node.children.append(block_node.namespace)
        
        all_nodes.append(block_node)
        
    return all_nodes

def build_context_tree(all_code, root, parent_node):
    """
    对于tree-sitter解析器解析出的语法树 构建上下文树
    """
    
    if parent_node.namespace == "sslyze.sslyze.cli.command_line_parser.CommandLineParser":
        qika = 1
    all_nodes = []
    #定义此block的开头
    indep_begin = root.start_byte
    for child in root.children:
        # 处理类定义
        if child.type == "class_definition":
            # 处理类前的独立代码块
            indep_code = all_code[indep_begin:child.start_byte]
            if len(indep_code):
                all_nodes += build_blocks(indep_code, parent_node, begin_pos = indep_begin)
            
            #更新上一个类和函数的结尾位置
            indep_begin = child.end_byte
            
            class_name = child.children[1].text.decode()  # 获取类名
            # 获取类头代码 - 从类定义开始到类体之前的部分
            class_header_end = child.children[-1].start_byte  # 类体开始前的位置
            class_header = all_code[child.start_byte:class_header_end]
            
            class_node = ContextNode(
                namespace=parent_node.namespace + "." + class_name,
                name=class_name,
                type="class",
                content=class_header+"..."
            )
            class_node.file_path = parent_node.file_path
            class_node.file_ns = parent_node.file_ns
            class_node.byte_begin = child.start_byte
            class_node.byte_end = child.end_byte
            class_node.parent = parent_node.namespace
            # 将类头作为独立代码块加入类节点的子节点
            class_head = build_blocks(class_header, class_node, begin_pos = child.start_byte, cut=False)
            # 将类体中的代码块加入类节点的子节点
            class_childs = build_class(all_code, child, class_node)
            parent_node.children.append(class_node.namespace)
            
            all_nodes.append(class_node)
            all_nodes += class_head
            all_nodes += class_childs
            
        # 处理函数定义 
        elif child.type == "function_definition":
            #处理函数前的独立代码块
            indep_code = all_code[indep_begin:child.start_byte]
            if len(indep_code):
                all_nodes += build_blocks(indep_code, parent_node, begin_pos = indep_begin)
            # 更新上一个类和函数的结尾位置
            indep_begin = child.end_byte
            
            func_name = child.children[1].text.decode()  # 获取函数名
            # 获取函数头代码 - 从函数定义开始到函数体之前的部分
            func_header_end = child.children[-1].start_byte  # 函数体开始前的位置
            func_header = all_code[child.start_byte:func_header_end]
            
            
            func_node = ContextNode(
                namespace=parent_node.namespace + "." + func_name, 
                name=func_name,
                type="function",
                content=func_header+"..."
            )
            func_node.file_path = parent_node.file_path
            func_node.file_ns = parent_node.file_ns
            func_node.byte_begin = child.start_byte
            func_node.byte_end = child.end_byte
            func_node.parent = parent_node.namespace
            
            # 将函数头作为独立代码块加入函数节点的子节点
            func_head = build_blocks(func_header, func_node, begin_pos = child.start_byte, cut=False)
            # 将函数体中的代码块加入函数节点的子节点
            func_nodes = bulid_func(all_code, child, func_node)
            parent_node.children.append(func_node.namespace)
            
            all_nodes.append(func_node)
            all_nodes += func_head
            all_nodes += func_nodes
        
        # 处理'decorated_definition'
        elif child.type == "decorated_definition":
            # 处理函数前的独立代码块
            indep_code = all_code[indep_begin:child.start_byte]
            if len(indep_code):
                all_nodes += build_blocks(indep_code, parent_node, begin_pos = indep_begin)
            # 更新上一个类和函数的结尾位置
            indep_begin = child.end_byte
            
            # 获取装饰器后的实际定义节点(函数或类)
            definition = child.children[-1]
            
            if definition.type == "function_definition":
                func_name = definition.children[1].text.decode()  # 获取函数名
                # 获取函数头代码(包含装饰器) - 从装饰器开始到函数体之前的部分
                func_header_end = definition.children[-1].start_byte  # 函数体开始前的位置
                func_header = all_code[child.start_byte:func_header_end]
                
                func_node = ContextNode(
                    namespace=parent_node.namespace + "." + func_name,
                    name=func_name, 
                    type="function",
                    content=func_header+"..."
                )
                func_node.file_path = parent_node.file_path
                func_node.file_ns = parent_node.file_ns
                func_node.byte_begin = child.start_byte
                func_node.byte_end = child.end_byte
                func_node.parent = parent_node.namespace
                
                # 将函数头(含装饰器)作为独立代码块加入函数节点的子节点
                func_head = build_blocks(func_header, func_node, begin_pos = child.start_byte, cut=False)
                # 将函数体中的代码块加入函数节点的子节点
                func_nodes = bulid_func(all_code, definition, func_node)
                parent_node.children.append(func_node.namespace)
                
                all_nodes.append(func_node)
                all_nodes += func_head
                all_nodes += func_nodes
                
            elif definition.type == "class_definition":
                class_name = definition.children[1].text.decode()  # 获取类名
                # 获取类头代码(包含装饰器) - 从装饰器开始到类体之前的部分
                class_header_end = definition.children[-1].start_byte  # 类体开始前的位置
                class_header = all_code[child.start_byte:class_header_end]
                
                class_node = ContextNode(
                    namespace=parent_node.namespace + "." + class_name,
                    name=class_name,
                    type="class", 
                    content=class_header+"..."
                )
                class_node.file_path = parent_node.file_path
                class_node.file_ns = parent_node.file_ns
                class_node.byte_begin = child.start_byte
                class_node.byte_end = child.end_byte
                class_node.parent = parent_node.namespace
                
                # 将类头(含装饰器)作为独立代码块加入类节点的子节点
                class_head = build_blocks(class_header, class_node, begin_pos = child.start_byte, cut=False)
                # 将类体中的代码块加入类节点的子节点
                class_childs = build_class(all_code, definition, class_node)
                parent_node.children.append(class_node.namespace)
                
                all_nodes.append(class_node)
                all_nodes += class_head
                all_nodes += class_childs
            
    #处理文件尾的独立代码块
    indep_code = all_code[indep_begin:parent_node.byte_end]
    all_nodes += build_blocks(indep_code, parent_node, begin_pos = indep_begin, cut = False)
    return all_nodes
    
def bulid_func(all_code, func_root, parent_node):
    """
    对于tree-sitter解析出的funcDef节点 构建上下文树
    """
    func_body = func_root.child_by_field_name("body")
    return build_context_tree(all_code, func_body, parent_node)
    
def build_class(all_code, class_root, parent_node):
    """
    对于tree-sitter解析器解析出的classDef节点 构建上下文树
    """
    class_body = class_root.child_by_field_name("body")
    return build_context_tree(all_code, class_body, parent_node)
    
def build_file(file_path, file_node):
    """
    递归构建上下文树 此函数用于处理Python文件的子内容
    将Python代码文件的内容切分 类或函数直接切分出来 其他block按空行切分
    file_path: Python文件路径
    file_node: Python文件节点
    """
    if file_node.namespace == "sslyze.tasks":
        qika = 1
    
    with open(file_path, "r") as f:
        code = f.read()
        
    # 使用TreeSitter库解析Python代码
    tree = parser.parse(code.encode())
    # 遍历语法树,提取类和函数定义
    root = tree.root_node
    
    
    file_node.byte_begin = root.start_byte
    file_node.byte_end = root.end_byte
    
    return build_context_tree(code, root, file_node)
        

def build_folder(curr_path, curr_node):
    """
    递归构建上下文树 此函数用于处理文件夹的子内容
    先处理全部文件，再处理子文件夹
    curr_path: 当前文件夹路径
    curr_node: 当前节点
    """
    all_nodes = []
    tqdmbar = tqdm(os.listdir(curr_path), desc=f"Building folder {curr_node.namespace}")

    for item in tqdmbar:
        item_path = os.path.join(curr_path, item) 
        # 先处理所有Python文件
        if item.endswith(".py"):
            # 处理Python文件
            file_node = ContextNode(
                namespace=curr_node.namespace + "." + item[:-3],
                name=item[:-3], 
                type="file",
            )
            file_node.content = "# file: " + file_node.namespace.replace(".", "/")+".py\n"
            file_node.file_path = item_path
            file_node.file_ns = file_node.namespace
            file_node.parent = curr_node.namespace
            file_head_node = ContextNode(
                namespace=file_node.namespace+"._head",
                name="_head",
                type="code",
                content= "# file: " + file_node.namespace.replace(".", "/")+".py\n"
            )
            file_head_node.file_path = file_node.file_path
            file_head_node.file_ns = file_node.namespace
            file_head_node.parent = file_node.namespace
            file_head_node.byte_begin = -1
            file_head_node.byte_end = 0
            
            file_node.children.append(file_head_node.namespace)
            file_children = build_file(item_path, file_node)
            
            curr_node.children.append(file_node.namespace)
            all_nodes.append(file_node)
            all_nodes.append(file_head_node)
            all_nodes.extend(file_children)
            
        
    for item in os.listdir(curr_path):
        item_path = os.path.join(curr_path, item)
        if os.path.isdir(item_path) and item[0] != "." and item != "tests":
            # 处理文件夹
            folder_node = ContextNode(
                namespace=curr_node.namespace + "." + item,
                name=item,
                type="folder",
            )
            folder_node.content = "# folder: " + folder_node.namespace.replace(".", "/")+"\n"
            folder_node.parent = curr_node.namespace
            folder_children = build_folder(item_path, folder_node)
            if len(folder_children):
                curr_node.children.append(folder_node.namespace)
                all_nodes.append(folder_node)
                all_nodes.extend(folder_children)
    
    return all_nodes
def get_tooleval(data_path, source_code_path, result_path):
    """
    将ToolEval中的单条Python数据的上下文组织成字典形式
    具体来说 data_path路径为data.jsonl数据文件 source_code_path文件夹中有一系列项目级上下文代码
    需要将data.jsonl中每条数据对应的项目级上下文组织为一系列ContextNode实例
    包括 文件夹、文件、类、函数、代码块 五个层级
    对于代码文件的分析使用TreeSitter库
    """    
    # 读取jsonl文件
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
            

    # 构建上下文树
    context_trees = []
    for data in data_list:
        # 获取项目路径
        project_path = os.path.join(source_code_path, data["project_path"])
        
        # 查看项目是否已被上下文化
        # if os.path.exists(os.path.join(result_path, data["project_path"])):
        #    continue
        
        # 创建根节点(项目文件夹)
        root_node = ContextNode(
            namespace=data["project_path"].split("/")[-1],
            name=data["project_path"].split("/")[-1],
            type="repository"
        )
        folder_nodes = build_folder(project_path, root_node)
        
        all_nodes = [root_node] + folder_nodes
        
        # 将上下文树转换为字典，key为namespace，value为节点
        context_trees_dict = {}
        for node in all_nodes:
            context_trees_dict[node.namespace] = node
        context_trees_dict[''] = all_nodes[0].namespace
        
        # 保存上下文树 内部存在Torch Tensor类型数据，因此需要使用pickle保存
        save_path = os.path.join(result_path, data["project_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(context_trees_dict, os.path.join(save_path, "ContextTree.pth"))
        
        context_trees.append(context_trees_dict)
        
    return context_trees

def DFS2leaf(context_dict, node: ContextNode, cut_byte = -1, level = -1):
    """
    深搜找到输入节点的子节点 需要位置在cut_byte之前 level为搜索的深度
    参数:
        node - 待搜索的节点
        cut_byte - 搜索的截止位置
        level - 搜索的深度 -1表示搜索到叶子
    """
    all_nodes = []
    
    # 如果当前节点是叶子节点(没有子节点)或者是代码块节点,则加入结果列表
    if len(node.children) == 0 or node.type == "code":
        # 如果指定了cut_byte且当前节点有byte_end属性
        # 则只保留完全在cut_byte之前的节点
        if cut_byte != -1 and node.byte_end is not None:
            if node.byte_end < cut_byte:
                all_nodes.append(node.namespace)
        else:
            all_nodes.append(node.namespace)
        return all_nodes
        
    # 递归处理所有子节点
    if level == 0:
        if node.byte_begin < cut_byte:
            all_nodes.append(node.namespace)
        return all_nodes
    if level != -1:
        level -= 1
    for child_namespace in node.children:
        child_node = context_dict[child_namespace]
        all_nodes.extend(DFS2leaf(context_dict, child_node, cut_byte, level))
        
    return all_nodes

def initial_input(context_dict, target_namespace, type = "file"):
    """
    根据待生成代码的目标命名空间 从完整项目中筛选输入 此部分直接进入代码token级别
    筛选根据type确定范围 默认为同文件上文
    
    参数：
        context_dict - 节点字典
        target_namespace - 目标节点的命名空间
    """
    # 获取目标节点
    target_node = context_dict[target_namespace]
    
    # 获取目标节点所在文件的节点
    file_namespace = target_node.file_ns
    file_node = context_dict[file_namespace]
    
    # 获取目标节点在文件中的位置
    target_byte_begin = target_node.byte_begin
    
    # 获取文件中目标节点之前的所有节点
    input_nodes = DFS2leaf(context_dict, file_node, target_byte_begin, -1)
    
    return input_nodes
def initial_context(context_dict, target_namespace):
    """
    根据待生成代码的目标命名空间，从完整上下文树中筛选初始节点子集
    筛选核心为相关性 设计如下：
        1. 与目标在同一类的，直接筛选到最细节的代码节点
        2. 与目标不在同一类的但在同一文件内的，筛选到比文件低一级别的节点（如独立的类和函数、代码块）
        3. 与目标不在同一类的，不在同一文件内的，则从下往上逐渐检索当前节点的兄弟
    
    参数：
        context_dict - 节点字典
        target_namespace - 目标节点的命名空间
    """
    
    target_node = context_dict[target_namespace]
    
    in_class_nodes = [] # 同一类中的节点
    in_file_nodes = [] # 同一文件内的节点
    cross_file_nodes = [] # 不在同一文件的节点
    
    ancestors_node = None # 祖先节点，用来找到目标代码祖先中的类节点和文件节点
    cut_byte = 0 # 只保留目标节点前的代码，此变量用于筛选
    target_class = None # 目标代码所在的类节点
    target_file = None # 目标代码所在的文件节点
    if target_namespace in context_dict:
        ancestors_node = context_dict[target_namespace] 
        cut_byte = target_node.byte_begin
    
    while ancestors_node and ancestors_node.type == "function" and ancestors_node.parent != None:
        ancestors_node = context_dict[ancestors_node.parent]
    
    # 找到祖先中的最低类节点
    if ancestors_node.type == "class":
        in_class_nodes.extend(DFS2leaf(context_dict, ancestors_node, cut_byte))
        target_class = ancestors_node.namespace
    
    while ancestors_node and ancestors_node.type != "file":
        ancestors_node = context_dict[ancestors_node.parent]
    
    # 找到祖先中的文件节点
    if ancestors_node.type == "file":
        in_file_nodes.extend(DFS2leaf(context_dict, ancestors_node, cut_byte, level=1))
        target_file = ancestors_node.namespace
    
    cross_file_nodes = []
    # 不断向上找祖先的兄弟节点
    while ancestors_node and ancestors_node.parent != None:
        father = context_dict[ancestors_node.parent]
        cross_file_nodes = father.children+cross_file_nodes
        cross_file_nodes.remove(ancestors_node.namespace)
        ancestors_node = father
    
    """
    root_namespace = context_dict['']
    root_node = context_dict[root_namespace]
    cross_file_nodes = root_node.children
    """
    
    if target_class and target_class in in_file_nodes:
        in_file_nodes.remove(target_class)
    if target_file and target_file in cross_file_nodes:
        cross_file_nodes.remove(target_file)
        
    return {
        "in_class_nodes": in_class_nodes,
        "in_file_nodes": in_file_nodes,
        "cross_file_nodes": cross_file_nodes,
        "target_namespace": target_namespace,
        "target_class": target_class,
        "target_file": target_file,
        "cut_byte": cut_byte,
    }
        
    

if __name__ == "__main__":
    # 测试get_tooleval函数
    source_code_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/Source_Code"
    data_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/data.jsonl"
    result_path = "/home/qikahh/projects/Structured_Code_Context/Datasets/ToolEval/RepoContext"
    
    # 获取上下文树
    context_trees = get_tooleval(data_path, source_code_path, result_path)
    
    # 打印第一棵树的基本信息
    first_repo = context_trees[0]
    root = first_repo[first_repo['']]
    print(f"根节点命名空间: {root.namespace}")
    print(f"根节点名称: {root.name}")
    print(f"根节点类型: {root.type}")
    print(f"子节点数量: {len(root.children)}")
    
    # 遍历打印树的前几层结构
    def print_tree(node, level=0):
        print("  " * level + f"- {node.name} ({node.type})")
        if level < 3:  # 只打印前几层
            for child in node.children:
                #跳过block 节点
                if first_repo[child].type == "code":
                    continue
                print_tree(first_repo[child], level + 1)
                
    print("\n树结构:")
    print_tree(root)
    
    
    # 随机找一个func节点构建初始化上下文
    func_nodes = [node for node in first_repo.values() if (hasattr(node, "type") and node.type == "function")]
    pos = 0
    for i in range(10):
        func_node = func_nodes[pos]
        pos += 1
        while first_repo[func_node.parent].type != "class":
            func_node = func_nodes[pos] 
            pos += 1
        # 构建初始化上下文
        init_context = initial_context(first_repo, func_node.namespace)
        print("\n选中的函数节点:", func_node.namespace)
        print("\n初始化上下文:")
        print("同类内节点:", init_context["in_class_nodes"])
        print("同文件内节点:", init_context["in_file_nodes"]) 
        print("跨文件节点:", init_context["cross_file_nodes"])
        print("目标命名空间:", init_context["target_namespace"])
        print("目标类:", init_context["target_class"])
        print("目标文件:", init_context["target_file"])
        print("截断位置:", init_context["cut_byte"])
    pass
