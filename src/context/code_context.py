from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import torch
from pathlib import Path
import ast

class ContextLevel(Enum):
    FOLDER = 0
    FILE = 1
    CLASS = 2
    FUNCTION = 3
    LINE = 4
    TOKEN = 5

@dataclass
class CodeContext:
    id: str  # 唯一标识符
    level: ContextLevel
    content: str
    children: List['CodeContext']
    metadata: Dict[str, Any]
    parent: Optional['CodeContext'] = None
    
    def __post_init__(self):
        for child in self.children:
            child.parent = self

class ContextManager:
    def __init__(self):
        self.root = None
        self.current_level = None
        self.kv_cache = None  # 将在初始化时设置
        
    def build_context_tree(self, project_path: str):
        """构建项目的层次化上下文树"""
        project_path = Path(project_path)
        self.root = self._build_folder_context(project_path)
        
    def _build_folder_context(self, folder_path: Path) -> CodeContext:
        """构建文件夹级别的上下文"""
        children = []
        for item in folder_path.iterdir():
            if item.is_dir():
                children.append(self._build_folder_context(item))
            elif item.suffix == '.py':
                children.append(self._build_file_context(item))
                
        return CodeContext(
            id=str(folder_path),
            level=ContextLevel.FOLDER,
            content=folder_path.name,
            children=children,
            metadata={'path': str(folder_path)}
        )
        
    def _build_file_context(self, file_path: Path) -> CodeContext:
        """构建文件级别的上下文"""
        with open(file_path) as f:
            content = f.read()
            
        # 解析Python代码
        try:
            tree = ast.parse(content)
            children = self._parse_ast_nodes(tree)
        except:
            children = []
            
        return CodeContext(
            id=str(file_path),
            level=ContextLevel.FILE,
            content=content,
            children=children,
            metadata={'path': str(file_path)}
        )
        
    def _parse_ast_nodes(self, tree: ast.AST) -> List[CodeContext]:
        """解析AST节点，构建类和函数级别的上下文"""
        contexts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                contexts.append(self._build_class_context(node))
            elif isinstance(node, ast.FunctionDef):
                contexts.append(self._build_function_context(node))
        return contexts
        
    def get_context_at_level(self, level: ContextLevel) -> List[CodeContext]:
        """获取特定层级的所有上下文"""
        if self.root is None:
            return []
            
        contexts = []
        def collect_contexts(node: CodeContext):
            if node.level == level:
                contexts.append(node)
            for child in node.children:
                collect_contexts(child)
                
        collect_contexts(self.root)
        return contexts
        
    def expand_context(self, context: CodeContext, attention_idx: int):
        """基于注意力分数展开下一层上下文"""
        if not context.children or attention_idx >= len(context.children):
            return
            
        target_child = context.children[attention_idx]
        # 将展开的上下文添加到当前处理队列
        self.current_level = target_child.level 