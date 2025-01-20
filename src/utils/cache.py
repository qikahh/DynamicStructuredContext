from typing import Dict, Any, Tuple
import torch
from collections import OrderedDict

class KVCache:
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        
    def add(self, key: str, value: Tuple[torch.Tensor, torch.Tensor]):
        """添加KV对到缓存"""
        if len(self.cache) >= self.max_cache_size:
            # 移除最早添加的项
            self.cache.popitem(last=False)
        
        self.cache[key] = value
        # 移动到最新位置
        self.cache.move_to_end(key)
        
    def get(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存的KV对"""
        if key in self.cache:
            # 更新访问顺序
            self.cache.move_to_end(key)
            return self.cache[key]
        return None 