import torch
import torch.nn as nn
from typing import Optional, Tuple

class HierarchicalAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                query: torch.Tensor,
                key_value_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        实现层次化注意力计算
        
        Args:
            query: 查询张量
            key_value_states: 可选的key-value状态
            attention_mask: 注意力掩码
            layer_idx: 当前transformer层的索引
            
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 根据层索引决定压缩率
        compression_rate = self._get_compression_rate(layer_idx)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key_value_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 压缩key-value cache
        if compression_rate > 1:
            attention_weights = self._compress_attention(attention_weights, compression_rate)
            key_value_states = self._compress_kv_cache(key_value_states, compression_rate)
            
        output = torch.matmul(attention_weights, key_value_states)
        
        return output, attention_weights
    
    def _get_compression_rate(self, layer_idx: int) -> float:
        """根据层索引确定压缩率"""
        # 高层使用更高的压缩率
        return 1.0 + (layer_idx / 12) * 3  # 示例：最高层压缩率为4 