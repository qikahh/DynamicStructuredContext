from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..context.code_context import ContextManager, ContextLevel

class HierarchicalQwen2Attention(Qwen2Attention):
    """扩展Qwen2的注意力层，支持层次化上下文处理"""
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.context_manager = None  # 将由外部设置
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # 根据层级获取合适的上下文表示
        context_level = self._get_context_level()
        context = self.context_manager.get_context_at_level(context_level)
        
        # 获取当前上下文的key-value表示
        if past_key_value is None:
            past_key_value = self._prepare_context_kv(context)
        
        # 调用原始的attention计算
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # 需要获取注意力权重
            use_cache=use_cache,
        )
        
        # 分析注意力分布，决定是否展开下一层
        if outputs[1] is not None:  # attention_weights
            self._handle_attention_weights(outputs[1], context)
            
        return outputs
        
    def _get_context_level(self) -> ContextLevel:
        """根据层索引确定上下文层级"""
        total_layers = self.config.num_hidden_layers
        ratio = self.layer_idx / total_layers
        
        if ratio < 0.25:
            return ContextLevel.FOLDER
        elif ratio < 0.5:
            return ContextLevel.FILE
        elif ratio < 0.75:
            return ContextLevel.CLASS
        else:
            return ContextLevel.TOKEN
            
    def _prepare_context_kv(self, context) -> Tuple[torch.Tensor]:
        """准备上下文的key-value表示"""
        # 检查缓存
        cache_key = f"layer_{self.layer_idx}_context_{context.id}"
        cached_kv = self.context_manager.kv_cache.get(cache_key)
        if cached_kv is not None:
            return cached_kv
            
        # 计算新的key-value
        context_hidden = self._encode_context(context)
        key = self.k_proj(context_hidden)
        value = self.v_proj(context_hidden)
        
        # 根据层级进行压缩
        compression_rate = self._get_compression_rate()
        if compression_rate > 1:
            key = self._compress_kv(key, compression_rate)
            value = self._compress_kv(value, compression_rate)
            
        kv_pair = (key, value)
        self.context_manager.kv_cache.add(cache_key, kv_pair)
        return kv_pair
        
    def _get_compression_rate(self) -> float:
        """获取当前层的KV压缩率"""
        base_rate = 1.0
        layer_ratio = self.layer_idx / self.config.num_hidden_layers
        return base_rate + layer_ratio * 3  # 最高层压缩率为4
        
    def _compress_kv(self, tensor: torch.Tensor, rate: float) -> torch.Tensor:
        """压缩key或value张量"""
        if rate <= 1:
            return tensor
            
        # 使用平均池化进行压缩
        orig_shape = tensor.shape
        pool = nn.AvgPool1d(kernel_size=int(rate), stride=int(rate))
        pooled = pool(tensor.transpose(1, 2)).transpose(1, 2)
        return pooled
        
    def _handle_attention_weights(self, attention_weights: torch.Tensor, context):
        """分析注意力权重，决定是否展开下一层上下文"""
        # 获取最大注意力分数
        max_attention = attention_weights.max().item()
        
        # 如果某个位置的注意力分数很高，考虑展开该位置的下一层上下文
        if max_attention > 0.5:  # 阈值可调
            max_idx = attention_weights.argmax().item()
            self.context_manager.expand_context(context, max_idx)

class HierarchicalQwen2ForCausalLM(Qwen2ForCausalLM):
    """扩展Qwen2模型，支持层次化上下文处理"""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.context_manager = ContextManager()
        
        # 替换原始的attention层
        for i, layer in enumerate(self.model.layers):
            hierarchical_attention = HierarchicalQwen2Attention(config, layer_idx=i)
            hierarchical_attention.context_manager = self.context_manager
            layer.self_attn = hierarchical_attention 