import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Model, DynamicCache

import matplotlib.pyplot as plt

from .hierarchical_context import ContextNode

logger = logging.getLogger(__name__)
# 设置日志级别
logger.setLevel(logging.INFO)
# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# 添加处理器到logger
logger.addHandler(console_handler)
def visualize_tensor(tensor, title="Tensor Visualization", save_path="/home/qikahh/projects/Structured_Code_Context/visualize.png"):
    """
    可视化PyTorch张量
    
    Args:
        tensor (torch.Tensor): 要可视化的张量
        title (str): 图表标题
        save_path (str): 保存图片的路径,如果为None则显示图片
    """
    # 确保张量在CPU上并转换为numpy数组
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.detach().to(torch.float).numpy()
    
    # 创建新的图表
    plt.figure(figsize=(10, 8))
    
    # 如果是1D张量,显示折线图
    if len(data.shape) == 1:
        plt.plot(data)
        plt.colorbar()
    
    # 如果是2D或更高维张量,显示最后两维的热力图
    else:
        # 如果维度大于2,只取最后两维
        if len(data.shape) > 2:
            # 计算需要展平的维度数
            dims_to_flatten = len(data.shape) - 2
            # 展平前面的维度
            new_shape = (-1,) + data.shape[-2:]
            data = data.reshape(new_shape)
            # 只显示第一个切片
            data = data[0]
            
        plt.imshow(data, cmap='viridis', aspect='auto')
        plt.colorbar()
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



class HierarchicalModel:
    """
    层次化上下文的模型包装类,实现逐层深入的上下文搜索和生成
    """
    def __init__(self, model, lm_head, tokenizer):
        self.model = model
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = model.device
        self.max_context_length = 4096
        self.max_length = 2048
        self.max_layer = 64
        self.output_attentions = False
        
        # 采样参数
        self.top_p = 0.95
        self.temperature = 0.2
        pass
    
    def encode_init_hidden(self, input_ids):
        """
        编码初始hidden 即模型输入未经过第一次Attn计算前的hidden
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        # create position embeddings to be shared across the decoder layers
        return inputs_embeds
    
    def encode_by_layer(self, hidden_states, start_layer, end_layer):
        key_list, value_list, hidden_list = [], [], []
        # create position embeddings to be shared across the decoder layers
        position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            )
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids.unsqueeze(0))
        causal_mask = self.model._update_causal_mask(
            None, hidden_states, position_ids, None, False
        )
        
        for layer_id in range(start_layer, end_layer+1):
            decoder_layer =  self.model.layers[layer_id]
            past_key_value = DynamicCache()
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids.unsqueeze(0),
                    use_cache = True,
                    past_key_value = past_key_value,
                    output_attentions=self.output_attentions,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]
            hidden_list.append(layer_outputs[0])
            key_value_cache = layer_outputs[2 if self.output_attentions else 1]
            key_list.append(key_value_cache.key_cache[layer_id])
            value_list.append(key_value_cache.value_cache[layer_id])
        return key_list, value_list, hidden_list
        
    def get_node_kv(self, node_list: list[ContextNode], layer_idx: int):
        """
        获取节点的KV向量对 计算时将node_list的内容拼接成一个整体进行编码
        
        如果节点已有向量缓存且层数足够,直接返回缓存的向量
        如果节点已有向量缓存但层数不足,基于最后一层缓存继续计算到目标层
        如果节点没有向量缓存,使用模型从头计算到目标层
        
        参数:
            node_list: 上下文节点列表
            layer_idx: 目标模型层索引
            
        返回:
            self_k: 节点key向量列表
            self_v: 节点value向量列表
            node_list: 原始节点列表
        """
        self_k, self_v = [], []
        node_list = [node for node in node_list if len(node.content)>0]
        # 获取所有节点都已经编码的最深层
        enc_layer_idx = self.max_layer
        for node in node_list:
            if node.vectors is not None:
                enc_layer_idx = min(enc_layer_idx, len(node.vectors)-1)
            else:
                enc_layer_idx = -1
        # 构建节点向量
        if enc_layer_idx >= layer_idx:
            # 使用缓存的向量
            now_pos = 0
            for node in node_list:
                now_pos += node.length
                self_k.append(node.vectors[layer_idx][0])
                self_v.append(node.vectors[layer_idx][1])
        elif 0 < enc_layer_idx < layer_idx:
            # 如果节点已经存在向量 但是长度小于当前层索引 则基于缓存的最后层向量继续计算获取到当前层
            # 首先将所有节点enc_layer_idx层的hidden拼接为一个序列
            hidden_seq = []
            hidden_length = []
            for node in node_list:
                hidden_seq.append(node.vectors[enc_layer_idx][-1])  # 获取每个节点最后缓存层的hidden state
                hidden_length.append(node.vectors[enc_layer_idx][-1].shape[1]) # 获取每个节点特征的长度
            last_hidden = torch.cat(hidden_seq, dim=1).to(self.device)  # 在序列维度上拼接
            
            # 基于最后一层缓存的向量继续计算到当前层
            key_list, value_list, hidden_list = self.encode_by_layer(last_hidden, enc_layer_idx+1, layer_idx)
            hidden_start = 0
            hidden_end = 0
            for i, node in enumerate(node_list):
                hidden_end += node.length
                for j in range(len(node.vectors), layer_idx+1):
                    hidden_ids = j-enc_layer_idx-1
                    node.vectors.append((key_list[hidden_ids][:,:,hidden_start:hidden_end].detach().cpu(), value_list[hidden_ids][:,:,hidden_start:hidden_end].detach().cpu(), hidden_list[hidden_ids][:,hidden_start:hidden_end].detach().cpu()))
                self_k.append(node.vectors[layer_idx][0])
                self_v.append(node.vectors[layer_idx][1])
                hidden_start = hidden_end
        else:
            # 如果节点不存在向量 则使用模型编码到当前层
            input_ids = []
            for node in node_list:
                inputs = self.tokenizer(node.content, return_tensors="pt").to(self.device)
                input_ids.append(inputs["input_ids"])
                node.length = inputs["input_ids"].shape[1]
            input_ids = torch.cat(input_ids, dim=1)
            init_hidden = self.encode_init_hidden(input_ids)
            key_list, value_list, hidden_list = self.encode_by_layer(init_hidden, 0, layer_idx)
            # 将新计算的向量按层添加到缓存中
            hidden_start = 0
            hidden_end = 0
            for i, node in enumerate(node_list):
                hidden_end += node.length
                for j in range(0, layer_idx+1):
                    node.vectors.append((key_list[j][:,:,hidden_start:hidden_end].detach().cpu(), value_list[j][:,:,hidden_start:hidden_end].detach().cpu(), hidden_list[j][:,hidden_start:hidden_end].detach().cpu()))
                    node.vectors_pos = hidden_start
                self_k.append(node.vectors[layer_idx][0])
                self_v.append(node.vectors[layer_idx][1])
                hidden_start = hidden_end
        
        return self_k, self_v, node_list
        
    def encode_nodeseq(self, nodes: List[str], layer_idx: int, context_dict, now_pos) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将上下文节点序列编码为KV对 此函数主要用于编码一个父节点下的兄弟节点序列 将它们拼接成一个整体进行kv计算
        一个节点的KV对由两部分组成 分别是其本身内容的向量 self_k, self_v 以及其子节点分别的内容平均向量 children_k, children_v
        其本身内容向量由和nodes列表其他内容拼接后计算得到 子节点内容向量由其所有子节点内容拼接后计算得到
        如果节点已经存在kv直接使用 否则用模型编码并缓存回上下文节点字典context_dict中
        TODO 现在now_pos未发挥作用 所有编码都是从0位置开始 需要将编码后的结果旋转到now_pos位置
        参数:
            nodes: 上下文节点列表, 元素为namespace
            layer_idx: 目标模型层索引
            context_dict: 上下文节点字典
        """
        all_keys = []
        all_values = []
        node_size = []
        
        node_list = []
        for node_ns in nodes:
            if node_ns not in context_dict:
                logger.warning(f"节点{node_ns}不存在, 将使用空字符串代替")
            else:
                node = context_dict[node_ns]
                if node.content is None:
                    logger.warning(f"节点{node_ns}不存在内容, 开始创建")
                    node.content = "# "+ node.type + ":"+ node.namespace.split('.', 1)[1].replace(".", os.sep) + "\n"
                    context_dict[node_ns] = node
                node_list.append(node)
        self_k, self_v, node_list = self.get_node_kv(node_list, layer_idx) # 节点本身内容对应的向量
        now_pos = sum([node.length for node in node_list])
        for i, node in enumerate(node_list):
            context_dict[node.namespace] = node

            # 构建子节点向量
            children_k, children_v = [], [] # 子节点内容平均向量
            children_list = []
            for child_ns in node.children:
                if child_ns not in context_dict:
                    logger.warning(f"节点{child_ns}不存在, 将使用空字符串代替")
                else:
                    child = context_dict[child_ns]
                    if child.content is None:
                        logger.warning(f"节点{child_ns}不存在内容, 开始创建")
                        child.content = "# "+ child.type + ":"+ child.namespace + "\n"
                        context_dict[child_ns] = child
                    if len(child.content):
                        children_list.append(child)
            child_k, child_v, children_list = self.get_node_kv(children_list, layer_idx) # 子节点内容对应的向量
            children_length = 0
            for j, child in enumerate(children_list):
                children_k.append(self.shift_pos(child_k[j], now_pos).mean(dim=2).unsqueeze(2))
                children_v.append(child_v[j].mean(dim=2).unsqueeze(2))
                context_dict[child.namespace] = child
            all_keys.append(torch.cat([self_k[i]]+children_k, dim=2))
            all_values.append(torch.cat([self_v[i]]+children_v, dim=2))
            node_size.append(all_keys[-1].shape[2])
                     
        return torch.cat(all_keys, dim=2), torch.cat(all_values, dim=2), node_size
    
    def select_high_attention_nodes(self, nodes_ns: List[str], attn_scores: torch.Tensor, context_dict: Dict, min_num=1) -> List[ContextNode]:
        """
        根据注意力分数筛选高注意力节点
        """
        nodes = [context_dict[ns] for ns in nodes_ns]
        node_size = [node.length+len(node.children) for node in nodes]
        node_attn = torch.zeros(len(nodes), dtype=torch.float32)
        now_pos = 0
        for i, node in enumerate(nodes):
            node_scores = attn_scores[0, :, :, now_pos:now_pos+node_size[i]]
            now_pos += node_size[i]
            node_attn[i] = node_scores[:,-1,:].max(dim=-1)[0].max(dim=-1)[0]
        
        # 选择前30%最高注意力的节点
        selected_nodes = []
        _, indices = torch.sort(node_attn, descending=True)
        top_k = max(1, min_num, int(len(nodes) * 0.3))  # 至少选择min_num个节点
        selected_indices = indices[:top_k]
        for idx in selected_indices:
            selected_nodes.append(nodes[idx])
        
        return selected_nodes
        
            
    def collect_children(self, nodes: List[ContextNode], context_dict: Dict) -> List[ContextNode]:
        """收集节点的所有子节点 如果不存在子节点则继续保留当前节点 按文件名以及父节点内子节点列表顺序排序"""
        nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.byte_begin))
        children = []
        for node in nodes:
            if len(node.children) == 0:
                children.append(node)
            if len(node.children) != 0:
                for child_ns in node.children:
                    children.append(context_dict[child_ns])
                
        return children

    def cluster_brothers(self, nodes: List[ContextNode]) -> List[List[str]]:
        """
        将节点列表按照兄弟节点进行聚类
        """
        node_parts = []
        now_part = []
        father_list = {}
        for node in nodes:
            if node.parent not in father_list:
                father_list[node.parent] = len(node_parts)
                if len(node.content):
                    node_parts.append([node])
                else:
                    qika = 1
            else:
                if len(node.content):
                    node_parts[father_list[node.parent]].append(node)
                else:
                    qika = 1
                
        return node_parts
    @staticmethod
    def remove_after_target(node_list, target_node):
        """
        删除节点列表中的目标节点以及文件内在目标节点之后的节点 并且将剩余节点按文件外节点、同文件节点 进行排序
        
        参数:
            node_list - 节点列表
            target_node - 目标节点
        返回:
            过滤后的节点列表
        """
        # 获取目标节点所在文件
        target_file = target_node.file_path
        # 过滤后的节点列表
        out_file_nodes = []
        in_file_nodes = []
        
        # 遍历节点列表
        for node in node_list:
            # 如果节点不在目标文件中,直接保留
            if node.file_path != target_file:
                out_file_nodes.append(node)
                continue
                
            # 如果节点在目标文件中且在目标节点之前,保留该节点
            if node.byte_begin < target_node.byte_begin:
                in_file_nodes.append(node)
        
        out_file_nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.byte_begin))
        in_file_nodes.sort(key=lambda x: x.byte_begin)
        
        return out_file_nodes+in_file_nodes
        
    def shift_pos(self, key:torch.Tensor, pos:int):
        """
        基于ROPE位置编码对输入向量进行平移
        输入:
            key: torch.Tensor(batch, head_num, seq_len, head_dim)
            value: torch.Tensor(batch, head_num, seq_len, head_dim)
            pos: int
        输出:
            key: torch.Tensor(batch, head_num, seq_len, head_dim)
            value: torch.Tensor(batch, head_num, seq_len, head_dim)
        """
        length = key.shape[2]
        if pos >= 0:
            # 由于是统一移动相同位置，因此position_ids值全部为pos
            position_ids = torch.full(
                (1, length), pos, device=key.device
            )
            position_embeddings = self.model.rotary_emb(key, position_ids)
            cos, sin = position_embeddings
        elif pos < 0:
            position_ids = torch.full(
                (1, length), -pos, device=key.device
            )
            position_embeddings = self.model.rotary_emb(key, position_ids)
            cos, sin = position_embeddings
            sin = -sin
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        return k_embed
            
    def sample_next_token(self, logits: torch.Tensor):
        """
        采样输出 基于top-p和temperature超参数
        """
        # 对logits应用temperature
        logits = logits / self.temperature
        
        # 计算softmax概率
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 按概率从大到小排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的位置
        mask = cumsum_probs > self.top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        
        # 将不在top_p范围内的概率置为0
        sorted_probs.masked_fill_(mask, 0.0)
        
        # 重新归一化概率
        sorted_probs_sum = torch.sum(sorted_probs, dim=-1, keepdim=True)
        sorted_probs = sorted_probs / sorted_probs_sum
        
        # 按概率采样
        idx = torch.multinomial(sorted_probs, num_samples=1)
        
        # 获取采样的token id
        next_token = sorted_indices[0, idx[0]]
        
        return next_token
        
    def generate_step(self, 
                        target_namespace, 
                        input_ids: torch.Tensor, 
                        past_key_values, 
                        context_dict: Dict, 
                        init_context_nodes: List[str]
                    ) -> Tuple[torch.Tensor, List[ContextNode]]:
        """
        执行一步生成,包含逐层深入的上下文搜索
        返回生成的token_id和下一步使用的上下文节点
        """
        curr_context = [context_dict[ns] for ns in init_context_nodes]
        layer_outputs = []
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        # 编码位置信息和掩码 输入的起始位置为0
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        
        # 逐层处理
        for layer_idx in range(len(self.model.layers)):
            # 去除无效上下文
            curr_context = [node for node in curr_context if len(node.content) > 0]
            # 输出当前层使用的上下文namespace
            logger.info(f"Layer {layer_idx} using context: {[node.namespace for node in curr_context]}")
            # 编码当前层上下文节点
            node_parts = self.cluster_brothers(curr_context)
            context_k = []
            context_v = []
            now_pos = 0
            for node_list in node_parts[-1::-1]:
                node_list = [node.namespace for node in node_list]
                k, v, node_size = self.encode_nodeseq(node_list, layer_idx, context_dict, 0)
                now_pos -= sum(node_size)
                # 所有上下文的位置从零开始向左延伸，故此段的起始位置now_pos为负数（上一个起始位置减去当前段的长度）
                k = self.shift_pos(k, now_pos)
                context_k.append(k)
                context_v.append(v)

            past_key_values:DynamicCache = DynamicCache()
            if len(context_k):
                context_k = torch.cat(context_k, dim=2).to(self.device)
                context_v = torch.cat(context_v, dim=2).to(self.device)
                past_key_values.update(context_k, context_v, layer_idx)
                past_key_values.update(context_k, context_v, 0)
            
                context_length = context_k.shape[2]
            
            # 编码当前层输入特征
            layer = self.model.layers[layer_idx]
            if layer_idx == 0:
                hidden_states = inputs_embeds
            else:
                hidden_states = layer_outputs[0]
            
            position_ids = cache_position.unsqueeze(0)
                
            causal_mask = self.model._update_causal_mask(
                None, inputs_embeds, cache_position+past_key_values.key_cache[layer_idx].shape[-2], past_key_values, output_attentions=True
            )
            
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            
                
            layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            
            if layer_idx%2 == 1:
                # 选择高注意力节点
                curr_context = [node.namespace for node in curr_context]
                attn_scores = layer_outputs[1]
                high_attn_nodes = self.select_high_attention_nodes(curr_context, attn_scores, context_dict, min_num=6)    
                # 收集子节点作为下一层上下文
                curr_context = self.collect_children(high_attn_nodes, context_dict)
                # 删去目标节点以及同文件内在目标节点之后的节点
                curr_context = self.remove_after_target(curr_context, context_dict[target_namespace])
            pass
            
        # 计算最终输出
        hidden_states = layer_outputs[0]
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # 使用类方法进行采样
        next_token = self.sample_next_token(logits[:, -1])
        
        return next_token, curr_context, context_dict, past_key_values

