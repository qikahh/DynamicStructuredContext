import logging
logging.basicConfig(level=logging.DEBUG)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from typing import List, Dict, Tuple
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Model, DynamicCache

from .hierarchical_context import ContextNode


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
        self.max_context_length = 2048
        self.max_length = 1024
        self.max_layer = 64
        self.output_attentions = False
        
        # 层展开参数
        self.past_layers = 4
        self.max_node_length = 128
        self.max_crossfile_node_num = 256
        self.max_infile_node_num = 32
        
        # 采样参数
        self.top_p = 0.8
        self.top_k = 20
        self.temperature = 0.7
        pass
    
    def encode_init_hidden(self, input_ids):
        """
        编码初始hidden 即模型输入未经过第一次Attn计算前的hidden
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        # create position embeddings to be shared across the decoder layers
        return inputs_embeds
    
    def encode_by_layer(self, hidden_states, start_layer, end_layer, begin_pos = 0):
        key_list, value_list, hidden_list = [], [], []
        # create position embeddings to be shared across the decoder layers
        position_ids = torch.arange(
                begin_pos, begin_pos+hidden_states.shape[1], device=hidden_states.device
            )
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids.unsqueeze(0))
        causal_mask = self.model._update_causal_mask(
            None, hidden_states, position_ids, None, False
        )
        past_key_value = DynamicCache()
        for layer_id in range(start_layer, end_layer+1):
            decoder_layer =  self.model.layers[layer_id]
            
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
            
        key_list = [key.detach().cpu() for key in past_key_value.key_cache[start_layer:end_layer+1]]
        value_list = [value.detach().cpu() for value in past_key_value.value_cache[start_layer:end_layer+1]]
        
        del past_key_value, position_ids, position_embeddings, causal_mask
        return key_list, value_list, hidden_list

    def filter_ids(self, input_ids: torch.Tensor, max_length: int = 512) -> torch.Tensor:
        """
        过滤输入的token ids，确保长度不超过最大长度

        如果输入的token ids长度超过最大长度，则裁剪到最大长度

        参数:
            input_ids: 输入的token ids
            max_length: 最大允许的长度

        返回:
            过滤后的token ids
        """
        if input_ids.shape[1] > max_length:
            return input_ids[:, :max_length]
        return input_ids
    def get_node_kv(self, node_list: list[ContextNode], layer_idx: int, begin_pos: int = 0):
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
        now_length = 0
        real_list = []
        for node in node_list:
            if now_length >= self.max_context_length:
                break
            if len(node.content)>0:
                real_list.append(node)
                now_length += node.length
        node_list = real_list
        
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
                self_k.append(self.shift_pos(node.vectors[layer_idx][0], now_pos+begin_pos-node.vectors_pos[layer_idx]))
                self_v.append(node.vectors[layer_idx][1])
                now_pos += node.length
            pass
        elif 0 <= enc_layer_idx < layer_idx:
            # 如果节点已经存在向量 但是长度小于当前层索引 则基于缓存的最后层向量继续计算获取到当前层
            # 首先将所有节点enc_layer_idx层的hidden拼接为一个序列
            hidden_seq = []
            hidden_length = []
            now_pos = 0
            for node in node_list:
                hidden_seq.append(node.vectors[enc_layer_idx][-1])  # 获取每个节点最后缓存层的hidden state
                hidden_length.append(node.vectors[enc_layer_idx][-1].shape[1]) # 获取每个节点特征的长度
                now_pos += node.length
                if now_pos >= self.max_context_length:
                    break
            node_list = node_list[:len(hidden_length)]
            last_hidden = torch.cat(hidden_seq, dim=1).to(self.device)  # 在序列维度上拼接
            
            # 基于最后一层缓存的向量继续计算到当前层
            key_list, value_list, hidden_list = self.encode_by_layer(last_hidden, enc_layer_idx+1, layer_idx, begin_pos)
            
            hidden_start = 0
            hidden_end = 0
            for i, node in enumerate(node_list):
                hidden_end += node.length
                for j in range(len(node.vectors), layer_idx+1):
                    hidden_ids = j-enc_layer_idx-1
                    node.vectors.append((key_list[hidden_ids][:,:,hidden_start:hidden_end].detach().cpu(), value_list[hidden_ids][:,:,hidden_start:hidden_end].detach().cpu(), hidden_list[hidden_ids][:,hidden_start:hidden_end].detach().cpu()))
                    node.vectors_pos.append(hidden_start)
                self_k.append(node.vectors[layer_idx][0])
                self_v.append(node.vectors[layer_idx][1])
                
                hidden_start = hidden_end
            pass
        else:
            # 如果节点不存在向量 则使用模型编码到当前层
            input_ids = []
            now_pos = 0
            for node in node_list:
                inputs = self.tokenizer(node.content, return_tensors="pt").to(self.device)['input_ids']
                if inputs.shape[1] > self.max_node_length:
                    inputs = self.filter_ids(inputs, self.max_node_length)
                input_ids.append(inputs)
                node.length = inputs.shape[1]
                now_pos += node.length
                if now_pos >= self.max_context_length:
                    break
            node_num = len(input_ids)
            node_list = node_list[:node_num]
            input_ids = torch.cat(input_ids, dim=1)
            init_hidden = self.encode_init_hidden(input_ids)
            key_list, value_list, hidden_list = self.encode_by_layer(init_hidden, 0, layer_idx, begin_pos)
            
            # 将新计算的向量按层添加到缓存中
            hidden_start = 0
            hidden_end = 0
            for i, node in enumerate(node_list):
                hidden_end += node.length
                node.vectors = []
                node.vectors_pos = [hidden_start] * (layer_idx+1)
                for j in range(0, layer_idx+1):
                    node.vectors.append((key_list[j][:,:,hidden_start:hidden_end].detach().cpu(), value_list[j][:,:,hidden_start:hidden_end].detach().cpu(), hidden_list[j][:,hidden_start:hidden_end].detach().cpu()))
                self_k.append(node.vectors[layer_idx][0])
                self_v.append(node.vectors[layer_idx][1])
                hidden_start = hidden_end
            pass
            
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
                logging.warning(f"节点{node_ns}不存在, 将使用空字符串代替")
            else:
                node = context_dict[node_ns]
                if node.content is None:
                    logging.warning(f"节点{node_ns}不存在内容, 开始创建")
                    node.content = "# "+ node.type + ":"+ node.namespace.split('.', 1)[1].replace(".", os.sep) + "\n"
                    context_dict[node_ns] = node
                if node.type in ["folder", "file"] and node.length == 0:
                    for child_ns in node.children:
                        if context_dict[child_ns].type in ["folder", "file", "function", "class"]:
                            node.content += context_dict[child_ns].content
                    pass
                node_list.append(node)
        self_k, self_v, node_list = self.get_node_kv(node_list, layer_idx) # 节点本身内容对应的向量
        now_pos = sum([node.length for node in node_list])
        for i, node in enumerate(node_list):
            context_dict[node.namespace] = node

            # 构建子节点向量
            children_k, children_v = [], [] # 子节点内容平均向量
            """children_list = []
            for child_ns in node.children:
                if child_ns not in context_dict:
                    logging.warning(f"节点{child_ns}不存在, 将使用空字符串代替")
                else:
                    child = context_dict[child_ns]
                    if child.content is None:
                        logging.warning(f"节点{child_ns}不存在内容, 开始创建")
                        child.content = "# "+ child.type + ":"+ child.namespace + "\n"
                        context_dict[child_ns] = child
                    if len(child.content):
                        children_list.append(child)
            # child_k, child_v, children_list = self.get_node_kv(children_list, layer_idx, sum(node_size)) # 子节点内容对应的向量
            children_length = 0
            
            # 去除子节点信息
            children_list = []
            
            for j, child in enumerate(children_list):
                children_k.append(self.shift_pos(child_k[j].to(self.device), now_pos).mean(dim=2).unsqueeze(2))
                children_v.append(child_v[j].to(self.device).mean(dim=2).unsqueeze(2))
                context_dict[child.namespace] = child"""
            all_keys.append(torch.cat([self_k[i].to(self.device)]+children_k, dim=2))
            all_values.append(torch.cat([self_v[i].to(self.device)]+children_v, dim=2))
            node.length = all_keys[-1].shape[2]
                     
        return torch.cat(all_keys, dim=2), torch.cat(all_values, dim=2), node_list 
        
            
    def collect_children(self, nodes: List[ContextNode], context_dict: Dict) -> List[ContextNode]:
        """收集节点的所有子节点 如果不存在子节点则继续保留当前节点 按文件名以及父节点内子节点列表顺序排序"""
        nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line))
        children = []
        keep_nodes = []
        for node in nodes:
            if len(node.children) == 0:
                children.append(node)
            if len(node.children) != 0:
                for child_ns in node.children:
                    children.append(context_dict[child_ns])
                
        return children

    def cluster_brothers(self, nodes: List[ContextNode], target_node, context_dict:Dict) -> List[List[str]]:
        """
        将节点列表按照文件进行聚类 并填补缺失的语法上级 例如如果保留了一个类函数里的代码片段 则此类函数的head节点 此类的head节点 此文件的head节点都需要保留以保证文件内节点组织成的代码段结构正确
        """
        node_parts = []
        now_part = []
        father_list = {}
        for node in nodes+[target_node]:
            file_ns = node.file_ns if node.type not in ['file', 'folder'] else node.namespace
            if file_ns not in father_list:
                now_part = []
                father_list[file_ns] = len(node_parts)
                if len(node.content):
                    parent = context_dict[node.parent]
                    if node.type not in ["file", "folder"]:
                        while parent.type not in ["folder", "repository"]:
                            now_part.append(context_dict[parent.children[0]])
                            parent = context_dict[parent.parent]
                    now_part = now_part[::-1]
                    if node not in now_part:
                        now_part.append(node)
                    node_parts.append(now_part)
                else:
                    qika = 1
            else:
                if len(node.content):
                    now_part = []
                    parent = context_dict[node.parent]
                    if node.type not in ["file", "folder"]:
                        while parent.type not in ["folder", "repository"]:
                            if context_dict[parent.children[0]] in node_parts[father_list[file_ns]]:
                                break
                            now_part.append(context_dict[parent.children[0]])
                            parent = context_dict[parent.parent]
                    now_part = now_part[::-1]
                    if node not in now_part:
                        now_part.append(node)
                    for now_part_node in now_part:
                        if now_part_node not in node_parts[father_list[file_ns]]:
                            node_parts[father_list[file_ns]].append(now_part_node)
                else:
                    qika = 1
        
        node_parts[father_list[target_node.file_ns]].remove(target_node)
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
            if node.file_path != target_file and len(node.content):
                out_file_nodes.append(node)
                continue
                
            # 如果节点在目标文件中且在目标节点之前,保留该节点
            if node.begin_line < target_node.begin_line and len(node.content):
                in_file_nodes.append(node)
        
        out_file_nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line))
        in_file_nodes.sort(key=lambda x: x.begin_line)
        
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
        key = key.to(self.device)
        if pos == 0:
            return key
        elif pos > 0:
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
        cos = cos.unsqueeze(1).to(key.device)
        sin = sin.unsqueeze(1).to(key.device)
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
        
        if self.top_k and self.top_k < sorted_probs.shape[-1]:
            sorted_probs[..., self.top_k:] = 0.0
        
        # 重新归一化概率
        sorted_probs_sum = torch.sum(sorted_probs, dim=-1, keepdim=True)
        sorted_probs = sorted_probs / sorted_probs_sum
        
        # 按概率采样
        idx = torch.multinomial(sorted_probs, num_samples=1)
        
        # 获取采样的token id
        next_token = sorted_indices[0, idx[0]]
        
        return next_token


    def select_high_attention_nodes(self, target_node, nodes_ns: List[str], attn_scores: torch.Tensor, context_dict: Dict, min_num=16) -> List[ContextNode]:
        """
        根据注意力分数筛选高注意力节点
        """
        def filter_file(node):
            if node.name != "__init__":
                return True
            return False
        
        target_file = target_node.file_ns
        nodes = [context_dict[ns] for ns in nodes_ns]
        if target_node.class_ns:
            inclass_codes = [node.namespace for node in nodes if (node.type not in ["file", "folder"] and node.file_ns == target_file and node.class_ns == target_node.class_ns)]
            infile_codes = [node.namespace for node in nodes if (node.type not in ["file", "folder"] and node.file_ns == target_file and node.class_ns != target_node.class_ns)]
        else:
            inclass_codes = []
            infile_codes = [node.namespace for node in nodes if (node.type not in ["file", "folder"] and node.file_ns == target_file)]
        cross_codes = [node.namespace for node in nodes if (node.type not in ["file", "folder"] and node.file_ns != target_file)]
        file_nodes = [node.namespace for node in nodes if (node.type in ["file"])]
        folder_nodes = [node.namespace for node in nodes if (node.type in ["folder"])]
        
        # 如果没有任何筛选 则直接返回所有节点
        if len(infile_codes)<=min_num and len(cross_codes) <= min_num and len(file_nodes) <= 2 and len(folder_nodes) <= 2:
            cross_context = self.collect_children([context_dict[node] for node in (folder_nodes + file_nodes + cross_codes)], context_dict)
            infile_context = self.collect_children([context_dict[node] for node in infile_codes], context_dict)
            inclass_context = self.collect_children([context_dict[node] for node in inclass_codes], context_dict)
            
        else:
            cross_num = len(cross_codes)
            infile_num = len(infile_codes)
            class_num = len(inclass_codes)
            file_num = len(file_nodes)
            folder_num = len(folder_nodes)

            """
            此处分开寻找 类内 文件内 和 跨文件 节点
            类内节点全部保留
            文件内节点选择前66%最高注意力的节点 
            跨文件节点选择前33%最高注意力的节点 
            另外 每类代码节点 至少选择min_num个节点 每类文件节点至少选择2个节点
            """
            infile_num = max(1, int(min_num), int(infile_num * 0.66)) 
            cross_num = max(1, int(min_num), int(cross_num * 0.33)) 
            file_num = max(2, int(file_num * 0.33))
            folder_num = max(2, int(folder_num * 0.33))
            max_cross_num = self.max_crossfile_node_num
            max_infile_num = self.max_infile_node_num 
            
            node_size = [node.length for node in nodes]
            node_attn = torch.zeros(len(nodes), dtype=torch.float32)
            now_pos = 0
            for i, node in enumerate(nodes):
                node_scores = attn_scores[0, :, :, now_pos:now_pos+node_size[i]]
                now_pos += node_size[i]
                try:
                    node_attn[i] = node_scores[:,:,:].mean(dim=-1).max(dim=-1)[0].max(dim=-1)[0]
                except:
                    node_attn[i] = 0
            
            filter_value = node_attn.mean().item()
            selected_nodes = {
                "inclass": [],
                "infile": [],
                "cross": [],
                "file": [],
                "folder": [],
            }
            flag_type = {
                "inclass": False,
                "infile": False,
                "cross": False,
                "file": False,
                "folder": False,
            }
            
            _, indices = torch.sort(node_attn, descending=True)
            # top_k = max(1, min_num, int(len(nodes) * 0.3))  # 至少选择min_num个节点
            # selected_indices = indices[:top_k]
            for idx in indices:
                idx = idx.item()
                node = nodes[idx]
                if node_attn[idx] < filter_value:
                    break
                if not flag_type["inclass"] and node.namespace in inclass_codes:
                    if len(selected_nodes["inclass"]) >= class_num:
                        flag_type["inclass"] = True
                        continue
                    selected_nodes["inclass"].append(node)
                    if node.type in ["class", "function"]:
                        max_infile_num -= max(1, len(node.children))
                        max_cross_num -= max(1, len(node.children))
                    
                elif not flag_type["infile"] and node.namespace in infile_codes:
                    if len(selected_nodes["infile"]) >= min(infile_num, max_infile_num):
                        flag_type["infile"] = True
                        continue
                    selected_nodes["infile"].append(node)
                    if node.type in ["class", "function"]:
                        max_infile_num -= max(1, len(node.children))
                        max_cross_num -= max(1, len(node.children))
                    
                elif not flag_type["cross"] and node.namespace in cross_codes:
                    if len(selected_nodes["cross"]) >= min(cross_num, max_cross_num):
                        flag_type["cross"] = True
                        continue
                    selected_nodes["cross"].append(node)
                    if node.type in ["class", "function"]:
                        max_cross_num -= max(1, len(node.children))
                        
                elif not flag_type["file"] and node.namespace in file_nodes:
                    if len(selected_nodes["file"]) >= file_num:
                        flag_high = True
                        continue
                    selected_nodes["file"].append(node)
                    max_cross_num -= max(1, len(node.children))
                
                elif not flag_type["folder"] and node.namespace in folder_nodes:
                    if len(selected_nodes["folder"]) >= folder_num:
                        flag_type["folder"] = True
                        continue
                    selected_nodes["folder"].append(node)
                    
                if flag_type["inclass"] and flag_type["infile"] and flag_type["cross"] and flag_type["file"] and flag_type["folder"]:
                    break
            
            
            # 收集子节点作为下一层上下文
            cross_context = self.collect_children(selected_nodes["folder"]+selected_nodes["file"]+selected_nodes["cross"], context_dict)
            infile_context = self.collect_children(selected_nodes["infile"], context_dict)
            inclass_context = self.collect_children(selected_nodes["inclass"], context_dict)
        

        # 删去目标节点以及同文件内在目标节点之后的节点
        cross_context = self.remove_after_target(cross_context, target_node)
        infile_context = self.remove_after_target(infile_context, target_node)
        inclass_context = self.remove_after_target(inclass_context, target_node)
        
        file_context = [node for node in cross_context if node.type in ['file', 'folder']]
        code_context = [node for node in cross_context if node.type not in ['folder', 'file']]
        
        if (len(infile_context)+len(inclass_context)) > self.max_infile_node_num:
            # 随机删除多的file内节点
            code_num = self.max_infile_node_num - len(inclass_context)
            if code_num<=0:
                logging.debug(f"infile remove all nodes")
                infile_context = []
            elif code_num<len(infile_context):
                logging.debug(f"infile remove {len(infile_context)-code_num} nodes")
                infile_context = random.sample(infile_context, code_num)
        
        if len(code_context) + len(infile_context) + len(inclass_context) > self.max_crossfile_node_num:
            # 随机删除多的crossfile节点
            code_num = self.max_crossfile_node_num - len(infile_context) - len(inclass_context)
            if code_num<=0:
                logging.debug(f"crossfile remove all nodes")
                code_context = []
            elif code_num<len(code_context):
                logging.debug(f"crossfile remove {len(code_context)-code_num} nodes")
                code_context = random.sample(code_context, code_num)
        
        all_context = file_context + code_context + infile_context + inclass_context
        
        change = False
        for node in all_context:
            if node.namespace not in nodes_ns:
                change = True
                break
        
        return all_context, change        
    def generate_step(self, 
                        target_namespace, 
                        input_ids: torch.Tensor, 
                        prefix_kv, 
                        context_dict: Dict, 
                        init_context_nodes: List[str]
                    ) -> Tuple[torch.Tensor, List[ContextNode]]:
        """
        执行一步生成,包含逐层深入的上下文搜索
        返回生成的token_id和使用的上下文节点
        """
        curr_context = [context_dict[ns] for ns in init_context_nodes]
        layer_outputs = []
        seen_context = []
        input_k = [None]*len(self.model.layers)
        input_v = [None]*len(self.model.layers)
        node_parts = None
        change_flag = True
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        begin_pos = 0
        if prefix_kv:
            begin_pos = prefix_kv.key_cache[0].shape[2]
        
        
        # 逐层处理
        for layer_idx in range(len(self.model.layers)):
            # 去除无效上下文
            if not node_parts:
                curr_context = [node for node in curr_context if len(node.content) > 0]
                node_parts = self.cluster_brothers(curr_context, context_dict[target_namespace], context_dict)
            
            context_k = []
            context_v = []
            now_pos = 0
            for part_idx, node_list in enumerate(node_parts[-1::-1]):
                node_list = [node.namespace for node in node_list]
                k, v, node_list = self.encode_nodeseq(node_list, layer_idx, context_dict, 0)\
                # 从位置0开始向右延伸
                k = self.shift_pos(k, now_pos)
                now_pos += sum([node.length for node in node_list])
                
                context_k = [k]+context_k
                context_v = [v]+context_v
                node_parts[part_idx] = node_list
                del k, v
            
            curr_context = [node for lst in node_parts for node in lst]
            past_key_values:DynamicCache = DynamicCache()
            prefix_length = 0
            if prefix_kv is not None and len(prefix_kv.key_cache) > layer_idx:
                prefix_length = prefix_kv.key_cache[layer_idx].shape[2]
                prefix_k = self.shift_pos(prefix_kv.key_cache[layer_idx], now_pos)
                context_k.append(prefix_k.to(self.device))
                context_v.append(prefix_kv.value_cache[layer_idx].to(self.device))
                
            if len(context_k):
                context_k = torch.cat(context_k, dim=2).to(self.device)
                context_v = torch.cat(context_v, dim=2).to(self.device)

                """max_length = self.max_context_length+prefix_length
                if context_k.shape[2] > max_length:
                    context_k = context_k[:, :, -max_length:]
                    context_v = context_v[:, :, -max_length:]"""
                
                past_key_values.update(context_k, context_v, layer_idx)
                if layer_idx != 0:
                    past_key_values.update(context_k, context_v, 0)
            
            del context_k, context_v
            
            # 编码当前层输入特征
            layer = self.model.layers[layer_idx]
            if layer_idx == 0:
                hidden_states = inputs_embeds
            else:
                hidden_states = layer_outputs[0]

            # 编码位置信息和掩码 输入的起始位置为上下文总长度+缓存输入的长度
            cache_position = torch.arange(
                now_pos+begin_pos, now_pos+begin_pos+inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            causal_mask = self.model._update_causal_mask(
                    None, inputs_embeds, cache_position, past_key_values, output_attentions=True
                )

            if layer_idx in [0, 35]:
                qika = 1
            
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
            
            # 将当前输入的kv加入prefix_kv:
            input_k[layer_idx] = self.shift_pos(layer_outputs[2].key_cache[layer_idx][:,:,-hidden_states.shape[1]:], -now_pos).detach().to("cpu")
            input_v[layer_idx] = layer_outputs[2].value_cache[layer_idx][:,:,-hidden_states.shape[1]:].detach().to("cpu")
            
            if layer_idx in [0, 35]:
                qika = 1
            
            if layer_idx == 0:
                # 分类输出节点数量
                folder_num = len([node for node in curr_context if node.type == 'folder'])
                file_num = len([node for node in curr_context if node.type == 'file'])
                code_num = len([node for node in curr_context if node.type not in ['folder', 'file']])
                logging.debug(f"Layer {layer_idx} get {folder_num} folders, {file_num} files, {code_num} codes")
                
                for node in curr_context:
                    if node.type == "code":
                        node = context_dict[node.parent]
                    if node.type in ['function', 'class'] and node.namespace not in seen_context:
                        seen_context.append(node.namespace)
                
            
            if change_flag and layer_idx%self.past_layers == self.past_layers-1:
                # 选择高注意力节点
                curr_context = [node.namespace for node in curr_context]
                attn_scores = layer_outputs[1]
                high_attn_nodes, change_flag = self.select_high_attention_nodes(context_dict[target_namespace], curr_context, attn_scores, context_dict, min_num=16)    
                
                for node in high_attn_nodes:
                    if node.type in ['function', 'class'] and node.namespace not in seen_context:
                        seen_context.append(node.namespace)
                
                node_parts = self.cluster_brothers(high_attn_nodes, context_dict[target_namespace], context_dict)
                curr_context = [node for node_part in node_parts for node in node_part]
                
                # 分类输出节点数量
                if change_flag:
                    folder_num = len([node for node in curr_context if node.type == 'folder'])
                    file_num = len([node for node in curr_context if node.type == 'file'])
                    code_num = len([node for node in curr_context if node.type not in ['file', "folder"]])
                    logging.debug(f"Layer {layer_idx} get {folder_num} folders, {file_num} files, {code_num} codes")
                
                pass
                
            # 释放显存
            del past_key_values, causal_mask
            del cache_position, position_embeddings, position_ids
            torch.cuda.empty_cache()
            
            pass
            
        # 保存当前步kv
        if prefix_kv is None:
            prefix_kv = DynamicCache()

        for layer_idx in range(len(self.model.layers)):
            prefix_kv.update(input_k[layer_idx], input_v[layer_idx], layer_idx)
        
        # 计算最终输出
        hidden_states = layer_outputs[0]
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # 使用类方法进行采样
        next_token = self.sample_next_token(logits[:, -1])
        
        return next_token, curr_context, context_dict, prefix_kv

