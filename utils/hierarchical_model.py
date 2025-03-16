import logging
logging.basicConfig(level=logging.DEBUG)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from typing import List, Dict, Tuple
import random
import matplotlib.pyplot as plt
import itertools
import cProfile
import yappi
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
        plt.xlabel("node")
        plt.ylabel("Value")
        plt.title(title)
        plt.show()
    
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
        self.max_layer = len(self.model.layers)
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = model.device
        self.max_context_length = 2048
        self.max_length = 512
        self.output_attentions = False
        
        # 项目数据库
        self.context_dict = None
        self.target_node = None
        
        # 层展开参数
        self.spread_layer = 8
        self.max_spread_turn = 16
        self.max_node_length = 256
        self.max_crossfile_node_num = 128
        self.max_infile_node_num = 32
        
        # 采样参数
        self.top_p = 0.8
        self.top_k = 20
        self.temperature = 0.7
        pass
    
    def node_length(self, node):
        """
        获取输入的token长度
        """
        length = min(self.max_context_length, len(self.tokenizer.tokenize(node.content)))
        return length
    def encode_init_hidden(self, input_ids):
        """
        编码初始hidden 即模型输入未经过第一次Attn计算前的hidden
        """
        input_ids = input_ids
        inputs_embeds = self.model.embed_tokens(input_ids)
        # create position embeddings to be shared across the decoder layers
        return inputs_embeds
    
    def encode_by_layer(self, hidden_states, start_layer, end_layer, begin_pos = 0):
        """
        将hidden_states从第start_layer层到第end_layer层进行编码
        参数:
            hidden_states: 输入hidden [seq_len, hidden_dim]
            start_layer: 起始层索引 int
            end_layer: 结束层索引 int
            begin_pos: 起始位置索引 int
        返回:
            all_key: key向量 [layer_num, head_num, seq_len, dim]
            all_value: value向量 [layer_num, head_num, seq_len, dim]
            all_hidden: hidden向量 [layer_num, seq_len, hidden_dim]
        """
        assert len(hidden_states.shape) == 2, "hidden_states must be [seq_len, hidden_dim]"
        seq_len = hidden_states.shape[0]
        hidden_list = []
        hidden_states = hidden_states.unsqueeze(0)
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
            decoder_layer = self.model.layers[layer_id]
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids.unsqueeze(0),
                    use_cache = True,
                    past_key_value = past_key_value,
                    output_attentions=False,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]
            hidden_list.append(layer_outputs[0].squeeze(0))

        all_key = torch.stack(past_key_value.key_cache[start_layer:end_layer+1], dim=1).squeeze(0).detach()
        all_value = torch.stack(past_key_value.value_cache[start_layer:end_layer+1], dim=1).squeeze(0).detach()
        all_hidden = torch.stack(hidden_list, dim=0).detach()
        
        del hidden_states, hidden_list, past_key_value, position_ids, position_embeddings, causal_mask
        assert all_key.shape[0] == all_value.shape[0] == all_hidden.shape[0] == end_layer - start_layer + 1, "layer num error"
        assert all_key.shape[-2] == all_value.shape[-2] == all_hidden.shape[-2] == seq_len, "seq_len error"
        return all_key, all_value, all_hidden

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

    def stat_0(self):
        pass
    
    def stat_1(self):
        pass
    
    def stat_2(self):
        pass
    
    def get_node_kv(self, node_list: list[ContextNode], begin_layer: int, end_layer:int, begin_pos: int = 0):
        """
        获取节点从begin_layer到end_layer的KV向量对 计算时将node_list的内容拼接成一个整体进行编码
        
        如果节点已有向量缓存且层数足够,直接返回缓存的向量
        如果节点已有向量缓存但层数不足,基于最后一层缓存继续计算到目标层
        如果节点没有向量缓存,使用模型从头计算到目标层
        
        参数:
            node_list: 上下文节点列表 list[ContextNode]
            begin_layer: 起始层索引 int
            end_layer: 结束层索引 int
            begin_pos: 起始位置索引 int
        返回:
            all_key: key向量 [layer_num, head_num, seq_len, dim]
            all_value: value向量 [layer_num, head_num, seq_len, dim]
            node_list: 编码节点列表 list[ContextNode]
        """
        all_key, all_value = [], []
        
        # 获取所有节点都已经编码的最深层
        enc_layer_idx = min([self.max_layer]+[node.vectors[0].shape[0]-1 if (node.vectors is not None and len(node.vectors)) else -1 for node in node_list])
        # 构建节点向量
        if enc_layer_idx >= end_layer:
            # 使用缓存的向量
            now_pos = 0
            for node in node_list:
                node_key_by_layer = []
                assert node.vectors_pos[begin_layer:end_layer+1].max() == node.vectors_pos[begin_layer:end_layer+1].min(), "node_vectors_pos error"
                node_key_by_layer= self.shift_pos(node.vectors[0][begin_layer:end_layer+1], now_pos+begin_pos-node.vectors_pos[begin_layer])
                node_value_by_layer = node.vectors[1][begin_layer:end_layer+1]
                all_key.append(node_key_by_layer)
                all_value.append(node_value_by_layer)
                now_pos += node.length
            all_key = torch.cat(all_key, dim=2)
            all_value = torch.cat(all_value, dim=2)
            self.stat_0()
            pass
        elif 0 <= enc_layer_idx < end_layer:
            # 如果节点已经存在向量 但是长度小于当前层索引 则基于缓存的最后层向量继续计算获取到当前层
            # 首先将所有节点enc_layer_idx层的hidden拼接为一个序列
            hidden_seq = []
            hidden_length = []
            now_pos = 0
            for node in node_list:
                hidden_seq.append(node.vectors[-1][enc_layer_idx])  # 获取每个节点最后缓存层的hidden state
                hidden_length.append(node.vectors[-1][enc_layer_idx].shape[1]) # 获取每个节点特征的长度
                now_pos += node.length
            node_list = node_list[:len(hidden_length)]
            last_hidden = torch.cat(hidden_seq, dim=-2)  # 在序列维度上拼接
            
            # 基于最后一层缓存的向量继续计算到当前层
            all_key, all_value, all_hidden = self.encode_by_layer(last_hidden, enc_layer_idx+1, end_layer, begin_pos)
            all_key = all_key.cpu()
            all_value = all_value.cpu()
            all_hidden = all_hidden.cpu()
            hidden_start = 0
            hidden_end = 0
            layer_start = enc_layer_idx+1
            for i, node in enumerate(node_list):
                hidden_end += node.length
                node.vectors[0] = torch.cat([node.vectors[0][:layer_start], all_key[:,:,hidden_start:hidden_end]], dim=0)
                node.vectors[1] = torch.cat([node.vectors[1][:layer_start], all_value[:,:,hidden_start:hidden_end]], dim=0)
                node.vectors[2] = torch.cat([node.vectors[2][:layer_start], all_hidden[:,hidden_start:hidden_end]], dim=0)
                node.vectors_pos = torch.cat([node.vectors_pos[:layer_start], torch.tensor([hidden_start], device='cpu').repeat(end_layer-enc_layer_idx)], dim=0)
                if node.vectors[0].shape[0] != node.vectors_pos.shape[0]:
                    qika = 1
                hidden_start = hidden_end
                node_list[i] = node
            self.stat_1()
            pass
        else:
            # 如果节点不存在向量 则使用模型编码到当前层
            if len(node_list[0].vectors):
                qika = 1
            if begin_layer > 0:
                qika = 1
            input_ids = []
            now_pos = 0
            for node in node_list:
                inputs = self.tokenizer(node.content, return_tensors="pt")['input_ids']
                inputs = inputs[:,-self.max_node_length:]
                node.length = inputs.shape[1]
                now_pos += node.length
                input_ids.append(inputs)
            node_num = len(input_ids)
            node_list = node_list[:node_num]
            input_ids = torch.cat(input_ids, dim=1)
            init_hidden = self.encode_init_hidden(input_ids).squeeze(0)
            all_key, all_value, all_hidden = self.encode_by_layer(init_hidden, 0, end_layer, begin_pos)
            all_key = all_key.cpu()
            all_value = all_value.cpu()
            all_hidden = all_hidden.cpu()
            # 将新计算的向量按层添加到缓存中
            hidden_start = 0
            hidden_end = 0
            for i, node in enumerate(node_list):
                hidden_end += node.length
                node.vectors = [[],[],[]]
                node.vectors_pos = torch.tensor([hidden_start], device='cpu').repeat(end_layer+1)
                node.vectors[0] = all_key[:,:,hidden_start:hidden_end]
                node.vectors[1] = all_value[:,:,hidden_start:hidden_end]
                node.vectors[2] = all_hidden[:,hidden_start:hidden_end]
                hidden_start = hidden_end
                node_list[i] = node
                if node.vectors[0].shape[0] != node.vectors_pos.shape[0]:
                    qika = 1
            self.stat_2()
            pass

        assert all_key.shape[0] == all_value.shape[0] == end_layer-begin_layer+1, "all_key/all_value/layer_num不匹配"
        assert all_key.shape[2] == all_value.shape[2] == sum([node.length for node in node_list]), "all_key/all_value/node_length不匹配"
        return all_key, all_value, node_list
    
    def extend_nodeseq(self, node_list: List[ContextNode]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果当前在处理代码节点序列，则为了整合上下文信息，将节点的前序节点拼接直到填满上下文窗口
        if len(node_list) == 0:
            return []
        
        file_ns = node_list[0].file_ns
        
        high_list = []
        node_parts = None
        file_length = {}
        for pos, node in enumerate(node_list):
            if node.type in ["file", "folder", "repository"]:
                high_list.append(node)
            elif node_parts == None:
                node_parts = [[node]]
                file_length[node.file_ns] = node.length
            elif node.file_ns == node_parts[-1][-1].file_ns and node.begin_line == node_parts[-1][-1].end_line:
                node_parts[-1].append(node)
                file_length[node.file_ns] += node.length
            else:
                node_parts.append([node])
                if node.file_ns not in file_length:
                    file_length[node.file_ns] = node.length
                else:
                    file_length[node.file_ns] += node.length
            
        extend_node_list = node_parts

        all_length = sum([node.length for node in node_list])
        unfinished_ids = [i for i in range(len(extend_node_list))]
        now_part = 0
        while (all_length < 31000) and len(unfinished_ids):
            now_part = now_part%len(unfinished_ids)
            now_ids = unfinished_ids[now_part]
            now_node = extend_node_list[now_ids][0]

            while now_node.type not in ["file", "folder", "repository"] and now_node.previous is None:
                now_node = self.context_dict[now_node.parent]

            if now_node.type in ["file", "folder", "repository"] or now_node.previous is None:
                unfinished_ids.remove(now_ids)
            else:
                previous_node = self.context_dict[now_node.previous]
                assert previous_node.type not in ["folder", "file", "repository"]
                
                """while len(previous_node.children):
                    previous_node = self.context_dict[previous_node.children[-1]]"""
                
                if previous_node.type in ["function", "class"] and previous_node.length == 0:
                    heads = previous_node.dfs_heads(self.context_dict)
                    previous_node.content = "\n".join([head.content for head in heads])+'\n'
                    self.context_dict[previous_node.namespace].content = previous_node.content
                    pass
                if not hasattr(previous_node, "token_ids") or previous_node.token_ids == None:
                    tokens = self.tokenizer(previous_node.content)['input_ids']
                    previous_node.token_ids = tokens
                previous_node.length = min(self.max_node_length, len(previous_node.token_ids))
                
                #判断前序是不是已经接触到上一个节点
                if now_ids > 0 and previous_node.begin_line < extend_node_list[now_ids-1][-1].end_line:
                    unfinished_ids.remove(now_ids)
                elif file_length[now_node.file_ns] + previous_node.length > int(self.max_context_length):
                    unfinished_ids.remove(now_ids)
                else:
                    extend_node_list[now_ids] = [previous_node]+extend_node_list[now_ids]
                    now_part += 1
                    all_length += previous_node.length
                    file_length[now_node.file_ns] += previous_node.length
        
        extend_parts = []
        extend_part = []
        for part in extend_node_list:
            if len(extend_part)==0 or part[0].file_ns == extend_part[0].file_ns:
                extend_part+=part
            else:
                extend_parts.append(extend_part)
                extend_part = part
        if len(extend_part):
            extend_parts.append(extend_part)

        return [high_list]+extend_parts
    

    def encode_nodeseq(self, nodes: List[str], begin_layer: int, end_layer: int, now_pos) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将上下文节点序列编码为KV对 此函数主要用于编码一个父节点下的兄弟节点序列 将它们拼接成一个整体进行kv计算
        一个节点的KV对由两部分组成 分别是其本身内容的向量 self_k, self_v 以及其子节点分别的内容平均向量 children_k, children_v
        其本身内容向量由和nodes列表其他内容拼接后计算得到 子节点内容向量由其所有子节点内容拼接后计算得到
        如果节点已经存在kv直接使用 否则用模型编码并缓存回上下文节点字典context_dict中
        TODO 现在now_pos未发挥作用 所有编码都是从0位置开始 需要将编码后的结果旋转到now_pos位置
        参数:
            nodes: 上下文节点列表, 元素为namespace
            layer_idx: 目标模型层索引
        """        
        node_list = []
        now_length = 0
        for node_ns in nodes:
            if now_length > self.max_context_length:
                break
            if node_ns not in self.context_dict:
                logging.warning(f"节点{node_ns}不存在, 将使用空字符串代替")
            else:
                node = self.context_dict[node_ns]
                now_length += node.length
                node_list.append(node)
        
        all_key, all_value, node_list = self.get_node_kv(node_list, begin_layer, end_layer) # 节点本身内容对应的向量
        now_pos = sum([node.length for node in node_list])
        """
        for i, node in enumerate(node_list):
            self.context_dict[node.namespace] = node

            # 构建子节点向量
            children_k, children_v = [], [] # 子节点内容平均向量
            children_list = []
            for child_ns in node.children:
                if child_ns not in self.context_dict:
                    logging.warning(f"节点{child_ns}不存在, 将使用空字符串代替")
                else:
                    child = self.context_dict[child_ns]
                    if child.content is None:
                        logging.warning(f"节点{child_ns}不存在内容, 开始创建")
                        child.content = "# "+ child.type + ":"+ child.namespace + "\n"
                        self.context_dict[child_ns] = child
                    if len(child.content):
                        children_list.append(child)
            # child_k, child_v, children_list = self.get_node_kv(children_list, layer_idx, sum(node_size)) # 子节点内容对应的向量
            children_length = 0
            
            # 去除子节点信息
            children_list = []
            
            for j, child in enumerate(children_list):
                children_k.append(self.shift_pos(child_k[j].to(self.device), now_pos).mean(dim=2).unsqueeze(2))
                children_v.append(child_v[j].to(self.device).mean(dim=2).unsqueeze(2))
                self.context_dict[child.namespace] = child
            all_keys.append(torch.cat([self_k[i].to(self.device)]+children_k, dim=2))
            all_values.append(torch.cat([self_v[i].to(self.device)]+children_v, dim=2))
        """
                     
        return all_key, all_value, node_list 

    def encode_fileseq(self, nodes: List[str], begin_layer: int, end_layer: int, now_pos=0) -> Tuple[torch.Tensor, torch.Tensor]:
        subpart_list = []
        subpart = []
        now_length = 0
        for idx, node_ns in enumerate(nodes):
            node_length = min(len(self.tokenizer.tokenize(self.context_dict[node_ns].content)), self.max_node_length)
            if now_length + node_length > self.max_context_length:
                subpart_list.append(subpart)
                subpart = [node_ns]
                now_length = node_length
            else:
                subpart.append(node_ns)
                now_length += node_length
        subpart_list.append(subpart)
        
        key_tensor = []
        value_tensor = []
        extend_list = []
        sub_pos = 0
        for subpart in subpart_list:
            subkey_tensor, subvalue_tensor, subextend_list = self.encode_nodeseq(subpart, begin_layer=begin_layer, end_layer=end_layer, now_pos=now_pos+sub_pos)
            key_tensor.append(subkey_tensor)
            value_tensor.append(subvalue_tensor)
            extend_list.extend(subextend_list)
            sub_pos += subkey_tensor.shape[2]
        key_tensor = torch.cat(key_tensor, dim=2)
        value_tensor = torch.cat(value_tensor, dim=2)
        return key_tensor, value_tensor, extend_list
    def collect_children(self, nodes: List[ContextNode]) -> List[ContextNode]:
        """收集节点的所有子节点 如果不存在子节点则继续保留当前节点 按父节点内子节点列表顺序排序"""
        nodes = [node for node in nodes if isinstance(node, ContextNode)]
        # nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line))
        children = []
        keep_nodes = []
        for node in nodes:
            if len(node.children) == 0:
                children.append(node)
            if len(node.children) != 0:
                for child_ns in node.children:
                    children.append(self.context_dict[child_ns])
                
        return children

    def cluster_brothers(self, nodes: List[ContextNode], target_node) -> List[List[str]]:
        """
        将节点列表按照文件进行聚类 并填补缺失的语法上级 例如如果保留了一个类函数里的代码片段 则此类函数的head节点 此类的head节点 此文件的head节点都需要保留以保证文件内节点组织成的代码段结构正确
        """
        node_parts = {}
        now_part = []
        nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line))
        for node in nodes+[target_node]:
            file_ns = node.file_ns if node.type not in ['file', 'folder'] else " "
            if file_ns not in node_parts:
                if len(node.content):
                    now_part = []
                    node_parts[file_ns] = set()
                    parent = self.context_dict[node.parent]
                    if node.type not in ["file", "folder"]:
                        while parent.type not in ["folder", "repository"]:
                            now_part.append(self.context_dict[parent.children[0]])
                            parent = self.context_dict[parent.parent]
                    now_part = now_part[::-1]
                    now_part.append(node)
                    node_parts[file_ns].update(now_part)
                else:
                    qika = 1
            else:
                if len(node.content):
                    now_part = []
                    parent = self.context_dict[node.parent]
                    if node.type not in ["file", "folder"]:
                        while parent.type not in ["folder", "repository"]:
                            if self.context_dict[parent.children[0]] in node_parts[file_ns]:
                                break
                            now_part.append(self.context_dict[parent.children[0]])
                            parent = self.context_dict[parent.parent]
                    now_part = now_part[::-1]
                    now_part.append(node)
                    node_parts[file_ns].update(now_part)
                else:
                    qika = 1
        
        # 排序并分类
        infile_part = None
        high_part = None
        cross_parts = []
        for key in node_parts:
            if key == " ":
                high_part = sorted(node_parts[key], key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line, x.end_line))
            elif key == target_node.file_ns:
                infile_part = sorted(node_parts[key], key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line, x.end_line))
            else:
                cross_parts.append(sorted(node_parts[key], key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line, x.end_line)))
        # 按照part文件路径与target_node文件路径的区别大小排序
        def sort_key(node_ns, target_ns):
            node_ns = node_ns.split(".")
            target_ns = target_ns.split(".")
            diff_pos = 0
            while diff_pos < len(node_ns) and diff_pos < len(target_ns):
                if node_ns[diff_pos] != target_ns[diff_pos]:
                    break
                diff_pos += 1
            return diff_pos
        cross_parts.sort(key=lambda x: (sort_key(x[0].file_ns if x[0].file_ns else x[0].namespace, target_node.file_ns), len((x[0].file_ns if x[0].file_ns else x[0].namespace).split("."))))
        node_parts = cross_parts
        if high_part:
            for node in high_part:
                if node.type in ["folder", "file"] and node.length == 0:
                    instruct = "\n## Here are cross-file contents from {}.".format(node.content[2:-1])
                    if node.type == "file":
                        instruct += " We simplified the code by removing some code blocks."
                    instruct += "\n\n"
                    node.content = instruct
                    for idx, child_ns in enumerate(node.children):
                        if idx == 0 and node.type == "file":
                            self.context_dict[child_ns].content = node.content
                        elif self.context_dict[child_ns].type in ["folder", "file"]:
                            node.content += self.context_dict[child_ns].content + '\n'
                        elif self.context_dict[child_ns].type in ["function", "class"]:
                            heads = self.context_dict[child_ns].dfs_heads(self.context_dict)
                            node.content += "\n".join([head.content for head in heads])+'\n'
                    node.content += "\n"
                    self.context_dict[node.namespace].content = node.content
                    pass
            node_parts = [high_part] + node_parts
        if infile_part:
            node_parts = node_parts + [infile_part]
        
        for part in node_parts:
            for node in part:
                if node.type in ["function", "class"] and node.length == 0:
                    heads = node.dfs_heads(self.context_dict)
                    node.content = "\n".join([head.content for head in heads])+'\n'
                    self.context_dict[node.namespace].content = node.content
                    pass
        
        for idx, part in enumerate(node_parts):
            part_length = 0
            for idy, node in enumerate(part):
                if not hasattr(node, "token_ids") or node.token_ids == None:
                    tokens = self.tokenizer(node.content)['input_ids']
                    node.token_ids = tokens
                if node.type in ["folder", "file"]:
                    node.length = min(self.max_context_length, len(node.token_ids))
                else:
                    node.length = min(self.max_node_length, len(node.token_ids))
                    if node.file_ns != target_node.file_ns and part_length + node.length > self.max_context_length:
                        node_parts[idx] = part[:idy]
                        break
        return node_parts

    def cut_to_encode(self, node_parts, extend_parts):
        """
        将part拼接成尽量填充模型窗口以降低编码次数
        """
        encode_parts = []
        encode_tokens = []
        encode_masks = []
        now_part = []
        now_token = []
        now_mask = []
        now_length = 0
        for idx, node_part in enumerate(node_parts):
            if node_part[0].type in ["file", "folder"]:
                now_length = 0
                now_part = []
                now_token = []
                now_mask = []
                for idy, node in enumerate(extend_parts[idx]):
                    now_length += node.length
                    if node in node_part:
                        now_part.append(node)
                    now_token += node.token_ids[:node.length]
                    now_mask += [1 if node in node_part else 0]*node.length
                    if (idy == len(extend_parts[idx])-1) or (now_length+extend_parts[idx][idy+1].length > self.max_context_length):
                        encode_parts.append(now_part)
                        encode_tokens.append(now_token)
                        encode_masks.append(now_mask)
                        now_length = 0
                        now_part = []
                        now_mask = []
                        now_token = []
            else:
                part_length = sum([node.length for node in extend_parts[idx]])
                now_length += part_length
                now_part += node_part
                for idy, node in enumerate(extend_parts[idx]):
                    now_token += node.token_ids[:node.length]
                    now_mask += [1 if node in node_part else 0]*node.length
                if (idx == len(node_parts)-1) or (now_length+sum([node.length for node in extend_parts[idx+1]]) > self.max_context_length):
                    encode_parts.append(now_part)
                    encode_tokens.append(now_token)
                    encode_masks.append(now_mask)
                    now_length = 0
                    now_part = []
                    now_mask = []
                    now_token = []
        return encode_parts, encode_tokens, encode_masks
            
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
        
        # out_file_nodes.sort(key=lambda x: ((x.file_path if x.file_path else x.namespace), x.begin_line))
        # in_file_nodes.sort(key=lambda x: x.begin_line)
        
        return out_file_nodes+in_file_nodes
        
    def shift_pos(self, key:torch.Tensor, pos:int):
        """
        基于ROPE位置编码对输入向量进行平移
        输入:
            key: torch.Tensor((num), head_num, seq_len, head_dim)
            pos: int
        输出:
            key: torch.Tensor((num), head_num, seq_len, head_dim)
        """
        if pos == 0:
            return key
        length = key.shape[-2]
        key_divice = key.device
        key = key
        dim_num = len(key.shape)
        if dim_num == 3:
            key = key.unsqueeze(0)
        if pos > 0:
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
        if dim_num == 3:
            k_embed = k_embed.squeeze(0)
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

    def visualize_attention(self, target_node, nodes, attn_scores: torch.Tensor, layer_id=0):
        target_file = target_node.file_path
        node_size = [node.length for node in nodes]
        node_attn = torch.zeros(len(nodes)+1, dtype=torch.float32)
        type_attn = torch.zeros(3, dtype=torch.float32)
        type_length = torch.zeros(3, dtype=torch.float32)
        now_pos = 0
        attn_scores = attn_scores[0, :, -1, :].detach().cpu()
        for i, node in enumerate(nodes):
            node_scores = attn_scores[:, now_pos:now_pos+node_size[i]]
            now_pos += node_size[i]
            try:
                node_attn[i] = node_scores[:,:].mean(dim=0).sum(dim=-1)
            except:
                node_attn[i] = 0
            if node.file_path and node.file_path == target_file:
                type_attn[1] += node_attn[i]
                type_length[1] += node_size[i]
            else:
                type_attn[2] += node_attn[i]
                type_length[2] += node_size[i]
        node_attn[-1] = attn_scores[:, now_pos:-1].mean(dim=0).sum(dim=-1)
        type_attn[0] = node_attn[-1]
        type_length[0] = attn_scores[:, now_pos:-1].shape[-1]
        
        mean_type_attn = type_attn/type_length
        # visualize_tensor(node_attn, layer_id)
        
        return mean_type_attn

    def select_high_attention_nodes(self, target_node, nodes_ns: List[str], attn_scores: torch.Tensor, min_num=8) -> List[ContextNode]:
        """
        根据注意力分数筛选高注意力节点
        """
        
        def weight_sum(attn_scores):
            """
            对注意力分数矩阵沿输入（即倒数第二维）进行加权求和，并返回加权后的结果。
            权重设计为最后一步为1，倒数第二步为0.5，倒数第三步为0.25，以此类推。
            """
            weights = torch.tensor([1, 0.5, 0.25, 0.125, 0.0625, 0.03125]).bfloat16()
            # 因为越靠后的步骤权重越大，因此需要倒序
            weights = torch.flip(weights, dims=[0])
            weights = weights.to(attn_scores.device)
            # 截断多余的部分
            attn_scores = attn_scores[:,:,-weights.shape[0]:,:]
            weights = weights[-attn_scores.shape[3]:]
            weighted_sum = torch.einsum('...ij,...i->...', attn_scores, weights, type='BFloat16')
            return weighted_sum
        
        
        def filter_file(node):
            if node.name != "__init__":
                return True
            return False
        
        target_file = target_node.file_ns
        nodes = [self.context_dict[ns] for ns in nodes_ns]
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
            new_cross_context = self.collect_children([self.context_dict[node] for node in (folder_nodes + file_nodes)])
            old_cross_code_context = self.collect_children(cross_codes,)
            infile_context = self.collect_children([self.context_dict[node] for node in infile_codes])
            inclass_context = self.collect_children([self.context_dict[node] for node in inclass_codes])
            
            min_cross_num = min_num 
            max_cross_num = 3*self.max_crossfile_node_num
            now_cross_num = max_cross_num
            max_infile_num = self.max_infile_node_num 
            
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
            infile_num = self.max_infile_node_num 
            file_num = min(max(3, int(file_num * 0.33)), 9)
            folder_num = min(max(3, int(folder_num * 0.66)), 9)
            # 之前的跨文件代码节点保留至少min_cross_num(1/3)，max_corss_num的其余位置如果有剩余则继续保留直到到顶
            min_cross_num = max(1, int(min_num), int(cross_num * 0.33)) 
            max_cross_num = 3*self.max_crossfile_node_num
            now_cross_num = max_cross_num if (len(file_nodes)+len(folder_nodes)>0) else self.max_crossfile_node_num
            max_infile_num = self.max_infile_node_num 
            
            node_size = [node.length for node in nodes]
            node_attn = torch.zeros(len(nodes), dtype=torch.float32)
            now_pos = 0
            attn_scores =  attn_scores[0,:,-1,:] # weight_sum(attn_scores)
            for i, node in enumerate(nodes):
                node_scores = attn_scores[:, now_pos:now_pos+node_size[i]]
                now_pos += node_size[i]
                try:
                    # 对节点的注意力为其20%最高注意力的平均
                    # 先找到20%最高注意力
                    node_scores = node_scores[:,:].topk(int(1 + node_size[i]*0.2), dim=-1).values
                    node_attn[i] = node_scores[:,:].mean(dim=-1).mean(dim=0)
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
                if not flag_type["inclass"] and node.namespace in inclass_codes:
                    if len(selected_nodes["inclass"]) >= class_num:
                        flag_type["inclass"] = True
                        continue
                    selected_nodes["inclass"].append(node)
                    
                elif not flag_type["infile"] and node.namespace in infile_codes and node.name != "_head":
                    selected_nodes["infile"].append(node)
                    max_infile_num -= max(1, len(node.children))
                    if max_infile_num <= 0:
                        flag_type["infile"] = True
                        continue
                    
                elif not flag_type["cross"] and node.namespace in cross_codes and node.name != "_head":
                    selected_nodes["cross"].append(node)
                    now_cross_num -= max(1, len(node.children))
                    if len(selected_nodes["cross"]) >= min_cross_num and now_cross_num<=0: # or node_attn[idx] < filter_value:
                        flag_type["cross"] = True
                        continue
                        
                elif not flag_type["file"] and node.namespace in file_nodes and filter_file(node):
                    if len(selected_nodes["file"]) >= file_num and now_cross_num <= 0:
                        flag_type["file"] = True
                        continue
                    selected_nodes["file"].append(node)
                    now_cross_num -= max(1, len(node.children))
                
                elif not flag_type["folder"] and node.namespace in folder_nodes:
                    if len(selected_nodes["folder"]) >= folder_num:
                        flag_type["folder"] = True
                        continue
                    selected_nodes["folder"].append(node)
                    
                if flag_type["inclass"] and flag_type["infile"] and flag_type["cross"] and flag_type["file"] and flag_type["folder"]:
                    break
            
            
            # 收集子节点作为下一层上下文
            new_cross_context = self.collect_children(selected_nodes["folder"]+selected_nodes["file"])
            old_cross_code_context = self.collect_children(selected_nodes["cross"])
            infile_context = self.collect_children(selected_nodes["infile"])
            inclass_context = self.collect_children(selected_nodes["inclass"])
        

        # 删去目标节点以及同文件内在目标节点之后的节点
        infile_context = self.remove_after_target(infile_context, target_node)
        inclass_context = self.remove_after_target(inclass_context, target_node)
        
        file_context = [node for node in new_cross_context if node.type in ['file', 'folder']]
        new_cross_code_context = [node for node in new_cross_context if node.type not in ['folder', 'file']]
        
        if (len(infile_context)) > self.max_infile_node_num:
            # 删除多的file内节点
            code_num = self.max_infile_node_num
            if code_num<=0:
                logging.debug(f"infile remove all nodes")
                infile_context = []
            elif code_num<len(infile_context):
                cut_num = len(infile_context)-code_num
                cut_range = min(len(infile_context), 3*cut_num)
                
                logging.debug(f"infile remove {len(infile_context)-code_num} nodes")
                infile_context = infile_context[:-cut_range]+random.sample(infile_context[-cut_range:], cut_range-cut_num)
                
        
        if len(new_cross_context):
            old_corss_num = max(min_cross_num, max_cross_num-len(new_cross_code_context))
        else:
            old_corss_num = self.max_crossfile_node_num
        if len(old_cross_code_context) > old_corss_num:
            # 删除多的旧crossfile节点
            if old_corss_num<=0:
                logging.debug(f"old crossfile remove all nodes")
                old_cross_code_context = []
            else:
                cut_num = len(old_cross_code_context)-old_corss_num
                cut_range = min(len(old_cross_code_context), 3*cut_num)
                
                logging.debug(f"crossfile old code remove {cut_num} nodes")
                old_cross_code_context = old_cross_code_context[:-cut_range]+random.sample(old_cross_code_context[-cut_range:], cut_range-cut_num)
        
        if len(old_cross_code_context) + len(new_cross_code_context) > max_cross_num:
            # 删除多的新crossfile节点
            code_num = max_cross_num - len(old_cross_code_context)
            if code_num<=0:
                logging.debug(f"new crossfile remove all nodes")
                new_cross_code_context = []
                
            elif code_num<len(new_cross_code_context):
                cut_num = len(new_cross_code_context) - code_num
                cut_range = min(len(new_cross_code_context), 3*cut_num)
                logging.debug(f"crossfile new code remove {cut_num} nodes")
                new_cross_code_context = new_cross_code_context[:-cut_range] + random.sample(new_cross_code_context[-cut_range:], cut_range-cut_num)
        
        cross_code_context = new_cross_code_context+old_cross_code_context
        
        all_context = file_context + cross_code_context + infile_context + inclass_context
        
        change = False
        for node in all_context:
            if node.namespace not in nodes_ns:
                change = True
                break
        
        if change == False:
            if len(cross_code_context) > self.max_crossfile_node_num:
                change = True
            for node in all_context:
                if len(node.vectors) == 0 or node.vectors_pos.shape[0] == 0:
                    qika = 1
        
        return all_context, change        

    def make_kv(self, node_parts, extend_parts):
        pass
        
    def generate_step(self, 
                        target_namespace, 
                        input_ids: torch.Tensor, 
                        prefix_kv, 
                        prefix_pos,
                        init_context_nodes: List[str]
                    ) -> Tuple[torch.Tensor, List[ContextNode]]:
        """
        执行一步生成,包含逐层深入的上下文搜索
        返回生成的token_id和使用的上下文节点
        """
        curr_context = [self.context_dict[ns] for ns in init_context_nodes]
        layer_output_hidden = [None]*self.max_layer
        attn_scores = None
        max_position = 0
        max_length = 0
        spread_turn = 0
        seen_context = []
        input_k = [None]*self.max_layer
        input_v = [None]*self.max_layer
        input_pos = [None]*self.max_layer
        context_hiddens = []
        past_key_values:DynamicCache = DynamicCache()
        node_parts = None
        extend_parts = None
        encode_parts = None
        encode_ids_parts = None
        encode_mask_parts = None
        change_flag = True
        type_attn_by_layer = []
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        begin_pos = 0
        if prefix_kv:
            begin_pos = prefix_kv.key_cache[0].shape[2]
        
        
        # 逐层处理
        start_layer_idx = 0
        end_layer_idx = 0
        all_mask = None
        all_hiddens = None
        
        while start_layer_idx < self.max_layer:
            end_layer_idx = min(start_layer_idx+9, self.max_layer)
            # 去除无效上下文
            if not node_parts:
                curr_context = [node for node in curr_context if len(node.content) > 0]
                node_parts = self.cluster_brothers(curr_context, self.target_node)
                extend_parts = self.extend_nodeseq([node for part in node_parts for node in part])
                """for part in node_parts:
                    extend_parts.append(self.extend_nodeseq(part))"""
            
            # 将全部ids切分组织成适合输入的大小
            if start_layer_idx == 0:
                encode_parts, encode_ids_parts, encode_mask_parts = self.cut_to_encode(node_parts, extend_parts)
                context_hiddens = []
            # 收集上下文kv
            now_pos = 0
            
            past_key_values:DynamicCache = DynamicCache()
            hidden_pos = 0
            if start_layer_idx > 0:
                all_hiddens = context_hiddens
                context_hiddens = []
            else:
                all_hiddens = list(itertools.chain.from_iterable(encode_ids_parts))
                all_hiddens = torch.tensor(all_hiddens, device=self.device)
                all_hiddens = self.encode_init_hidden(all_hiddens).squeeze(0)
                all_mask = torch.tensor(list(itertools.chain.from_iterable(encode_mask_parts)), device=self.device).bool()
            
            for idx, part in enumerate(encode_parts):
                part_length = len(encode_mask_parts[idx])
                hidden = all_hiddens[now_pos:now_pos+part_length]
                mask = all_mask[now_pos:now_pos+part_length]
                
                part_key, part_value, part_hidden = self.encode_by_layer(hidden, start_layer_idx, end_layer_idx-1, now_pos)
                now_pos += part_key.shape[2]
                key = part_key[:,:,mask]
                value = part_value[:,:,mask]
                for layer in range(start_layer_idx, end_layer_idx):
                    past_key_values.update(key[layer-start_layer_idx].unsqueeze(0), value[layer-start_layer_idx].unsqueeze(0), layer)
                context_hiddens.append(part_hidden[end_layer_idx-1-start_layer_idx])
                # del hidden, part_key, part_value, part_hidden
                # del key, value, mask
            context_hiddens = torch.cat(context_hiddens, dim = 0)
                

            """encode_part = []
            encode_extend_part = []
            encode_mask = []
            encode_part_length = 0
            new_node_parts = []
            new_extend_parts = []
            for part_idx, node_list in enumerate(node_parts):
                if node_list[0].type in ['file', 'folder']:
                    # 对于文件和文件夹级part，切分成不超过上限的subpart
                    extend_list = [node.namespace for node in extend_parts[part_idx]]
                    key_tensor, value_tensor, extend_list = self.encode_fileseq(extend_list, begin_layer=start_layer_idx, end_layer=end_layer_idx-1, now_pos=0)
                    node_list = [node.namespace for node in node_list]
                else:
                    # 对于代码级part，拼接part直到到达上下文窗口尺寸
                    extend_part = [node.namespace for node in extend_parts[part_idx]]
                    encode_extend_part.extend(extend_part)
                    encode_part.extend(node_list)
                    over_flag = False
                    if part_idx+1 < len(node_parts):
                        next_part_length = sum([self.node_length(node) for node in extend_parts[part_idx+1]])
                        if encode_part_length+next_part_length > self.max_context_length-self.max_node_length:
                            over_flag = True
                        else:
                            encode_part_length += next_part_length
                    else:
                        over_flag = True
                    if over_flag:
                        key_tensor, value_tensor, extend_list = self.encode_nodeseq(encode_extend_part, begin_layer=start_layer_idx, end_layer=end_layer_idx-1, now_pos=0)
                        
                        node_list = [node.namespace for node in encode_part]
                        # 开始下一轮收集
                        encode_extend_part = []
                        encode_part = []
                        encode_part_length = 0
                    else:
                        continue
                        
                
                part_key, part_value, new_node_list = [], [], []
                for idx, node in enumerate(extend_list):
                    self.context_dict[node.namespace] = node
                    if node.namespace in node_list:
                        part_key.append(node.vectors[0][start_layer_idx: end_layer_idx])
                        part_value.append(node.vectors[1][start_layer_idx: end_layer_idx])
                        new_node_list.append(node)
                part_key = torch.cat(part_key, dim=2)
                part_value = torch.cat(part_value, dim=2)
                node_list = new_node_list
                new_node_parts.append(node_list)
                new_extend_parts.append(extend_list)    
                     
                # 从位置0开始向右延伸
                part_key = self.shift_pos(part_key, now_pos)
                now_pos += key_tensor.shape[2]
                
                for layer_idx in range(start_layer_idx, end_layer_idx):
                    past_key_values.update(part_key[layer_idx-start_layer_idx].unsqueeze(0).to(self.device), part_value[layer_idx-start_layer_idx].unsqueeze(0).to(self.device), layer_idx)
                del key_tensor, value_tensor, part_key, part_value
            node_parts = new_node_parts
            extend_parts = new_extend_parts"""
            
            curr_context = list(itertools.chain.from_iterable(node_parts))
           
            if prefix_kv is not None and len(prefix_kv.key_cache) >= end_layer_idx:
                for layer_idx in range(start_layer_idx, end_layer_idx):
                    prefix_k = self.shift_pos(prefix_kv.key_cache[layer_idx], now_pos)
                    prefix_v = prefix_kv.value_cache[layer_idx]
                    past_key_values.update(prefix_k, prefix_v, layer_idx)
            
            if start_layer_idx != 0:
                past_key_values.update(past_key_values.key_cache[start_layer_idx], past_key_values.value_cache[start_layer_idx], 0)
            
            # 编码位置信息和掩码 输入的起始位置为上下文总长度+缓存输入的长度
            cache_position = torch.arange(
                now_pos+begin_pos, now_pos+begin_pos+inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if cache_position[-1] >= 32768:
                logging.info("超出32768的上下文尺寸")
            position_ids = cache_position.unsqueeze(0)
            max_position = max(max_position, cache_position[0])
            max_length = past_key_values._seen_tokens
            causal_mask = self.model._update_causal_mask(
                    None, inputs_embeds, cache_position, past_key_values, output_attentions=True
                )
            
            # 从start_layer_idx到end_layer_idx-1层运算
            for layer_idx in range(start_layer_idx, end_layer_idx):
            
                # 编码当前层输入特征
                layer = self.model.layers[layer_idx]
                if layer_idx == 0:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = layer_output_hidden[layer_idx-1]

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
                
                # layer_attn = self.visualize_attention(self.target_node, curr_context, layer_outputs[1].detach().to("cpu"), layer_idx)
                # type_attn_by_layer.append(layer_attn)
                layer_output_hidden[layer_idx] = layer_outputs[0]
                
                # 记录当前输入的kv:
                input_k[layer_idx] = self.shift_pos(layer_outputs[2].key_cache[layer_idx][:,:,-hidden_states.shape[1]:], -now_pos)
                # input_k[layer_idx] = layer_outputs[2].key_cache[layer_idx][:,:,-hidden_states.shape[1]:]
                input_v[layer_idx] = layer_outputs[2].value_cache[layer_idx][:,:,-hidden_states.shape[1]:]
                input_pos[layer_idx] = now_pos
                
                if change_flag and layer_idx > 2 and layer_idx <= self.spread_layer:
                    if attn_scores == None:
                        attn_scores = layer_outputs[1].detach().to("cpu")
                    else:
                        attn_scores = torch.cat([attn_scores, layer_outputs[1].detach().to("cpu")], dim=1)
                
                # 释放显存
                del layer_outputs
                
                if layer_idx in [0, 35]:
                    qika = 1
            
                if change_flag and layer_idx in [self.spread_layer]:  # layer_idx%self.past_layers == self.past_layers-1:
                    # 选择高注意力节点
                    curr_context = [node.namespace for node in curr_context]
                    high_attn_nodes, change_flag = self.select_high_attention_nodes(self.target_node, curr_context, attn_scores, min_num=16)    
                    
                    spread_turn += 1
                    if spread_turn >= self.max_spread_turn:
                        logging.info("到达展开轮次上限{}".format(self.max_spread_turn))
                        change_flag = False
                    node_parts = self.cluster_brothers(high_attn_nodes, self.target_node)
                    curr_context = [node for node_part in node_parts for node in node_part]
                    extend_parts = self.extend_nodeseq(curr_context)
                    
                    # 分类输出节点数量
                    folder_num = len([node for node in curr_context if node.type == 'folder'])
                    file_num = len([node for node in curr_context if node.type == 'file'])
                    code_num = len([node for node in curr_context if node.type not in ['file', "folder"]])
                    logging.debug(f"get {folder_num} folders, {file_num} files, {code_num} codes")
                    
                    # type_attn_by_layer = type_attn_by_layer[:self.begin_layer]
                    attn_scores = None
                    end_layer_idx = 0
                    break
                
                pass
            
                
            del past_key_values, causal_mask
            del cache_position, position_embeddings, position_ids
            # torch.cuda.empty_cache()
            
            start_layer_idx = end_layer_idx
            pass
            
        # 保存当前步kv
        if prefix_kv is None:
            prefix_kv = DynamicCache()
            prefix_pos = [None]*self.max_layer

        for layer_idx in range(self.max_layer):
            prefix_kv.update(input_k[layer_idx], input_v[layer_idx], layer_idx)
            prefix_pos[layer_idx] = input_pos[layer_idx]
        del input_k, input_v
        
        # 计算最终输出
        hidden_states = layer_output_hidden[layer_idx-1]
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # 使用类方法进行采样
        next_token = self.sample_next_token(logits[:, -1])
        
        # type_attn_by_layer = torch.stack(type_attn_by_layer, dim=0)
        
        seen_context = []
        for node in curr_context:
            if node.type == "code":
                node = self.context_dict[node.parent]
            if node.type in ['function', 'class'] and node.namespace not in seen_context:
                seen_context.append(node.namespace)
        info_dict = {
            "logits": logits[:, -1],
            "next_token": next_token,
            "curr_context": curr_context,
            "seen_context": seen_context,
            "prefix_kv": prefix_kv,
            "prefix_pos": prefix_pos,
            "type_attn_by_layer": type_attn_by_layer,
            "position": max_position,
            "length": max_length,
        }
        
        return info_dict

