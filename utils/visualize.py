"""
统计绘图相关的函数
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt



def visualize_line(data: torch.tensor, label = [], save_path="/home/qikahh/projects/Structured_Code_Context/visualize.png"):
    """
    对于多维tensor 绘制折线图
    最后一维为x轴
    倒数第二维为不同样本，需要在它们之间计算均值和标准差
    如果有倒数第三维 则对应不同折线 名称为label里的对应元素
    """
    # 检查输入维度，不足的补全至3维，超过的从前方维度融合
    if len(data.shape) > 3:
        data = data.reshape(data.shape[-3:])
    while len(data.shape) < 3:
        data = data.unsqueeze(0)  
    
    # 检查label元素数量与data第一维长度是否一致
    if len(label) != data.shape[0]:
        label = [str(i) for i in range(data.shape[0])]
        
    # 沿倒数第二维进行平均和标准差计算
    mean = data.mean(dim=-2)
    std = data.std(dim=-2)
    
    # 建造画布
    plt.figure(figsize=(10, 6))
    
    for i in range(data.shape[0]):
        plt.plot(range(mean[i].shape[-1]), mean[i], label=label[i])
        plt.fill_between(range(mean[i].shape[-1]), mean[i] - std[i], mean[i] + std[i], alpha=0.3)
    
    plt.xlabel('idx')
    plt.ylabel('value')
    
    plt.legend()
    plt.savefig(save_path)
    pass
        


if __name__ == "__main__":
    data = torch.randn(3, 5, 10)
    label = ["a", "b", "c"]
    visualize_line(data, label)
    pass