# diffusion_timeseries/embeddings.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoidal_embedding(t, embed_dim):
    """
    获取正弦/余弦时间嵌入。
    """
    t = torch.tensor([t], dtype=torch.float) if not torch.is_tensor(t) else t.float()
    half_dim = embed_dim // 2
    emb_constant = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb_constant).to(t.device)
    emb = t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        """
        时间嵌入模块，包含线性变换、激活与 Dropout。
        """
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, t):
        t_embed = get_sinusoidal_embedding(t, self.linear.in_features).to(t.device)
        return self.dropout(self.activation(self.linear(t_embed)))