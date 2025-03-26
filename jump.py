# diffusion_timeseries/jump.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveJumpModule(nn.Module):
    def __init__(self, cond_dim, hidden_dim=32):
        """
        用于非复合泊松情况下生成跳跃噪声概率和跳跃尺度。
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出 jump_prob 和 jump_scale
        )
    
    def forward(self, h):
        out = self.fc(h)
        jump_prob = torch.sigmoid(out[:, 0])
        jump_scale = F.softplus(out[:, 1])
        return jump_prob, jump_scale