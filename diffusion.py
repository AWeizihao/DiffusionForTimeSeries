# diffusion_timeseries/diffusion.py

import math
import torch
import torch.nn as nn

def get_dynamic_beta_schedule(T, volatility, s_base=0.008, alpha=1.0):
    """
    根据波动率调整余弦调度参数，生成 beta 序列。
    
    参数:
      T: 扩散步数
      volatility: 波动率指标（例如 log_return 的标准差）
      s_base: 基础 s 参数
      alpha: 控制波动率对 s 的影响因子
    
    返回:
      正则化后的 beta 张量
    """
    s = s_base * (1 + alpha * volatility)
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def get_beta_schedule(T, s=0.008):
    """
    原始静态 beta 调度函数。
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class Diffusion(nn.Module):
    def __init__(self, T, volatility=0.0, use_dynamic_beta=True, s_base=0.008, alpha=1.0,
                 use_compound_poisson=False, jump_intensity=0.1, jump_scale=1.0):
        """
        扩散模块：
          - 可选动态 beta 调度与静态 beta 调度
          - 可选添加复合泊松跳跃噪声或使用自适应跳跃模块
          
        参数主要暴露给用户：
          T: 扩散步数
          volatility: 波动率指标
          use_dynamic_beta: 是否使用动态 beta 调度函数（True/False）
          s_base, alpha: beta 调度参数控制因子
          use_compound_poisson: 是否使用复合泊松过程 (True/False)
          jump_intensity: 复合泊松跳跃参数 λ
          jump_scale: 跳跃尺度（标准差）
        """
        super().__init__()
        self.T = T
        if use_dynamic_beta:
            betas = get_dynamic_beta_schedule(T, volatility, s_base, alpha)
        else:
            betas = get_beta_schedule(T, s_base)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        
        # 复合泊松过程参数设置
        self.use_compound_poisson = use_compound_poisson
        self.jump_intensity = jump_intensity
        self.jump_scale = jump_scale
        
    def q_sample(self, x0, t, noise=None, jump_prob=None, jump_scale_adaptive=None):
        """
        根据当前扩散步 t 为输入 x0 添加噪声。
        当 use_compound_poisson 为 True 时，使用复合泊松跳跃噪声；
        否则根据 jump_prob 与 jump_scale_adaptive 添加自适应跳跃噪声。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t]
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1)
        x_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1) * x0 + torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1) * noise

        if self.use_compound_poisson:
            batch_size = x0.shape[0]
            n_jumps = torch.poisson(torch.full((batch_size, 1, 1), self.jump_intensity, device=x0.device))
            jump_noise = torch.sqrt(n_jumps) * self.jump_scale * torch.randn_like(x0[:, :, 0:1])
            x_t[:, :, 0:1] = x_t[:, :, 0:1] + jump_noise
        elif jump_prob is not None and jump_scale_adaptive is not None:
            batch_size = x0.shape[0]
            if isinstance(jump_prob, torch.Tensor):
                jump_prob = jump_prob.view(batch_size, 1, 1)
            else:
                jump_prob = torch.full((batch_size, 1, 1), jump_prob, device=x0.device)
            if isinstance(jump_scale_adaptive, torch.Tensor):
                jump_scale_adaptive = jump_scale_adaptive.view(batch_size, 1, 1)
            else:
                jump_scale_adaptive = torch.full((batch_size, 1, 1), jump_scale_adaptive, device=x0.device)
            jump_mask = (torch.rand(batch_size, 1, 1, device=x0.device) < jump_prob).float()
            jump_noise = torch.randn_like(x0[:, :, 0:1]) * jump_scale_adaptive
            x_t[:, :, 0:1] = x_t[:, :, 0:1] + jump_mask * jump_noise

        return x_t