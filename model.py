# diffusion_timeseries/model.py

import torch
import torch.nn as nn
from diffusion import Diffusion
from encoder import MultiScaleConditionEncoder
from embeddings import TimeEmbedding
from denoiser import DirectDenoiserSequenceWithRisk
from jump import AdaptiveJumpModule

class DiffusionTimeSeriesModelRiskMultiScale(nn.Module):
    def __init__(self, seq_len, feature_dim, cond_input_dim, cond_dim, time_embed_dim,
                 hidden_dim, T, volatility=0.0, use_dynamic_beta=True, s_base=0.008, alpha=1.0,
                 use_compound_poisson=False, jump_intensity=0.1, jump_scale=1.0):
        """
        整体扩散时间序列模型：
          - 将数据的条件信息通过多尺度编码器提取
          - 根据配置选择是否应用复合泊松跳跃或自适应跳跃模块
          - 内部使用 DDIM 采样，可通过 eta 参数控制随机性
          
        参数均对用户开放，可灵活设置：
          如 beta 序列生成方式、是否添加跳跃噪声、以及相关超参数。
        """
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.T_diffusion = T
        self.diffusion = Diffusion(T, volatility, use_dynamic_beta, s_base, alpha,
                                   use_compound_poisson, jump_intensity, jump_scale)
        self.cond_encoder = MultiScaleConditionEncoder(cond_input_dim, cond_dim,
                                                       hidden_dim, num_layers=2, nhead=4, dropout=0.1)
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.denoiser = DirectDenoiserSequenceWithRisk(feature_dim, time_embed_dim, cond_dim, hidden_dim, num_layers=2)
        self.jump_adaptive_module = AdaptiveJumpModule(cond_dim, hidden_dim=hidden_dim//2)
    
    def forward(self, x0, c, t):
        h = self.cond_encoder(c)
        if self.diffusion.use_compound_poisson:
            noise = torch.randn_like(x0)
            x_t = self.diffusion.q_sample(x0, t, noise=noise)
        else:
            jump_prob, jump_scale_adaptive = self.jump_adaptive_module(h)
            noise = torch.randn_like(x0)
            x_t = self.diffusion.q_sample(x0, t, noise=noise,
                                           jump_prob=jump_prob, jump_scale_adaptive=jump_scale_adaptive)
        t_embed = self.time_embed(t)
        x0_pred, log_sigma = self.denoiser(x_t, t_embed, h)
        return x0_pred, log_sigma
    
    def sample_ddim(self, c, ddim_steps=50, eta=0.0):
        """
        使用 DDIM 采样生成样本，eta 参数控制随机性（eta=0 时确定性采样）。
        
        更新公式：
          sigma_t = eta * sqrt(((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * (1 - alpha_bar_t / alpha_bar_{t-1}))
          x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1} - sigma_t^2)*eps_pred + sigma_t * z
        """
        device = c.device
        batch_size = c.shape[0]
        x = torch.randn(batch_size, self.seq_len, self.feature_dim, device=device)
        ddim_timesteps = torch.linspace(0, self.T_diffusion - 1, steps=ddim_steps, dtype=torch.long).flip(0)
        h = self.cond_encoder(c)
        
        if self.diffusion.use_compound_poisson:
            jump_prob = None
            jump_scale_adaptive = None
        else:
            jump_prob, jump_scale_adaptive = self.jump_adaptive_module(h)
        
        last_log_sigma = None
        for i in range(len(ddim_timesteps) - 1):
            t_val = ddim_timesteps[i].item()
            t_next = ddim_timesteps[i+1].item()
            
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            t_embed = self.time_embed(t_tensor)
            x0_pred, log_sigma = self.denoiser(x, t_embed, h)
            last_log_sigma = log_sigma
            
            alpha_bar_t = self.diffusion.alpha_bars[t_val]
            if alpha_bar_t.dim() == 0:
                alpha_bar_t = alpha_bar_t.view(1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            eps_pred = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
            
            alpha_bar_t_next = self.diffusion.alpha_bars[t_next]
            if alpha_bar_t_next.dim() == 0:
                alpha_bar_t_next = alpha_bar_t_next.view(1)
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
            
            sigma_t = eta * torch.sqrt(((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) *
                                       (1 - alpha_bar_t / alpha_bar_t_next))
            sqrt_coef = torch.sqrt(1 - alpha_bar_t_next - sigma_t ** 2)
            
            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0
            
            x = sqrt_alpha_bar_t_next * x0_pred + sqrt_coef * eps_pred + sigma_t * noise
        
        risk_sigma = torch.exp(0.5 * last_log_sigma)
        return x, risk_sigma