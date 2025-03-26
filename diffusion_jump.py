import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import argparse

# 假设以下模块已实现数据加载和评估
import All_methods.file_operations as imp
import All_methods.assess as aser

###############################################
# 1. 动态噪声调度函数
###############################################
def get_dynamic_beta_schedule(T, volatility, s_base=0.008, alpha=1.0):
    """
    根据传入的波动率调整余弦调度参数，生成 beta 值序列。
    
    参数:
      T: 扩散步数
      volatility: 代表数据波动率的数值（例如 log_return 的标准差）
      s_base: 基础 s 参数
      alpha: 控制波动率对 s 的影响因子
    
    返回:
      正则化后的 beta 序列张量
    """
    # 动态调整 s 参数
    s = s_base * (1 + alpha * volatility)
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

# 原始静态调度函数（保留备用）
def get_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

###############################################
# 自适应跳跃参数模块（可选，当不使用复合泊松时启用）
###############################################
class AdaptiveJumpModule(nn.Module):
    def __init__(self, cond_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 输出两个数，分别对应 jump_prob 和 jump_scale
        )
    
    def forward(self, h):
        # h: [B, cond_dim]
        out = self.fc(h)  # [B, 2]
        jump_prob = torch.sigmoid(out[:, 0])  # 保证在 [0, 1] 之间
        jump_scale = F.softplus(out[:, 1])      # 保证为正数
        return jump_prob, jump_scale

###############################################
# 2. 扩散模块（增加动态噪声调度和复合泊松跳跃过程）
###############################################
class Diffusion(nn.Module):
    def __init__(self, T, volatility=0.0, use_dynamic_beta=True, s_base=0.008, alpha=1.0,
                 use_compound_poisson=False, jump_intensity=0.1, jump_scale=1.0):
        """
        参数:
          T: 扩散步数
          volatility: 用于动态噪声调度的波动率指标
          use_dynamic_beta: 是否使用动态调整的 beta 调度
          s_base, alpha: 动态调度参数
          use_compound_poisson: 是否使用复合泊松过程建模跳跃
          jump_intensity: 跳跃发生的强度（泊松过程参数 λ）
          jump_scale: 跳跃幅度尺度（用于生成正态分布跳跃幅度）
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
        根据给定扩散步 t 对输入 x0 进行加噪采样。
        当 use_compound_poisson 为 True 时，额外添加复合泊松跳跃噪声；否则可选用自适应跳跃模块生成的跳跃噪声。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t]
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.view(1)
        # 标准扩散采样
        x_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1) * x0 + torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1) * noise

        # 添加跳跃噪声
        if self.use_compound_poisson:
            batch_size = x0.shape[0]
            # 对于每个样本，按照泊松过程采样跳跃次数（dt 假设为1）
            n_jumps = torch.poisson(torch.full((batch_size, 1, 1), self.jump_intensity, device=x0.device))
            # 当 n_jumps > 0 时，相当于聚合了 n_jumps 次跳跃，累积噪声标准差为 sqrt(n_jumps)*jump_scale
            jump_noise = torch.sqrt(n_jumps) * self.jump_scale * torch.randn_like(x0[:, :, 0:1])
            # 这里对第0通道加入复合泊松跳跃噪声
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

###############################################
# 3. Sinusoidal 时间嵌入（保持不变）
###############################################
def get_sinusoidal_embedding(t, embed_dim):
    t = torch.tensor([t], dtype=torch.float) if not torch.is_tensor(t) else t.float()
    half_dim = embed_dim // 2
    emb_constant = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb_constant).to(t.device)
    emb = t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

###############################################
# 4. 多尺度 Transformer 条件编码器（保持不变）
###############################################
class MultiScaleConditionEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=64, num_layers=2, nhead=4, dropout=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 细尺度分支：Transformer Encoder
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 粗尺度分支：1D 卷积
        self.conv_coarse = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1)
        # 融合后映射到 cond_dim
        self.fc_merge = nn.Linear(hidden_dim * 2, cond_dim)
    
    def forward(self, x):
        # x: [B, seq_len, input_dim]
        B, seq_len, _ = x.size()
        # 细尺度分支：Transformer
        fine = self.fc_in(x)          
        fine = fine.transpose(0, 1)     
        fine_transformed = self.transformer_encoder(fine)
        fine_transformed = fine_transformed.transpose(0, 1)  
        fine_feat = fine_transformed.mean(dim=1)             
        # 粗尺度分支：1D 卷积
        coarse_in = x.transpose(1, 2)   
        coarse_feature = self.conv_coarse(coarse_in)  
        coarse_feat = coarse_feature.mean(dim=2)       
        fused = torch.cat([fine_feat, coarse_feat], dim=-1)  
        cond = self.fc_merge(fused)  
        return cond

###############################################
# 5. 时间嵌入模块（保持不变）
###############################################
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, t):
        t_embed = get_sinusoidal_embedding(t, self.linear.in_features).to(t.device)
        return self.dropout(self.activation(self.linear(t_embed)))

###############################################
# 6. 去噪网络：直接预测 x0 和风险指标（保持不变）
###############################################
class DirectDenoiserSequenceWithRisk(nn.Module):
    def __init__(self, feature_dim, time_embed_dim, cond_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_embed_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(feature_dim + time_embed_dim + cond_dim,
                            hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(hidden_dim, feature_dim * 2)
    
    def forward(self, x_t, t_embed, h):
        B, seq_len, _ = x_t.shape
        t_proj = self.time_embed_proj(t_embed)           
        t_proj_seq = t_proj.unsqueeze(1).repeat(1, seq_len, 1)  
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)        
        x_concat = torch.cat([x_t, t_proj_seq, h_seq], dim=-1)  
        lstm_out, _ = self.lstm(x_concat)  
        lstm_out = self.dropout(lstm_out)
        out = self.fc_out(lstm_out)           
        x0_pred, log_sigma = torch.chunk(out, 2, dim=-1)
        return x0_pred, log_sigma

###############################################
# 7. 扩散时间序列模型：联合预测与风险测度（更新以支持新功能）
###############################################
class DiffusionTimeSeriesModelRiskMultiScale(nn.Module):
    def __init__(self, seq_len, feature_dim, cond_input_dim, cond_dim, time_embed_dim,
                 hidden_dim, T, volatility=0.0, use_dynamic_beta=True, s_base=0.008, alpha=1.0,
                 use_compound_poisson=False, jump_intensity=0.1, jump_scale=1.0):
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
        # 根据是否启用复合泊松过程决定跳跃噪声的添加方式
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
        device = c.device
        batch_size = c.shape[0]
        # 初始状态为噪声
        x = torch.randn(batch_size, self.seq_len, self.feature_dim, device=device)
        # 生成采样时间步（降序排列，相当于从 T 到 0）
        ddim_timesteps = np.linspace(0, self.T_diffusion - 1, ddim_steps, dtype=int)[::-1]
        h = self.cond_encoder(c)
        
        if self.diffusion.use_compound_poisson:
            jump_prob = None
            jump_scale_adaptive = None
        else:
            jump_prob, jump_scale_adaptive = self.jump_adaptive_module(h)
        
        last_log_sigma = None
        for i in range(len(ddim_timesteps) - 1):
            t_val = ddim_timesteps[i]
            t_next = ddim_timesteps[i+1]
            
            # 获取当前时间步的条件嵌入
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            t_embed = self.time_embed(t_tensor)
            
            # 通过去噪网络预测 x0 和风险指标
            x0_pred, log_sigma = self.denoiser(x, t_embed, h)
            last_log_sigma = log_sigma
            
            # 当前 alpha_bar
            alpha_bar_t = self.diffusion.alpha_bars[t_val]
            if alpha_bar_t.dim() == 0:
                alpha_bar_t = alpha_bar_t.view(1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            # 计算预测的噪声
            eps_pred = (x - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

            # 下一时间步 alpha_bar (对应 t-1)
            alpha_bar_t_next = self.diffusion.alpha_bars[t_next]
            if alpha_bar_t_next.dim() == 0:
                alpha_bar_t_next = alpha_bar_t_next.view(1)
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
            
            # 按照 DDIM 公式计算 sigma_t（跳跃噪声幅度）
            sigma_t = eta * torch.sqrt(
                ((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * (1 - alpha_bar_t / alpha_bar_t_next)
            )
            # 计算确定性部分放缩系数
            sqrt_coef = torch.sqrt(1 - alpha_bar_t_next - sigma_t ** 2)
            
            # 采样新 x（如果 eta 为 0，则 sigma_t = 0，不添加额外噪声）
            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0
            
            x = sqrt_alpha_bar_t_next * x0_pred + sqrt_coef * eps_pred + sigma_t * noise
        
        risk_sigma = torch.exp(0.5 * last_log_sigma)
        return x, risk_sigma

###############################################
# 8. 稳健异方差损失函数（保持不变）
###############################################
def robust_heteroscedastic_loss(x0_pred, log_sigma, x0_true):
    huber_loss = F.smooth_l1_loss(x0_pred, x0_true, reduction='none')
    loss = 0.5 * (torch.exp(-log_sigma) * huber_loss + log_sigma)
    return loss.mean()

###############################################
# 9. 数据加载函数（保持不变）
###############################################
def load_train_test_data(start_date, end_date, cond_len, seq_len, target_fields):
    ts_data = imp.load_entire_data('600004', start_date, end_date).dropna()
    data = ts_data[target_fields].values.astype(np.float32)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-6
    norm_data = (data - mean) / std

    split_idx = int(len(norm_data) * 0.8)
    train_data = norm_data[:split_idx]
    test_data = norm_data[split_idx - (cond_len + seq_len) + 1:]
    
    n_samples_train = len(train_data) - (cond_len + seq_len) + 1
    train_c_list = [train_data[i:i+cond_len] for i in range(n_samples_train)]
    train_y_list = [train_data[i+cond_len:i+cond_len+seq_len] for i in range(n_samples_train)]
    
    n_samples_test = len(test_data) - (cond_len + seq_len) + 1
    test_c_list = [test_data[i:i+cond_len] for i in range(n_samples_test)]
    test_y_list = [test_data[i+cond_len:i+cond_len+seq_len] for i in range(n_samples_test)]
    
    train_c_tensor = torch.tensor(train_c_list, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y_list, dtype=torch.float32)
    test_c_tensor = torch.tensor(test_c_list, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y_list, dtype=torch.float32)
    
    return train_c_tensor, train_y_tensor, test_c_tensor, test_y_tensor, mean, std

###############################################
# 10. 滚动窗口验证（保持不变）
###############################################
def rolling_window_validation(model, test_c, test_y, device, ddim_steps=50, eta=0):
    model.eval()
    predictions_list = []
    risk_list = []
    with torch.no_grad():
        for i in range(test_c.size(0)):
            cond_sample = test_c[i].unsqueeze(0).to(device)
            pred, risk = model.sample_ddim(cond_sample, ddim_steps=ddim_steps, eta=eta)
            predictions_list.append(pred.cpu())
            risk_list.append(risk.cpu())
    predictions = torch.cat(predictions_list, dim=0)
    risks = torch.cat(risk_list, dim=0)
    return predictions, risks

###############################################
# 11. 主训练与评测流程（注意传入 volatility 和是否启用复合泊松）
###############################################
def main():
    config = {
        "start_date": "2015-01-01",
        "end_date": "2024-12-31",
        "cond_len": 60,
        "seq_len": 10,
        "target_fields": ["log_return", "volume", "turnover","high","low",'EMA_20', 'EMA_50', 'RSI_14','MACD'],
        "target_dim": 9,
        "cond_input_dim": 9,
        "cond_dim": 32,
        "time_embed_dim": 32,
        "hidden_dim": 64,
        "T_diffusion": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "n_epochs": 8000,
        "volatility": 0.5,          # 根据数据计算或预设，例如取 log_return 的标准差
        "use_dynamic_beta": True,
        "s_base": 0.008,
        "alpha": 1.0,
        "use_compound_poisson": True,   # 开启复合泊松过程
        "jump_intensity": 0.05,         # 每步期望跳跃次数，可根据业务调整
        "jump_scale": 1.0,              # 跳跃噪声标准差
    }
    
    train_c, train_y, test_c, test_y, data_mean, data_std = load_train_test_data(
        config["start_date"], config["end_date"],
        config["cond_len"], config["seq_len"],
        config["target_fields"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_c, train_y = train_c.to(device), train_y.to(device)
    test_c, test_y = test_c.to(device), test_y.to(device)
    
    x_dim = config["target_dim"]
    
    model = DiffusionTimeSeriesModelRiskMultiScale(
        config["seq_len"],
        x_dim,
        config["cond_input_dim"],
        config["cond_dim"],
        config["time_embed_dim"],
        config["hidden_dim"],
        config["T_diffusion"],
        volatility=config["volatility"],
        use_dynamic_beta=config["use_dynamic_beta"],
        s_base=config["s_base"],
        alpha=config["alpha"],
        use_compound_poisson=config["use_compound_poisson"],
        jump_intensity=config["jump_intensity"],
        jump_scale=config["jump_scale"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for epoch in range(config["n_epochs"]):
        t = torch.randint(1, config["T_diffusion"], (train_c.shape[0],), device=device)
        optimizer.zero_grad()
        x0 = train_y  
        with torch.cuda.amp.autocast():
            x0_pred, log_sigma = model(x0, train_c, t)
            loss = robust_heteroscedastic_loss(x0_pred, log_sigma, x0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']}, Robust Heteroscedastic Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        predictions_ddim, risk_sigma = model.sample_ddim(test_c, ddim_steps=50, eta=0.1)
        predictions_ddim = predictions_ddim.view(test_y.shape[0], config["seq_len"], config["target_dim"])
        risk_sigma = risk_sigma.view(test_y.shape[0], config["seq_len"], config["target_dim"])
        metrics_ddim = aser.evaluate_financial_performance_with_coverage(
            predictions_ddim, test_y, risk_sigma, data_mean, data_std
        )
        print("Evaluation Metrics (整体DDIM采样):")
        for key, value in metrics_ddim.items():
            print(f"{key}: {value:.4f}")
    
    rolling_preds, rolling_risks = rolling_window_validation(model, test_c, test_y, device,
                                                               ddim_steps=50, eta=0.1)
    metrics_rolling = aser.evaluate_financial_performance_with_coverage(
        rolling_preds, test_y, rolling_risks, data_mean, data_std
    )
    print("Evaluation Metrics (滚动窗口验证):")
    for key, value in metrics_rolling.items():
        print(f"{key}: {value:.4f}")
    
    torch.save(model, "diffusion_model_multi_risk_multiscale_transformer.pth")

if __name__ == '__main__':
    main()