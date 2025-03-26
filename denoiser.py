# diffusion_timeseries/denoiser.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectDenoiserSequenceWithRisk(nn.Module):
    def __init__(self, feature_dim, time_embed_dim, cond_dim, hidden_dim, num_layers=2):
        """
        组合 LSTM 网络，用于直接预测 x0 与风险指标 log_sigma。
        """
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