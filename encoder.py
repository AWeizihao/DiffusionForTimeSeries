# diffusion_timeseries/encoder.py

import torch
import torch.nn as nn

class MultiScaleConditionEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=64, num_layers=2, nhead=4, dropout=0.05):
        """
        利用 Transformer 与 1D 卷积提取细尺度与粗尺度特征，
        最后融合为条件向量。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv_coarse = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                     kernel_size=3, stride=2, padding=1)
        self.fc_merge = nn.Linear(hidden_dim * 2, cond_dim)
    
    def forward(self, x):
        # x: [B, seq_len, input_dim]
        B, seq_len, _ = x.size()
        fine = self.fc_in(x)
        fine = fine.transpose(0, 1)
        fine_transformed = self.transformer_encoder(fine)
        fine_transformed = fine_transformed.transpose(0, 1)
        fine_feat = fine_transformed.mean(dim=1)
        coarse_in = x.transpose(1, 2)
        coarse_feature = self.conv_coarse(coarse_in)
        coarse_feat = coarse_feature.mean(dim=2)
        fused = torch.cat([fine_feat, coarse_feat], dim=-1)
        cond = self.fc_merge(fused)
        return cond