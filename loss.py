# diffusion_timeseries/losses.py

import torch
import torch.nn.functional as F

def robust_heteroscedastic_loss(x0_pred, log_sigma, x0_true):
    """
    稳健异方差损失函数，结合 Huber 损失与预测风险.
    """
    huber_loss = F.smooth_l1_loss(x0_pred, x0_true, reduction='none')
    loss = 0.5 * (torch.exp(-log_sigma) * huber_loss + log_sigma)
    return loss.mean()