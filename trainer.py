# diffusion_timeseries/trainer.py

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from load_data import load_train_test_data
from loss import robust_heteroscedastic_loss
from model import DiffusionTimeSeriesModelRiskMultiScale

import All_methods.assess as aser  # 假设已有的评估模块

def train_model(config,input,target_fields=None):
    """
    根据配置训练模型，配置中可控制：
      - 数据加载的相关参数（start_date, end_date, cond_len, seq_len, target_fields）
      - 模型超参数（T_diffusion、volatility、beta 调度方式、是否复合泊松、跳跃参数等）
      - 训练过程参数（lr, n_epochs, weight_decay 等）
      - 采样时 DDIM 参数（ddim_steps, eta）
    """
    train_c, train_y, test_c, test_y, data_mean, data_std = load_train_test_data(input,config["cond_len"], config["seq_len"],target_fields)
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
    scaler = GradScaler()
    
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
            print(f"Epoch {epoch+1}/{config['n_epochs']}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        predictions_ddim, risk_sigma = model.sample_ddim(test_c, ddim_steps=config.get("ddim_steps", 50), eta=config.get("eta", 0.0))
        predictions_ddim = predictions_ddim.view(test_y.shape[0], config["seq_len"], config["target_dim"])
        risk_sigma = risk_sigma.view(test_y.shape[0], config["seq_len"], config["target_dim"])
        metrics_ddim = aser.evaluate_financial_performance_with_coverage(
            predictions_ddim, test_y, risk_sigma, data_mean, data_std
        )
        print("Evaluation Metrics (DDIM Sampling):")
        for key, value in metrics_ddim.items():
            print(f"{key}: {value:.4f}")
    
    return model

def rolling_window_validation(model, test_c, test_y, device, ddim_steps=50, eta=0):
    """
    使用滚动窗口验证，在每个测试样本上运行采样
    """
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