# 在你的主程序中直接定义配置字典
config = {
    "cond_len": 60,
    "seq_len": 10,
    "target_fields": ["log_return", "volume", "turnover", "high", "low", "EMA_20", "EMA_50", "RSI_14", "MACD"],
    "target_dim": 9,
    "cond_input_dim": 9,
    "cond_dim": 32,
    "time_embed_dim": 32,
    "hidden_dim": 64,
    "T_diffusion": 100,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "n_epochs": 8000,
    "volatility": 0.5,
    "use_dynamic_beta": True,
    "s_base": 0.008,
    "alpha": 1.0,
    "use_compound_poisson": True,
    "jump_intensity": 0.05,
    "jump_scale": 1.0,
    "ddim_steps": 50,
    "eta": 0.1
}

# 然后直接调用训练函数
from trainer import train_model
import All_methods.file_operations as fo

data=fo.load_entire_data('600004','2022-01-01','2023-12-31')

model = train_model(config,data)