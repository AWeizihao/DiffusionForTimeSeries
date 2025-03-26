# 参数配置字典
config = {
    "cond_len": 60, # 条件数据长度（模型记忆长度）
    "seq_len": 10,  # 需要模型预测的序列长度
    "target_dim": 9, # 需要预测的维度，需要与 target_fields 数量一致，或与 data 列数一致（不含索引列）
    "cond_input_dim": 9, # 输入数据的维度，需要与 target_fields 数量一致，或与 data 列数一致（不含索引列）
    "cond_dim": 32,  # 条件数据嵌入维度
    "time_embed_dim": 32,  # 时间嵌入维度
    "hidden_dim": 64,  # 隐藏层神经元个数
    "T_diffusion": 100, # 扩散步骤数
    "lr": 0.001,       # 学习率
    "weight_decay": 0.0001, # # 权重衰减
    "n_epochs": 8000, # 训练轮数
    "volatility": 0.5,  # 波动率
    "use_dynamic_beta": True,  # True 则使用余弦 Beta 计划，False 则为线性计划
    "s_base": 0.008, # 基础 S 值
    "alpha": 1.0, # alpha 参数
    "use_compound_poisson": True, # 是否使用复合泊松过程
    "jump_intensity": 0.05, # 跳跃强度，置为 0 则关闭跳跃扩散
    "jump_scale": 1.0, # 跳跃尺度
    "ddim_steps": 50, # DDIM 反扩散步骤数
    "eta": 0.1
}

from trainer import train_model

# 自行载入数据 data ，然后运行本程序即可。

# data 推荐格式为 Pandas DataFrame。如果使用 Pandas DataFrame 格式数据，请将时间列（如果有）设置为索引或直接删去。
# data 如使用字典、numpy数组，会尝试转化为 Pandas DataFrame。

target_fields=["log_return", "volume", "turnover", "high", "low", "EMA_20", "EMA_50", "RSI_14", "MACD"]

# 注意 target_fields 列表必须是 Pandas DataFrame 的列名（不包括索引），如果不匹配，则只会使用 target_fields 字段与 Pandas DataFrame 的列名重合的部分列。
# 为避免问题，可以直接不传入 target_field 参数（target_field 是 train_model 的可选参数）。

model = train_model(config,data)