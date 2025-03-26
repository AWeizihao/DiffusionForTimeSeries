import torch
import numpy as np
import pandas as pd
import warnings

def load_train_test_data(input, cond_len, seq_len, target_fields=None):
    """
    加载并预处理数据，返回训练和测试张量以及均值、标准差（用于反标准化）。

    参数:
      - input: 用户传入的原始数据，支持 pandas.DataFrame、numpy 数组或其它可转换为 DataFrame 的格式
      - cond_len: 历史数据长度（用于模型预测条件）
      - seq_len: 预测序列长度
      - target_fields: 可选参数，若不为 None，则只使用这些字段；如果为 None，则使用所有列

    返回:
      - train_c_tensor: 训练样本条件（历史数据）的 tensor
      - train_y_tensor: 训练样本目标输出序列的 tensor
      - test_c_tensor: 测试样本条件数据的 tensor
      - test_y_tensor: 测试样本目标输出序列的 tensor
      - mean: 用于归一化的均值
      - std: 用于归一化的标准差
    """
    
    # 如果输入数据不是 DataFrame，则尝试转换为 DataFrame
    if not isinstance(input, pd.DataFrame):
        try:
            input = pd.DataFrame(input)
        except Exception as Ex:
            raise ValueError("数据无法转换为 Pandas DataFrame。请使用 Pandas 序列、字典等数据格式。") from Ex

    # 删除缺失值
    ts_data = input.dropna().copy()
    
    # 如果指定了 target_fields，则先验证字段并只保留指定字段
    if target_fields:
        missing_fields = [f for f in target_fields if f not in ts_data.columns]
        last= [f for f in target_fields if f  in ts_data.columns]
        if missing_fields:
            raise Warning(f"输入数据中缺少字段: {missing_fields}")
        ts_data = ts_data[last]
    
    # 检查数据量是否足够
    if len(ts_data) < cond_len + seq_len:
        raise ValueError("数据量不足以生成指定长度的训练/测试样本")
    
    # 转换为 NumPy 数组并转换数据类型
    data = ts_data.to_numpy(dtype=np.float32)
    
    # 计算均值和标准差，并处理标准差为 0 的情况
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, 1e-6, std)
    norm_data = (data - mean) / std
    
    # 按 80% 分割训练和测试数据
    split_idx = int(len(norm_data) * 0.8)
    train_data = norm_data[:split_idx]
    test_data = norm_data[split_idx - (cond_len + seq_len) + 1:]
    
    # 构造训练样本
    n_samples_train = len(train_data) - (cond_len + seq_len) + 1
    train_c_list = [train_data[i:i + cond_len] for i in range(n_samples_train)]
    train_y_list = [train_data[i + cond_len:i + cond_len + seq_len] for i in range(n_samples_train)]
    
    # 构造测试样本
    n_samples_test = len(test_data) - (cond_len + seq_len) + 1
    test_c_list = [test_data[i:i + cond_len] for i in range(n_samples_test)]
    test_y_list = [test_data[i + cond_len:i + cond_len + seq_len] for i in range(n_samples_test)]
    
    # 转换为 torch 张量
    train_c_tensor = torch.tensor(train_c_list, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y_list, dtype=torch.float32)
    test_c_tensor = torch.tensor(test_c_list, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y_list, dtype=torch.float32)
    
    return train_c_tensor, train_y_tensor, test_c_tensor, test_y_tensor, mean, std