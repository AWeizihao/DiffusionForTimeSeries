import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def evaluate_financial_performance(predictions, ground_truth, data_mean, data_std, 
                                   plot_results=True, sample_plots=5, target_channel=0):
    """
    评估模型的金融预测表现，仅针对特定通道（默认 log_return，对应第0个通道）。

    参数:
        predictions: 模型的预测输出，Tensor或ndarray。
                     单变量预测：形状为 [样本数, 序列长度]；
                     多变量预测：形状为 [样本数, 序列长度, 变量数]。
        ground_truth: 真实目标数据，Tensor或ndarray，形状与predictions一致。
        data_mean: 用于反标准化的均值，可为标量或与每个变量对应的向量。
        data_std: 用于反标准化的标准差，同上。
        plot_results: 是否绘制部分样本的预测结果 (默认为 True)。
        sample_plots: 最多绘制的样本数 (默认为 5)。
        target_channel: 指定需要评估的通道索引，默认 0（即 log_return）。
        
    返回:
        metrics: 包含评价指标的字典，包括 "MSE", "RMSE", "MAE", "Median_AE", "MAPE (%)",
                 "Directional Accuracy (%)", "Pearson Correlation", "R2 Score"
    """
    # 转换为 NumPy 数组（如果输入为 Tensor）
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().detach().numpy()

    # 反标准化数据
    predictions_real = predictions * data_std + data_mean
    ground_truth_real = ground_truth * data_std + data_mean

    # 如果是多变量（3D），则仅选择目标通道进行评估
    if predictions_real.ndim == 3:
        predictions_real = predictions_real[:, :, target_channel]
        ground_truth_real = ground_truth_real[:, :, target_channel]

    # 将数据展平以便计算全局指标
    pred_flat = predictions_real.reshape(-1)
    gt_flat = ground_truth_real.reshape(-1)
    
    epsilon = 1e-6  # 防止除以0的错误

    # 计算方向一致性 (Directional Accuracy)
    # 对于 2D 数据 (样本数, 序列长度)
    pred_direction = np.sign(np.diff(predictions_real, axis=1))
    gt_direction = np.sign(np.diff(ground_truth_real, axis=1))
    directional_accuracy = np.mean(pred_direction == gt_direction) * 100

    # 计算各项指标
    mse = np.mean((predictions_real - ground_truth_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_real - ground_truth_real))
    median_ae = np.median(np.abs(pred_flat - gt_flat))
    mape = np.mean(np.abs((pred_flat - gt_flat) / (np.abs(gt_flat) + epsilon))) * 100
    pearson_corr = np.corrcoef(pred_flat, gt_flat)[0, 1]
    r2 = r2_score(gt_flat, pred_flat)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Median_AE": median_ae,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": directional_accuracy,
        "Pearson Correlation": pearson_corr,
        "R2 Score": r2
    }
    
    # 绘制部分样本的预测与实际值（单变量情况）
    if plot_results:
        n_samples_to_plot = min(sample_plots, predictions_real.shape[0])
        for i in range(n_samples_to_plot):
            plt.figure(figsize=(8, 4))
            plt.plot(ground_truth_real[i, :], label="Actual", marker="o")
            plt.plot(predictions_real[i, :], label="Prediction", marker="x")
            plt.title(f"Sample {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    return metrics

def evaluate_financial_performance_with_coverage(predictions, ground_truth, risk_sigma, 
                                                 data_mean, data_std, plot_results=True, 
                                                 sample_plots=5, target_channel=0):
    """
    评估模型的金融预测表现并计算置信区间覆盖率，仅针对特定通道（默认 log_return，对应第0个通道）。
    
    参数:
      predictions: 模型预测输出 (预测值)，Tensor 或 ndarray，
                   单变量：形状 [样本数, 序列长度]；
                   多变量：形状 [样本数, 序列长度, 变量数]
      ground_truth: 实际目标数据，形状与 predictions 相同。
      risk_sigma: 模型预测的不确定性指标 (标准差)，形状与 predictions 相同，
                  用于构造置信区间。
      data_mean, data_std: 用于反标准化的均值和标准差（可为标量或数组，每个变量对应一个值）。
      plot_results: 是否绘制样本情况 (默认为 True)。
      sample_plots: 最多绘制的样本数 (默认为 5)。
      target_channel: 指定需要评估的通道索引，默认 0（即 log_return）。
    
    返回:
      metrics: 一个字典，其中包括 "MSE", "RMSE", "MAE", "Median_AE", "MAPE (%)", 
               "Directional Accuracy (%)", "Pearson Correlation", "R2 Score",
               以及置信区间覆盖率 "Coverage Ratio (%)"。
    """
    # 转换为 numpy 数组（如果输入为 Tensor）
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().detach().numpy()
    if torch.is_tensor(risk_sigma):
        risk_sigma = risk_sigma.cpu().detach().numpy()
    
    # 反标准化数据
    predictions_real = predictions * data_std + data_mean
    ground_truth_real = ground_truth * data_std + data_mean
    risk_sigma_real = risk_sigma * data_std  # 风险不确定性也反归一化

    # 针对多变量输入，选择目标通道，并保持维度
    if predictions_real.ndim == 3:
        predictions_real = predictions_real[:, :, target_channel:target_channel+1]
        ground_truth_real = ground_truth_real[:, :, target_channel:target_channel+1]
        risk_sigma_real = risk_sigma_real[:, :, target_channel:target_channel+1]
    elif predictions_real.ndim == 2 and predictions_real.shape[1] > 1:
        predictions_real = predictions_real[:, target_channel:target_channel+1]
        ground_truth_real = ground_truth_real[:, target_channel:target_channel+1]
        risk_sigma_real = risk_sigma_real[:, target_channel:target_channel+1]
    
    # 如果最后一个维度为1，则 squeeze 成二维数组 [B, seq_len]
    if predictions_real.ndim == 3 and predictions_real.shape[-1] == 1:
        predictions_real = np.squeeze(predictions_real, axis=-1)
        ground_truth_real = np.squeeze(ground_truth_real, axis=-1)
        risk_sigma_real = np.squeeze(risk_sigma_real, axis=-1)
    
    # 将数据展平以便计算全局指标
    pred_flat = predictions_real.reshape(-1)
    gt_flat = ground_truth_real.reshape(-1)
    
    epsilon = 1e-6  # 防止除0错误

    from scipy.stats import ttest_1samp

    differences = predictions_real - ground_truth_real
    t_stat, p_val = ttest_1samp(differences, 0)
    print("点差数组：", differences)
    print("t检验统计量为：", t_stat, ", p值为：", p_val)

    # 计算方向一致性
    pred_direction = np.sign(np.diff(predictions_real, axis=1))
    gt_direction = np.sign(np.diff(ground_truth_real, axis=1))
    directional_accuracy = np.mean(pred_direction == gt_direction) * 100

    # 计算各项指标
    mse = np.mean((predictions_real - ground_truth_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_real - ground_truth_real))
    median_ae = np.median(np.abs(pred_flat - gt_flat))
    mape = np.mean(np.abs((pred_flat - gt_flat) / (np.abs(gt_flat) + epsilon))) * 100
    pearson_corr = np.corrcoef(pred_flat, gt_flat)[0, 1]
    r2 = r2_score(gt_flat, pred_flat)
    
    # 计算 95% 置信区间
    upper_bound = predictions_real + 1.96 * risk_sigma_real
    lower_bound = predictions_real - 1.96 * risk_sigma_real

    # 计算覆盖率：真实值落在置信区间内的比例
    in_interval = np.logical_and(ground_truth_real >= lower_bound, ground_truth_real <= upper_bound)
    coverage_ratio = np.mean(in_interval) * 100  # 百分比

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Median_AE": median_ae,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": directional_accuracy,
        "Pearson Correlation": pearson_corr,
        "R2 Score": r2,
        "Coverage Ratio (%)": coverage_ratio
    }
    
    # 绘制部分样本的预测与置信区间（单变量情况）
    if plot_results:
        n_samples_to_plot = min(sample_plots, predictions_real.shape[0])
        for i in range(n_samples_to_plot):
            plt.figure(figsize=(8, 4))
            plt.plot(ground_truth_real[i, :], label="Actual", marker="o")
            plt.plot(predictions_real[i, :], label="Prediction", marker="x")
            plt.fill_between(np.arange(ground_truth_real.shape[1]), 
                             lower_bound[i, :],
                             upper_bound[i, :],
                             color='gray', alpha=0.3, label="95% CI")
            plt.title(f"Sample {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    return metrics