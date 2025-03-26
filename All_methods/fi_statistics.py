import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 关闭GUI
plt.switch_backend('Agg')
import seaborn as sns
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from flask import Flask, render_template, request, jsonify

import sys
import io as io
import base64 as base64
import time
import logging

sns.set_style("whitegrid")



def statistic_overview(data):
    #data是一个pandas数据库DataFrame，要求有以下结构：以日期为索引，任意列数，不同列数据属性一致。
    # 计算统计特征，包括平均收益率、标准差、偏度、峰度
    if data.empty:
        raise ValueError(f"股票统计 {stock_code} 在 {start_date} ~ {end_date} 无数据！")
    return  data.mean(),data.std(),data.apply(skew),data.apply(kurtosis)

def cumulative_return_visible(data):
    #data是一个包含列名为'stock_log_return'和'sh_index_log_return'的两列数据表。
    # 计算累积收益率
    data['stock_cum_return'] = np.exp(data['stock_log_return'].cumsum()) - 1
    data['sh_index_cum_return'] = np.exp(data['sh_index_log_return'].cumsum()) - 1

    # 绘制累积收益率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['stock_cum_return'], label=f'Stock  Cumulative Return', color='blue')
    plt.plot(data.index, data['sh_index_cum_return'], label='Shanghai Index Cumulative Return', color='red')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def cor(data, stock1, stock2, info='对数收益率'):
    """
    计算相关性，并绘制散点图。info 为数据描述字符串
    """
    correlation = data.corr().iloc[0, 1]
    
    # 创建 Matplotlib 图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=data.iloc[:, 0], y=data.iloc[:, 1], scatter_kws={'alpha': 0.5}, ax=ax)
    ax.set_xlabel(f"{stock1} {info}")
    ax.set_ylabel(f"{stock2} {info}")
    ax.set_title(f"{stock1} vs {stock2} {info}相关性")

    # 将图像保存到内存
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    plt.close(fig)  # 关闭图形，释放内存

    return correlation, img_base64


def volatility_clustering(data, stock1, stock2, lags=20):
    """
    进行波动聚集性分析 (收益率绝对值的自相关性)
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_acf(np.abs(data.iloc[:, 0]).dropna(), lags=lags, title=f"{stock1} 波动率自相关")

    plt.subplot(1, 2, 2)
    plot_acf(np.abs(data.iloc[:, 1]).dropna(), lags=lags, title=f"{stock2} 波动率自相关")

    plt.show()

def lag_effect(data, stock1, stock2, max_lag=10):
    """
    计算两个股票的收益率滞后效应，并返回 ACF/PACF 图的 Base64 编码
    """

    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # 绘制 ACF 和 PACF 图
    plot_acf(data.iloc[:, 0].dropna(), lags=max_lag, ax=axes[0, 0])
    axes[0, 0].set_title(f"{stock1} 收益率自相关 (ACF)")

    plot_pacf(data.iloc[:, 0].dropna(), lags=max_lag, ax=axes[0, 1])
    axes[0, 1].set_title(f"{stock1} 收益率偏自相关 (PACF)")

    plot_acf(data.iloc[:, 1].dropna(), lags=max_lag, ax=axes[1, 0])
    axes[1, 0].set_title(f"{stock2} 收益率自相关 (ACF)")

    plot_pacf(data.iloc[:, 1].dropna(), lags=max_lag, ax=axes[1, 1])
    axes[1, 1].set_title(f"{stock2} 收益率偏自相关 (PACF)")

    plt.tight_layout()

    # 将图像保存到内存中并转换为 Base64 编码
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    plt.close(fig)  # 关闭 Matplotlib 图形，释放内存

    # 返回 Base64 编码的图片
    return img_base64


def rolling_cor(data, stock1, stock2, window_size=30):
    """
    计算两个股票的滚动相关性，并返回相关性图的 Base64 编码。默认滚动窗口30 days
    """
    data['rolling_corr'] = data.iloc[:, 0].rolling(window=window_size).corr(data.iloc[:, 1])

    # 绘制滚动相关性图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['rolling_corr'], label=f"Rolling Correlation ({window_size}-day)", color="purple")
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{stock1} vs {stock2} 滚动相关性")
    ax.legend()

    # 将图像保存到内存
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    plt.close(fig)  # 关闭图形，释放内存

    return img_base64


def beta(data):
    # 计算β值:{stock1} 对 {stock2} 的 Beta 值
    X = sm.add_constant(data.iloc[:, 1].dropna())  # 添加截距项
    y = data.iloc[:, 0].dropna()
    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[1]

    return f"stock1 对 stock2 的 Beta 值: {beta:.4f}"