import pandas as pd
import numpy as np
import akshare as ak
import io as io
import base64 as base64


file_path = './Database/sh_stock_data.h5'

"""start_date = '2023-01-01'
end_date = '2023-12-31'"""



def load_stock_data(stock_code, start_date, end_date, index='log_return'):
    """
    读取 HDF5 文件中的股票数据，并返回包含日期和 log_return 的 DataFrame
    """
    data = pd.read_hdf(
        file_path,
        key='stock_date',
        where=f"stock_code='{stock_code}' & stock_date>='{start_date}' & stock_date<='{end_date}'"
    )
    data = data[['stock_date', index]].set_index('stock_date')
    data.rename(columns={f'{index}': f'{index}_{stock_code}'}, inplace=True)
    return data

def merge_stocks(data1, data2):
    """
    合并两个股票的数据，按日期对齐
    """
    merged_data = data1.join(data2, how='inner')  # 仅保留共同日期的数据
    return merged_data

def get_shanghai_index(start_date, end_date,info):# 获取上证数据
    sh_index = ak.stock_zh_index_daily(symbol="sh000001")  # 上证指数代码
    sh_index['date'] = pd.to_datetime(sh_index['date'])  # 转换日期
    sh_index = sh_index[(sh_index['date'] >= start_date) & (sh_index['date'] <= end_date)]
    sh_index['log_return'] = np.log(sh_index['close'] / sh_index['close'].shift(1))  # 计算对数收益率
    
    return sh_index[['date', f"{info}"]].dropna().set_index('date')

def load_entire_data(stock_code, start_date, end_date):
    """
    读取 HDF5 文件中的股票数据，并返回包含日期和 log_return 的 DataFrame
    """
    data = pd.read_hdf(
        file_path,
        key='stock_date',
        where=f"stock_code='{stock_code}' & stock_date>='{start_date}' & stock_date<='{end_date}'"
    )
    data = data.set_index('stock_date')
    
    return data

print(load_entire_data('600004','2023-01-01','2023-12-31'))
