import requests
import time
import pandas as pd
from datetime import datetime
import threading

MAX_REQUESTS_PER_SECOND = 100
semaphore = threading.Semaphore(MAX_REQUESTS_PER_SECOND)

def get_symbols():
    url = "https://api.bybit.com/v2/public/symbols"
    response = requests.get(url).json()
    symbols = [item['name'] for item in response['result']]
    return symbols

def timestamp_to_datetime(t):
    return datetime.utcfromtimestamp(int(int(t)/1000))

# 获取指定合约的1分钟K线数据
def get_minute_ohlcv(symbol,now_time):
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "symbol": symbol,
        "interval": "1",  # 1分钟K线
        "start": (now_time - 60)*1000,  # 获取最近1分钟的K线
        "end": (now_time)*1000  # 获取最近1分钟的K线
    }
    with semaphore:
        response = requests.get(url, params=params).json()
    if response["retCode"] == 0 and response["result"]:
        result = response["result"]
        return response["result"]['list'][-1]  # 返回最新的K线数据
    return None

def get_1000_ohlcv(symbol,now_time):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": "1",  # 1分钟K线
        "start": (now_time)*1000,  # 获取最近1分钟的K线
        "end": (now_time+60*999)*1000,  # 获取最近1分钟的K线
        "limit": 1000,
    }
    with semaphore:  
        response = requests.get(url, params=params).json()
    if response["retCode"] == 0 and response["result"]:
        return response["result"]['list']  # 返回最新的K线数据
    return None

def fetch_symbol_data_parallel(symbol,now_time):
    data = get_minute_ohlcv(symbol,now_time)
    if data:
        print(f"Symbol: {symbol}, Kline: {data}")
    else:
        print(f"No data for {symbol}")

def fetch_symbol_data_parallel_his(symbol,now_time):
    data = get_1000_ohlcv(symbol,now_time)
    if data and len(data)>0:
        res_open = pd.Series(name=symbol)
        res_high = pd.Series(name=symbol)
        res_low = pd.Series(name=symbol)
        res_close = pd.Series(name=symbol)
        res_volume = pd.Series(name=symbol)
        res_turnover = pd.Series(name=symbol)
        for data_point in reversed(data):
            timestamp = data_point[0]
            index_t = timestamp_to_datetime(timestamp)
            open_data = data_point[1]
            high_data = data_point[2]
            low_data = data_point[3]
            close_data = data_point[4]
            volume_data = data_point[5]
            turnover_data = data_point[6]
            res_open.loc[index_t] = open_data
            res_high.loc[index_t] = high_data
            res_low.loc[index_t] = low_data
            res_close.loc[index_t] = close_data
            res_volume.loc[index_t] = volume_data
            res_turnover.loc[index_t] = turnover_data
        return symbol,[res_open,res_high,res_low,res_close,res_volume,res_turnover]
    else:
        return symbol,None