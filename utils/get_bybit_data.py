import requests
import time
def get_symbols():
    url = "https://api.bybit.com/v2/public/symbols"
    response = requests.get(url).json()
    symbols = [item['name'] for item in response['result']]
    return symbols

# 获取指定合约的1分钟K线数据
def get_minute_ohlcv(symbol,now_time):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": "1",  # 1分钟K线
        "start": (now_time - 60)*1000,  # 获取最近1分钟的K线
        "end": (now_time)*1000  # 获取最近1分钟的K线
    }
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
    response = requests.get(url, params=params).json()
    if response["retCode"] == 0 and response["result"]:
        result = response["result"]
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
    if data:
        print(f"Symbol: {symbol}, Kline: {data}")
    else:
        print(f"No data for {symbol}")