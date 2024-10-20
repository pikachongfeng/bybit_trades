import time
import hmac
import hashlib
import requests

API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
BASE_URL = 'https://api.bybit.com'  # 测试网：https://api-testnet.bybit.com

def generate_signature(secret, params):
    """生成HMAC-SHA256签名"""
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()

def get_latest_price(symbol):
    """获取最新成交价格"""
    endpoint = '/v5/market/tickers'
    params = {
        'api_key': API_KEY,
        'symbol': symbol,
        'category': 'linear',  # 永续合约为'linear'
        'timestamp': int(time.time() * 1000),
    }
    params['sign'] = generate_signature(API_SECRET, params)

    url = BASE_URL + endpoint
    response = requests.get(url, params=params)
    return response.json()

# 示例：查询BTCUSDT永续合约的最新成交价格
response = get_latest_price('BTCUSDT')
if 'result' in response:
    latest_price = response['result']['list'][0]['lastPrice']
    print(f"最新成交价格: {latest_price}")
else:
    print("获取数据失败:", response)