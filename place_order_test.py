import time
import hashlib
import hmac
import requests

# 填入你的 API 密钥和私钥
API_KEY = 'GC4VJaD02PsTIuadLI'
API_SECRET = 'kR5T7qYmtcwMjEJvObtBAgBhiKx0CF07mZcA'
BASE_URL = 'https://api-testnet.bybit.com'  # 主网地址，测试网地址为 https://api-testnet.bybit.com
def generate_signature(secret, params):
    """生成HMAC-SHA256签名"""
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()

def place_order(symbol, side, qty, price, order_type='Limit'):
    """限价下单"""
    params = {
        'category': 'linear',
        'api_key': API_KEY,
        'symbol': symbol,
        'side': side,  # 'Buy' or 'Sell'
        'orderType': order_type,  # 'Limit' or 'Market'
        'qty': qty,
        'price': price,
        'options': {
            'adjustForTimeDifference': True,  # exchange-specific option
        }
    }
    url = "https://api-testnet.bybit.com/v5/order/create"

    headers = {
        'X-BAPI-API-KEY': 'GC4VJaD02PsTIuadLI',
        'X-BAPI-TIMESTAMP': str(time.time()),
        'X-BAPI-RECV-WINDOW': '20000',
        'X-BAPI-SIGN': 'kR5T7qYmtcwMjEJvObtBAgBhiKx0CF07mZcA'
    }

    response = requests.request("POST", url, headers=headers, data=params)

    return response.json()

# 示例：下单一个BTCUSDT合约的限价买单，价格为30,000美元
response = place_order('BTCUSDT', 'Buy', qty=0.01, price=68399.00)
print(response)