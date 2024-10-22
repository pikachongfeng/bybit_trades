import easy_kline
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--symbol',type=str)

args = parser.parse_args()
symbol = args.symbol


from utils.get_bybit_data import *

if __name__ == "__main__":
    """ symbols = get_symbols()
    for symbol in symbols:
        easy_kline.bybit(symbol, '1m', '2023-01-01 00:00', futures=True) """
    easy_kline.bybit(symbol, '1m', '2023-01-01 00:00', futures=True)