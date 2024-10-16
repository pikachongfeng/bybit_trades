import requests
import time
import datetime
from datetime import datetime
from concurrent import futures
import argparse
from utils.get_bybit_data import *
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--num_workers',type=int,default=30)
args = parser.parse_args()

if __name__ == "__main__":
    while True:
        current_seconds = datetime.now().second
        if current_seconds <= 4:
            time.sleep(1)
        symbols = get_symbols()
        the_futures = []
        now_time = int(time.time())
        with futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for symbol in symbols:
                future = executor.submit(fetch_symbol_data_parallel,symbol,now_time)
                the_futures.append(future)
        for future in futures.as_completed(the_futures):
            pass

