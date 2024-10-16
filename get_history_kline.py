import requests
import time
from datetime import datetime
from concurrent import futures
import argparse
import pandas as pd
from utils.get_bybit_data import *
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--num_workers',type=int,default=30)
args = parser.parse_args()

start_date = 20230101

def date_to_unix(date_str):
    dt = datetime.strptime(str(date_str), "%Y%m%d")
    return int(dt.timestamp())

if __name__ == "__main__":
    start_timestamp = date_to_unix(start_date)
    end_timestamp = int(time.time())
    symbols = get_symbols()
    
    for t in range(start_timestamp,end_timestamp,1000):
        the_futures = []
        with futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for symbol in symbols:
                future = executor.submit(fetch_symbol_data_parallel_his,symbol,t)
                the_futures.append(future)
        for future in futures.as_completed(the_futures):
            pass

