import requests
import time
from datetime import datetime,timedelta
from concurrent import futures
import argparse
import pandas as pd
from utils.get_bybit_data import *
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--num_workers',type=int,default=30)
args = parser.parse_args()

start_date = 20240101

def date_to_unix(date_str):
    dt = datetime.strptime(str(date_str), "%Y%m%d")
    return int(dt.timestamp())

def timestamps_to_datetime_list(timestamp1, timestamp2):
    dt_list = []
    start_dt = datetime.utcfromtimestamp(min(timestamp1, timestamp2))
    end_dt = datetime.utcfromtimestamp(max(timestamp1, timestamp2))

    current_dt = start_dt
    while current_dt <= end_dt:
        dt_list.append(current_dt)
        current_dt += timedelta(minutes=1)

    return dt_list

if __name__ == "__main__":
    start_timestamp = date_to_unix(start_date)
    end_timestamp = int(time.time())
    symbols = get_symbols()
    date_list = timestamps_to_datetime_list(start_timestamp,end_timestamp)
    res_close = pd.DataFrame(columns = symbols,index=date_list)
    res_open = pd.DataFrame(columns = symbols,index=date_list)
    res_low = pd.DataFrame(columns = symbols,index=date_list)
    res_high = pd.DataFrame(columns = symbols,index=date_list)
    res_turnover = pd.DataFrame(columns = symbols,index=date_list)
    res_volume = pd.DataFrame(columns = symbols,index=date_list)

    for t in range(start_timestamp,end_timestamp,1000*60):
        the_futures = []
        """ for symbol in symbols:
            print(symbol)
            res = fetch_symbol_data_parallel_his(symbol,t)
            symbol = res[0]
            data = res[1]
            if data:
                open_data = data[0]
                high_data = data[1]
                low_data = data[2]
                close_data = data[3]
                volume_data = data[4]
                turnover_data = data[5]
                res_open.loc[open_data.index,symbol] = open_data.values
                res_high.loc[high_data.index] = high_data.values
                res_low.loc[low_data.index] = low_data.values
                res_close.loc[close_data.index] = close_data.values
                res_volume.loc[volume_data.index] = volume_data.values
                res_turnover.loc[turnover_data.index] = turnover_data.values """
        with futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for symbol in symbols:
                future = executor.submit(fetch_symbol_data_parallel_his,symbol,t)
                the_futures.append(future)
        for future in futures.as_completed(the_futures):
            res = future.result()
            symbol = res[0]
            data = res[1]
            if data:
                open_data = data[0]
                high_data = data[1]
                low_data = data[2]
                close_data = data[3]
                volume_data = data[4]
                turnover_data = data[5]
                res_open.loc[open_data.index,symbol] = open_data.values
                res_high.loc[high_data.index,symbol] = high_data.values
                res_low.loc[low_data.index,symbol] = low_data.values
                res_close.loc[close_data.index,symbol] = close_data.values
                res_volume.loc[volume_data.index,symbol] = volume_data.values
                res_turnover.loc[turnover_data.index,symbol] = turnover_data.values
    res_open.to_pickle('./open_data.pkl')
    res_high.to_pickle('./high_data.pkl')
    res_low.to_pickle('./low_data.pkl')
    res_close.to_pickle('./close_data.pkl')
    res_volume.to_pickle('./volume_data.pkl')
    res_turnover.to_pickle('./turnover_data.pkl')