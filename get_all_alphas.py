import multiprocessing
import alphas.get_alphas_all_488

close_minute_df = None
turnover_minute_df = None
volume_minute_df = None
low_minute_df = None
high_minute_df = None
open_minute_df = None
standard_universe_minute = None
standard = None

def run_function_all(func,index):
    return func(index)

with multiprocessing.Pool(processes=multiprocessing.cpu_count(),initializer=alphas.get_alphas_all_488.init_workers,initargs=(close_minute_df,turnover_minute_df,volume_minute_df,low_minute_df,high_minute_df,open_minute_df,standard_universe_minute,standard)) as pool:
    results = pool.starmap(run_function_all,[(func,idx)] for idx,func in enumerate())
    for res in results:
        pass