import pandas as pd
import numpy as np
import talib as ta
from functools import partial
import pybind11_eigen
op = pybind11_eigen.operator()

close_minute_df = None
turnover_minute_df = None
volume_minute_df = None
low_minute_df = None
high_minute_df = None
open_minute_df = None
standard_universe_minute = None
standard = None

def init_workers(close,turnover,volume,low,high,open,universe,st):
    global close_minute_df
    global turnover_minute_df
    global volume_minute_df
    global low_minute_df
    global high_minute_df
    global open_minute_df
    global standard_universe_minute
    global standard
    close_minute_df = close
    turnover_minute_df = turnover
    volume_minute_df = volume
    low_minute_df = low
    high_minute_df = high
    open_minute_df = open
    standard_universe_minute = universe
    standard = st

def standarization(df):
    df2 = df.values
    median = np.nanmedian(df2,axis=1,keepdims=True)
    mad = np.nanmedian(np.abs(df2-median),axis=1,keepdims=True)
    df2 = np.where(df2 > median+3*1.4826*mad,median+3*1.4826*mad,df2)
    df2 = np.where(df2 < median-3*1.4826*mad,median-3*1.4826*mad,df2)
    df2 = pd.DataFrame(df2,index=df.index,columns=df.columns)
    df2 = df2.sub(df2.mean(1),axis=0).div(df2.std(1),axis=0)
    return df2

def bf0_5(index,window=5):
    tmp_alpha = (close_minute_df-close_minute_df.rolling(window).mean()) / close_minute_df.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf1_5(index,window=5):
    tmp_alpha = (turnover_minute_df-turnover_minute_df.rolling(window).mean()) / turnover_minute_df.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf2_5(index,window=5):
    tmp_alpha = close_minute_df.rolling(window).corr(turnover_minute_df)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf3_5(index,window=5):
    tmp_alpha = close_minute_df / close_minute_df.shift(window) - 1
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf4_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    tmp_alpha = tmp_ret.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf5_5(index,window=5):
    tmp_alpha = (low_minute_df-low_minute_df.rolling(window).mean()) / low_minute_df.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf6_5(index,window=5):
    tmp_alpha = (high_minute_df-high_minute_df.rolling(window).mean()) / high_minute_df.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf7_5(index,window=5):
    tmp_alpha = (open_minute_df-open_minute_df.rolling(window).mean()) / open_minute_df.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf8_5(index,window=5):
    tmp_alpha = turnover_minute_df.rolling(window).sum() / turnover_minute_df.rolling(30*1440).sum()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf9_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    mkt_mom1 = tmp_ret.mean(1)
    tmp_alpha = tmp_ret.rolling(window).corr(mkt_mom1)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf10_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    resi_ret = tmp_ret.sub(tmp_ret.mean(1),axis=0)
    tmp_alpha = resi_ret.rolling(window).std()
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf11_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    mkt_mom1 = pd.DataFrame(np.repeat(tmp_ret.mean(1).values[:,np.newaxis],tmp_ret.shape[1],axis=1),index=tmp_ret.index,columns=tmp_ret.columns)
    tmp_alpha = pd.DataFrame(op.ts_resi_r2(tmp_ret,mkt_mom1,window),index=tmp_ret.index,columns=tmp_ret.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf12_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    tmp_ret = tmp_ret.abs()
    iliq = tmp_ret / turnover_minute_df
    iliq[np.isinf(iliq)] = np.nan
    tmp_alpha = np.log(iliq.rolling(window,min_periods=1).mean())
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    size = np.log(turnover_minute_df.rolling(30*1440).sum())
    size[np.isinf(size)] = np.nan
    tmp_alpha = pd.DataFrame(op.cross_resi(tmp_alpha,size),index=size.index,columns=size.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf13_5(index,window=5):
    vwap = turnover_minute_df / volume_minute_df
    vwap[np.isinf(vwap)] = np.nan
    top = vwap.rolling(window,min_periods=1).mean()
    bot = turnover_minute_df.rolling(window).sum() / volume_minute_df.rolling(window).sum()
    bot[np.isinf(bot)] = np.nan
    tmp_alpha = np.log(top/bot)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf14_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    tmp_alpha = tmp_ret.rolling(window).skew()    
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf15_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    tmp_alpha = tmp_ret.rolling(window).kurt()    
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf16_5(index,window=5):
    tmp_rate = volume_minute_df / volume_minute_df.rolling(window).sum()
    tmp_rate[np.isinf(tmp_rate)] = np.nan
    tmp_alpha = (np.power(tmp_rate,2)).rolling(window).sum()

    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf17_5(index,window=5):
    tmp_rate = volume_minute_df / volume_minute_df.rolling(window).sum()
    tmp_rate[np.isinf(tmp_rate)] = np.nan
    tmp_alpha = (np.power(tmp_rate,2)).rolling(window).sum()

    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf18_5(index,window=5):
    tmp_low = low_minute_df.rolling(window).min()
    tmp_high = high_minute_df.rolling(window).max()
    tmp_price = ((high_minute_df+low_minute_df+open_minute_df+close_minute_df)/4).rolling(window).mean()
    tmp_alpha = (tmp_price - tmp_low) / (tmp_high - tmp_low)

    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf19_5(index,window=5):
    tmp_alpha = volume_minute_df.rolling(window).std() / volume_minute_df.rolling(window).sum()

    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf20_5(index,window=5):
    tmp_1 = high_minute_df / low_minute_df - 1
    tmp_1[np.isinf(tmp_1)] = np.nan
    tmp_alpha = tmp_1.rolling(window).mean()

    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def bf21_5(index,window=5):
    tmp_ret = close_minute_df / close_minute_df.shift(1) - 1
    tmp_ret[np.isinf(tmp_ret)] = np.nan
    mkt_mom1 = tmp_ret.mean(1)
    tmp_alpha = tmp_ret.rolling(window).corr(mkt_mom1)
    tmp_alpha = np.power(tmp_alpha,2)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def aro_5(index,window=5):
    real = np.concatenate([ta.AROONOSC(high_minute_df.values[:,i],low_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def rsi_5(index,window=5):
    real = np.concatenate([ta.RSI(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def mfi_5(index,window=5):
    real = np.concatenate([ta.MFI(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],volume_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def lin_5(index,window=5):
    real = np.concatenate([ta.LINEARREG(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def lin_a_5(index,window=5):
    real = np.concatenate([ta.LINEARREG_ANGLE(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def lin_s_5(index,window=5):
    real = np.concatenate([ta.LINEARREG_SLOPE(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def lin_i_5(index,window=5):
    real = np.concatenate([ta.LINEARREG_INTERCEPT(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def trix_5(index,window=5):
    real = np.concatenate([ta.TRIX(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def tsf_5(index,window=5):
    real = np.concatenate([ta.TSF(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def roc_5(index,window=5):
    real = np.concatenate([ta.ROC(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def adxr_5(index,window=5):
    real = np.concatenate([ta.ADXR(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def adx_5(index,window=5):
    real = np.concatenate([ta.ADX(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def cmo_5(index,window=5):
    real = np.concatenate([ta.CMO(close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def plus_5(index,window=5):
    real = np.concatenate([ta.PLUS_DI(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def minus_5(index,window=5):
    real = np.concatenate([ta.MINUS_DI(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def beta_5(index,window=5):
    real = np.concatenate([ta.BETA(high_minute_df.values[:,i],low_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def natr_5(index,window=5):
    real = np.concatenate([ta.NATR(high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def corr_5(index,window=5):
    real = np.concatenate([ta.CORREL(high_minute_df.values[:,i],low_minute_df.values[:,i],timeperiod=window)[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index

def BOP(index):
    real = np.concatenate([ta.BOP(open_minute_df.values[:,i],high_minute_df.values[:,i],low_minute_df.values[:,i],close_minute_df.values[:,i])[:,np.newaxis] for i in range(high_minute_df.shape[1])],axis=1)
    tmp_alpha = pd.DataFrame(real,index=high_minute_df.index,columns=high_minute_df.columns)
    tmp_alpha[np.isinf(tmp_alpha)] = np.nan
    tmp_alpha = tmp_alpha[standard_universe_minute==1]
    tmp_alpha = standarization(tmp_alpha)
    return tmp_alpha,index