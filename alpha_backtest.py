import pandas as pd
from lightgbm import LGBMRegressor

time_point = 1440

start_date = 20230301
end_date = 20241014
fee = 0.0005
interval = 15
pred_freq_list = [480]

alpha_name = 'lgbm_rolling_' + str(pred_freq) + 'min_' + str(training_len) + '_' + str(test_len) + '_' + str(max_depth) + '_' + str(colsample_bytree) + '_' + str(lr) + '_' + str(tree_num)
save_folder = './backtest_results'
save_path = os.path.join(save_folder,alpha_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

Turnover = None
turnover_min = None
close_minute_df = None
standard_univer_minute = None
rtn_1440 = ((close_minute_df/close_minute_df.shift(pred_freq) - 1).shift(-pred_freq-1))[standard_univer_minute==1]

def standarization(df):
    df2 = df.values
    median = np.nanmedian(df2,axis=1,keepdims=True)
    mad = np.nanmedian(df2,axis=1,keepdims=True)
    df2 = np.where(df2>median+3*1.4826*mad,median+3*1.4826*mad,df2)
    df2 = np.where(df2<median-3*1.4826*mad,median-3*1.4826*mad,df2)
    df2 = pd.DataFrame(df2,index=df.index,columns=df.columns)
    df2 = df2.sub(df2.mean(1),axis=0).div(df2.std(1),axis=0)
    return df2

alpha = pd.read_pickle('')
alpha = alpha[~alpha.index.duplicated(keep='last')]
alpha = alpha.reindex(close_minute_df.index)

res = pd.DataFrame(index=pred_freq_list,columns=['IC','ICIR','LongShort Sharp','LongShort Rtn','ValueWeighted Sharp','ValueWeighted Rtn'])

for pred_freq in pred_freq_list:
    rtn_1440 = ((close_minute_df/close_minute_df.shift(pred_freq)-1).shift(-pred_freq-1))[standard_univer_minute==1]
    alpha = alpha[standard_univer_minute==1]
    rtn_1440 = rtn_1440.reindex(alpha.columns,axis=1)
    time_list = range(0,time_point,15)
    ii = rtn_1440.index.get_level_values(1).isin(time_list)
    ic_min = alpha[ii].corrwith(rtn_1440[ii],axis=1)
    ric_min = alpha[ii].corrwith(rtn_1440[ii],axis=1,method='spearman')
    cs_excessrtn = rtn_1440[alpha.rank(pct=True,axis=1)>0.9]
    cs_excessrtn2 = rtn_1440[alpha.rank(pct=True,axis=1)<0.1]
    cs_excessrtnls = cs_excessrtn.combine_first(cs_excessrtn2)
    holds = pd.DataFrame(0,index=cs_excessrtn.index,columns=cs_excessrtn.columns)
    holds[~cs_excessrtn.isna()] = 1
    holds[~cs_excessrtn2.isna()] = -1
    holds = holds.div(holds.abs().sum(1),axis=0)
    turn_rate = (holds-holds.shift(-pred_freq)).abs().sum(1)
    transaction_cost = turn_rate * fee
    CS_daily = (cs_excessrtnls[ii].mean(1)).fillna(0)
    alpha_tmp  =alpha[~cs_excessrtnls.isna()]
    weight = alpha_tmp.div(alpha_tmp.abs().sum(1),axis=0)
    CS_daily2 = ((cs_excessrtnls[ii]*weight[ii]).sum(1) - (weight - weight.shift(-pred_freq)).abs().sum(1)[ii]*fee).fillna(0)

    CS_daily.cumsum().plot(title = 'longshort series')
    plt.save_fig(os.path.join(save_path,'longshort_series_' + str(pred_feeq)+'.png'))
    plt.clf()

    CS_daily2.cumsum().plot(title = 'weighted longshort series')
    plt.save_fig(os.path.join(save_path,'weighted_longshort_series_' + str(pred_feeq)+'.png'))
    plt.clf()

    ic_min.cumsum().plot(title = 'ic series')
    plt.save_fig(os.path.join(save_path,'ic_series_' + str(pred_feeq)+'.png'))
    plt.clf()

    winrate = (CS_daily>0).sum() / len(CS_daily)
    winrate2 = (CS_daily2>0).sum() / len(CS_daily2)
    sharp = CS_daily.mean()/CS_daily.std()
    sharp2 = CS_daily2.mean()/CS_daily2.std()

    freq_res = pd.Series([ic_min.mean(),ic_min.mean()/ic_min.std(),sharp,CS_daily.mean(),sharp2,CS_daily2.mean()],index=['IC','ICIR','LongShort Sharp','LongShort Rtn','ValueWeighted Sharp','ValueWeighted Rtn'])
    res.loc[pred_freq] = freq_res

res.to_csv(os.path.join(save_path,'backtest_results.csv'))