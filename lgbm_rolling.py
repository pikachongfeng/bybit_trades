import pandas as pd
from lightgbm import LGBMRegressor
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--max_depth',type=int,required=True)
parser.add_argument('--colsample_bytree',type=int,required=True)
parser.add_argument('--tree_num',type=int,required=True)
parser.add_argument('--lr',type=int,required=True)
parser.add_argument('--pred_freq',type=int,required=True)

args = parser.parse_args()

max_depth = args.max_depth
colsample_bytree = args.colsample_bytree
tree_num = args.tree_num
lr = args.lr
pred_freq = args.pred_freq

time_point = 1440
training_len = 90
test_len = 30

start_date = 20230301
end_date = 20241014
fee = 0.0005
interval = 15

save_folder = './lgbm_results'
alpha_name = 'lgbm_rolling_' + str(pred_freq) + 'min_' + str(training_len) + '_' + str(test_len) + '_' + str(max_depth) + '_' + str(colsample_bytree) + '_' + str(lr) + '_' + str(tree_num)

Turnover = None
turnover_min = None
close_minute_df = None
standard_univer_minute = None
rtn_1440 = ((close_minute_df/close_minute_df.shift(pred_freq) - 1).shift(-pred_freq-1))[standard_univer_minute==1]
rtn_1440[np.isinf(rtn_1440)] = np.nan
dts = close_minute_df.index.get_level_values(0).unique().sort_values()
dts_test = dts[training_len:]
mi = pd.MultiIndex.from_product([dts,range(0,time_point,interval),rtn_1440.columns])
mi_test = pd.MultiIndex.from_product([dts_test,range(0,time_point,interval),rtn_1440.columns])
rtn_series = rtn_1440.stack(dropna=False).reindex(mi).rename('label')

def eli_ex(row):
    up = row.quantile(0.99)
    down = row.quantile(0.01)
    row = row.apply(lambda x : up if x > up else x)
    row = row.apply(lambda x : down if x < down else x)
    return row

code_num = rtn_1440.shape[1]
selected_codes = rtn_1440.columns
alpha_df = None
total_step = mail.ceil((len(mi_test))/test_len)
print('total_step=',total_step)

factor_df = pd.DataFrame(index=mi_test,columns=selected_codes).astype(np.float64)

def backtest(step):
    step_test_start_date = dts_test[step*test_len]
    step_test_end_date = dts[(step+1)*test_len-1]
    test_data = alpha_df.loc[step_test_start_date:step_test_end_date]
    if step == 0:
        training_data = alpha_df.loc[:dts[training_len-1]]
        training_label = rtn_series.loc[:dts[training_len-1]]
    else:
        training_data = alpha_df.loc[:dts_test[step*test_len-1]]
    combined_df = pd.concat([training_data,training_label],aix=s1)
    combined_df = combined_df.loc[combined_df.index.get_level_values(1).isin(range(0,1440,15))]
    combined_df = combined_df.dropna()
    if combined_df.shape[0] == 0:
        test_signal = pd.Series(np.nan,index=test_data.index)
        return test_signal
    train_input = combined_df[training_data.columns].fillna(0)
    training_label = combined_df.iloc[:,-1]

    best_model = LGBMRegressor(n_estimator=tree_num,learning_rate=lr,min_data_in_leaf=60,max_depth=max_depth,reg_lambda=100,colsample_by_tree=colsample_bytree,objective='regression',random_state=2024)
    best_model.fit(train_input,training_label,verbose=100)
    predict = best_model.predict(test_data.fillna(0),num_iteration=best_model.best_iteration_)
    test_signal = pd.Series(predict,index=test_data.index)
    return test_signal

for step in range(total_step):
    test_signal = backtest(step)
    test_signal = test_signal.unstack(2).reindex(rtn_1440.columns,axis=1)
    factor_df.loc[test_signal.index] = test_signal

factor_df = factor_df[standard_univer_minute==1]


save_path = os.path.join(save_folder,alpha_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

factor_df = factor_df.dropna(how='all')
factor_df = factor_df.sub(factor_df.mean(1),axis=0).div(factor_df.std(1),axis=0)
factor_df.to_pickle(os.path.join(save_path,'factor_df.pkl'))