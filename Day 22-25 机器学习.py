import tushare as ts
import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
import alphalens as al
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import seaborn as sns
ts.set_token(" ")
pro = ts.pro_api()

#LightGBM合成因子（含因子重要性分析）

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取五个因子面板 ====================
def load_factor(filepath, index_col='stock_code'):
    df = pd.read_csv(filepath, index_col=index_col)
    if 'stock_name' in df.columns:
        df = df.drop('stock_name', axis=1)
    df.columns = pd.to_datetime(df.columns)
    return df

mom = load_factor('动量因子_2024_2026_with_names.csv')
size = load_factor('规模因子_市值倒数_标准化_2024_2026.csv')
value = load_factor('价值因子_BP_processed.csv')
roe = load_factor('盈利因子_ROE_processed.csv')
cma = load_factor('投资因子_CMA_processed.csv')

common_dates = mom.columns.intersection(size.columns).intersection(value.columns).intersection(
    roe.columns).intersection(cma.columns)
common_dates = common_dates.sort_values()
common_stocks = mom.index.intersection(size.index).intersection(value.index).intersection(roe.index).intersection(
    cma.index)

mom = mom.loc[common_stocks, common_dates]
size = size.loc[common_stocks, common_dates]
value = value.loc[common_stocks, common_dates]
roe = roe.loc[common_stocks, common_dates]
cma = cma.loc[common_stocks, common_dates]

factors = {
    '动量': mom,
    '规模': size,
    '价值': value,
    '盈利': roe,
    '投资': cma
}

# ==================== 2. 读取已保存的 IC 序列（用于IC加权） ====================
ic_mom = pd.read_csv('ic_momentum.csv', index_col=0, squeeze=True)
ic_size = pd.read_csv('ic_size.csv', index_col=0, squeeze=True)
ic_value = pd.read_csv('ic_value.csv', index_col=0, squeeze=True)
ic_roe = pd.read_csv('ic_roe.csv', index_col=0, squeeze=True)
ic_cma = pd.read_csv('ic_cma.csv', index_col=0, squeeze=True)

ic_mom.index = pd.to_datetime(ic_mom.index)
ic_size.index = pd.to_datetime(ic_size.index)
ic_value.index = pd.to_datetime(ic_value.index)
ic_roe.index = pd.to_datetime(ic_roe.index)
ic_cma.index = pd.to_datetime(ic_cma.index)

ic_df = pd.DataFrame({
    '动量': ic_mom,
    '规模': ic_size,
    '价值': ic_value,
    '盈利': ic_roe,
    '投资': ic_cma
})
ic_df = ic_df.loc[common_dates]

# ==================== 3. 准备未来收益数据 ====================
pivot = pd.read_csv('monthly_close_wide.csv', index_col='stock_code')
pivot.columns = pd.to_datetime(pivot.columns)
pivot = pivot.loc[common_stocks, common_dates]
future_ret = pivot.pct_change(axis=1).shift(-1, axis=1)
future_ret = future_ret.iloc[:, :-1]
common_dates_ret = future_ret.columns
for name in factors:
    factors[name] = factors[name][common_dates_ret]
ic_df = ic_df.loc[common_dates_ret]

# ==================== 4. 等权组合权重 ====================
weights_equal = pd.DataFrame(1 / len(factors), index=common_dates_ret, columns=list(factors.keys()))

# ==================== 5. IC加权组合权重（滚动历史均值，无未来信息） ====================
ic_shifted = ic_df.shift(1)
rolling_ic_mean = ic_shifted.rolling(window=12, min_periods=1).mean()
abs_sum = rolling_ic_mean.abs().sum(axis=1)
weights_hist = rolling_ic_mean.div(abs_sum, axis=0)
weights_hist = weights_hist.fillna(1 / len(factors))

# ==================== 6. 合成等权和IC加权组合得分 ====================
def combine_factors(factors, weights_df):
    combined = pd.DataFrame(index=factors[list(factors.keys())[0]].index,
                            columns=weights_df.index)
    for date in weights_df.index:
        w = weights_df.loc[date]
        score = 0
        for name, f_df in factors.items():
            score += w[name] * f_df[date]
        combined[date] = score
    return combined

score_equal = combine_factors(factors, weights_equal)
score_hist = combine_factors(factors, weights_hist)

# ==================== 7. 分组回测函数（分5组） ====================
def group_backtest(score_df, ret_df, n_groups=5):
    group_returns = []
    for month in score_df.columns:
        factor = score_df[month].dropna()
        ret = ret_df[month].loc[factor.index]
        if len(factor) == 0:
            continue
        try:
            groups = pd.qcut(factor.rank(method='first'), n_groups, labels=False) + 1
        except:
            groups = pd.cut(factor, n_groups, labels=False) + 1
        group_mean = ret.groupby(groups).mean()
        group_mean.name = month
        group_returns.append(group_mean)
    group_ret_df = pd.DataFrame(group_returns).T
    long_short = group_ret_df.loc[n_groups] - group_ret_df.loc[1]
    return group_ret_df, long_short

group_eq, ls_eq = group_backtest(score_equal, future_ret, n_groups=5)
group_hist, ls_hist = group_backtest(score_hist, future_ret, n_groups=5)

# ==================== 8. 绩效指标函数 ====================
def calc_performance(returns_series):
    if returns_series.empty or returns_series.isna().all():
        return (np.nan,) * 5
    annual_return = (1 + returns_series).prod() ** (12 / len(returns_series)) - 1
    annual_vol = returns_series.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    cum = (1 + returns_series).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    win_rate = (returns_series > 0).mean()
    return annual_return, annual_vol, sharpe, max_dd, win_rate

# ==================== 9. 准备 LightGBM 训练数据（长格式） ====================
def panel_to_long(factors_dict, ret_df, dates, stocks):
    data = []
    for date in dates:
        for stock in stocks:
            f_vals = [factors_dict[name].loc[stock, date] for name in factors_dict]
            ret = ret_df.loc[stock, date]
            if np.isnan(ret) or any(np.isnan(f_vals)):
                continue
            data.append([date, stock, ret] + f_vals)
    columns = ['date', 'stock', 'ret'] + list(factors_dict.keys())
    return pd.DataFrame(data, columns=columns)

dates_ordered = sorted(common_dates_ret)
df_long = panel_to_long(factors, future_ret, dates_ordered, common_stocks)
print(f"长格式数据形状: {df_long.shape}")

# ==================== 10. 滚动预测 + 动态时间序列交叉验证（无泄露） + 重要性记录 ====================
n_fill = 6
pred_scores = []
unique_dates = dates_ordered

param_grid = {'num_leaves': [15,20], 'learning_rate': [0.03, 0.04, 0.05],
              'reg_lambda': [0.5, 1.0], 'reg_alpha': [0.0, 0.1, 0.3]}

base_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'n_estimators': 100
}

# 新增：存储特征重要性（gain）
importance_records = []

for i, test_date in enumerate(unique_dates):
    if i < n_fill:
        if test_date in score_hist.columns:
            pred_series = score_hist[test_date].copy()
        else:
            pred_series = pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        # 填充重要性为 NaN
        importance_records.append([np.nan] * len(factors))
        continue

    train_df = df_long[df_long['date'] < test_date].copy()
    if train_df.empty:
        pred_series = pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        importance_records.append([np.nan] * len(factors))
        continue

    test_df = df_long[df_long['date'] == test_date]
    if test_df.empty:
        pred_series = pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        importance_records.append([np.nan] * len(factors))
        continue

    X_train_all = train_df[list(factors.keys())].values
    y_train_all = train_df['ret'].values
    unique_train_months = train_df['date'].nunique()

    if unique_train_months < 6:
        best_params = base_params.copy()
        best_params.update({'num_leaves': 31, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'reg_alpha': 0.1})
    else:
        tscv = TimeSeriesSplit(n_splits=3)
        best_ic = -np.inf
        best_params = base_params.copy()
        best_params.update({'num_leaves': 31, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'reg_alpha': 0.1})
        for nl in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                for rl in param_grid['reg_lambda']:
                    for ra in param_grid['reg_alpha']:
                        params = base_params.copy()
                        params.update({'num_leaves': nl, 'learning_rate': lr, 'reg_lambda': rl, 'reg_alpha': ra})
                        fold_ics = []
                        for train_idx, val_idx in tscv.split(X_train_all):
                            X_cv_train, X_cv_val = X_train_all[train_idx], X_train_all[val_idx]
                            y_cv_train, y_cv_val = y_train_all[train_idx], y_train_all[val_idx]
                            model = lgb.LGBMRegressor(**params, random_state=42)
                            model.fit(X_cv_train, y_cv_train)
                            pred_val = model.predict(X_cv_val)
                            ic = spearmanr(pred_val, y_cv_val)[0]
                            if not np.isnan(ic):
                                fold_ics.append(ic)
                        avg_ic = np.mean(fold_ics) if fold_ics else -np.inf
                        if avg_ic > best_ic:
                            best_ic = avg_ic
                            best_params = params.copy()

    final_params = best_params.copy()
    final_params['n_estimators'] = 200
    final_model = lgb.LGBMRegressor(**final_params, random_state=42)
    final_model.fit(X_train_all, y_train_all)

    # 记录特征重要性（gain）
    importance_gain = final_model.booster_.feature_importance(importance_type='gain')
    importance_records.append(importance_gain)

    X_test = test_df[list(factors.keys())].values
    y_pred = final_model.predict(X_test)
    pred_series = pd.Series(y_pred, index=test_df['stock'].values, name=test_date)
    pred_scores.append(pred_series)

score_ml = pd.concat(pred_scores, axis=1)
score_ml = score_ml.reindex(columns=common_dates_ret)
print("机器学习合成因子形状:", score_ml.shape)

group_ml, ls_ml = group_backtest(score_ml, future_ret, n_groups=5)

# ==================== 11. 绩效对比 ====================
metrics_eq = calc_performance(ls_eq.dropna())
metrics_hist = calc_performance(ls_hist.dropna())
metrics_ml = calc_performance(ls_ml.dropna())

print("\n===== 策略绩效对比（完整区间）=====")
print(f"等权组合      : 年化收益={metrics_eq[0]:.2%}, 夏普={metrics_eq[2]:.2f}, 最大回撤={metrics_eq[3]:.2%}")
print(f"滚动IC加权组合    : 年化收益={metrics_hist[0]:.2%}, 夏普={metrics_hist[2]:.2f}, 最大回撤={metrics_hist[3]:.2%}")
print(f"LightGBM组合  : 年化收益={metrics_ml[0]:.2%}, 夏普={metrics_ml[2]:.2f}, 最大回撤={metrics_ml[3]:.2%}")

# ==================== 12. 特征重要性可视化 ====================
importance_df = pd.DataFrame(importance_records, index=unique_dates, columns=list(factors.keys()))
# 剔除前6个月（无模型）
importance_df_clean = importance_df.iloc[n_fill:].dropna(how='all')

plt.figure(figsize=(12,6))
for factor in importance_df_clean.columns:
    plt.plot(importance_df_clean.index, importance_df_clean[factor], marker='o', label=factor)
plt.title('LightGBM 特征重要性（gain）随时间变化（无正则化）')
plt.xlabel('月份')
plt.ylabel('重要性（累计增益）')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
for factor in importance_df_clean.columns:
    series = importance_df_clean[factor]
    if series.max() > series.min():
        normalized = (series - series.min()) / (series.max() - series.min())
    else:
        normalized = series
    plt.plot(importance_df_clean.index, normalized, marker='.', label=factor)
plt.title('特征重要性（归一化，无正则化）')
plt.xlabel('月份')
plt.ylabel('归一化重要性')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

rank_df = importance_df_clean.rank(axis=1, ascending=False, method='min')
plt.figure(figsize=(12,6))
for factor in rank_df.columns:
    plt.plot(rank_df.index, rank_df[factor], marker='.', label=factor)
plt.title('特征重要性排名变化（无正则化）')
plt.xlabel('月份')
plt.ylabel('排名')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==================== 13. 原有可视化（4图） ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

group_cum_ml = (1 + group_ml).cumprod(axis=1)
group_cum_ml.T.plot(ax=axes[0, 0])
axes[0, 0].set_title('LightGBM组合 - 分组累计净值（5组）')
axes[0, 0].set_ylabel('累计净值')
axes[0, 0].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'], loc='best')

group_cum_hist = (1 + group_hist).cumprod(axis=1)
group_cum_hist.T.plot(ax=axes[0, 1])
axes[0, 1].set_title('滚动IC加权组合 - 分组累计净值（5组）')
axes[0, 1].set_ylabel('累计净值')
axes[0, 1].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'], loc='best')

cum_eq_full = (1 + ls_eq).cumprod()
cum_hist_full = (1 + ls_hist).cumprod()
cum_ml_full = (1 + ls_ml).cumprod()
ax = axes[1, 0]
cum_eq_full.plot(ax=ax, label='等权')
cum_hist_full.plot(ax=ax, label='滚动IC加权')
cum_ml_full.plot(ax=ax, label='LightGBM', linestyle='--')
ax.axhline(1, color='gray', linestyle='--')
ax.set_title('完整区间多空累计净值（前6个月LightGBM=滚动IC加权）')
ax.legend()
ax.grid(True)

ax = axes[1, 1]
ax.plot(ls_eq.index, ls_eq.values, marker='o', linestyle='-', linewidth=1, markersize=3, label='等权')
ax.plot(ls_hist.index, ls_hist.values, marker='s', linestyle='-', linewidth=1, markersize=3, label='IC加权')
ax.plot(ls_ml.index, ls_ml.values, marker='^', linestyle='-', linewidth=1, markersize=3, label='LightGBM')
ax.axhline(0, color='black', linestyle='--')
ax.set_title('多空月度收益率对比')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()

# ==================== 14. 保存机器学习合成因子 ====================
try:
    import tushare as ts
    pro = ts.pro_api()
    stock_info = pro.stock_basic(fields='ts_code,name')
    name_map = dict(zip(stock_info['ts_code'], stock_info['name']))
    df_ml = score_ml.reset_index()
    df_ml.rename(columns={'index': 'stock_code'}, inplace=True)
    df_ml.insert(0, 'stock_name', df_ml['stock_code'].map(name_map))
    df_ml.to_csv('combined_factor_lightgbm_dynamic_cv_5groups.csv', index=False)
    print("机器学习合成因子已保存")
except:
    print("无法获取股票名称，保存无名称版本")
    score_ml.to_csv('4hyper_combined_factor_lightgbm_dynamic_cv_5groups.csv', index=True)

# 保存重要性数据
importance_df.to_csv('feature_importance_gain_no_reg.csv', index=True)
print("无正则化重要性时序数据已保存")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取五个因子面板 ====================
def load_factor(filepath, index_col='stock_code'):
    df = pd.read_csv(filepath, index_col=index_col)
    if 'stock_name' in df.columns:
        df = df.drop('stock_name', axis=1)
    df.columns = pd.to_datetime(df.columns)
    return df

mom = load_factor('动量因子_2024_2026_with_names.csv')
size = load_factor('规模因子_市值倒数_标准化_2024_2026.csv')
value = load_factor('价值因子_BP_processed.csv')
roe = load_factor('盈利因子_ROE_processed.csv')
cma = load_factor('投资因子_CMA_processed.csv')

common_dates = mom.columns.intersection(size.columns).intersection(value.columns).intersection(roe.columns).intersection(cma.columns)
common_dates = common_dates.sort_values()
common_stocks = mom.index.intersection(size.index).intersection(value.index).intersection(roe.index).intersection(cma.index)

mom = mom.loc[common_stocks, common_dates]
size = size.loc[common_stocks, common_dates]
value = value.loc[common_stocks, common_dates]
roe = roe.loc[common_stocks, common_dates]
cma = cma.loc[common_stocks, common_dates]

factors = {'动量': mom, '规模': size, '价值': value, '盈利': roe, '投资': cma}

# ==================== 2. 读取IC序列 ====================
ic_mom = pd.read_csv('ic_momentum.csv', index_col=0, squeeze=True)
ic_size = pd.read_csv('ic_size.csv', index_col=0, squeeze=True)
ic_value = pd.read_csv('ic_value.csv', index_col=0, squeeze=True)
ic_roe = pd.read_csv('ic_roe.csv', index_col=0, squeeze=True)
ic_cma = pd.read_csv('ic_cma.csv', index_col=0, squeeze=True)

for ic in [ic_mom, ic_size, ic_value, ic_roe, ic_cma]:
    ic.index = pd.to_datetime(ic.index)

ic_df = pd.DataFrame({'动量': ic_mom, '规模': ic_size, '价值': ic_value, '盈利': ic_roe, '投资': ic_cma})
ic_df = ic_df.loc[common_dates]

# ==================== 3. 准备未来收益数据 ====================
pivot = pd.read_csv('monthly_close_wide.csv', index_col='stock_code')
pivot.columns = pd.to_datetime(pivot.columns)
pivot = pivot.loc[common_stocks, common_dates]
future_ret = pivot.pct_change(axis=1).shift(-1, axis=1)
future_ret = future_ret.iloc[:, :-1]
common_dates_ret = future_ret.columns
for name in factors:
    factors[name] = factors[name][common_dates_ret]
ic_df = ic_df.loc[common_dates_ret]

# ==================== 4. 等权组合权重 ====================
weights_equal = pd.DataFrame(1/len(factors), index=common_dates_ret, columns=list(factors.keys()))

# ==================== 5. IC加权组合权重（滚动历史均值） ====================
ic_shifted = ic_df.shift(1)
rolling_ic_mean = ic_shifted.rolling(window=12, min_periods=1).mean()
abs_sum = rolling_ic_mean.abs().sum(axis=1)
weights_hist = rolling_ic_mean.div(abs_sum, axis=0).fillna(1/len(factors))

# ==================== 6. 合成得分 ====================
def combine_factors(factors, weights_df):
    combined = pd.DataFrame(index=factors[list(factors.keys())[0]].index, columns=weights_df.index)
    for date in weights_df.index:
        w = weights_df.loc[date]
        score = sum(w[name] * factors[name][date] for name in factors)
        combined[date] = score
    return combined

score_equal = combine_factors(factors, weights_equal)
score_hist = combine_factors(factors, weights_hist)

# ==================== 7. 分组回测（5组） ====================
def group_backtest(score_df, ret_df, n_groups=5):
    group_returns = []
    for month in score_df.columns:
        factor = score_df[month].dropna()
        ret = ret_df[month].loc[factor.index]
        if len(factor) == 0:
            continue
        try:
            groups = pd.qcut(factor.rank(method='first'), n_groups, labels=False) + 1
        except:
            groups = pd.cut(factor, n_groups, labels=False) + 1
        group_mean = ret.groupby(groups).mean()
        group_mean.name = month
        group_returns.append(group_mean)
    group_ret_df = pd.DataFrame(group_returns).T
    long_short = group_ret_df.loc[n_groups] - group_ret_df.loc[1]
    return group_ret_df, long_short

group_eq, ls_eq = group_backtest(score_equal, future_ret, n_groups=5)
group_hist, ls_hist = group_backtest(score_hist, future_ret, n_groups=5)

# ==================== 8. 绩效指标函数 ====================
def calc_performance(returns_series):
    if returns_series.empty or returns_series.isna().all():
        return (np.nan,)*5
    annual_return = (1 + returns_series).prod() ** (12 / len(returns_series)) - 1
    annual_vol = returns_series.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    cum = (1 + returns_series).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    win_rate = (returns_series > 0).mean()
    return annual_return, annual_vol, sharpe, max_dd, win_rate

# ==================== 9. 准备长格式数据 ====================
def panel_to_long(factors_dict, ret_df, dates, stocks):
    data = []
    for date in dates:
        for stock in stocks:
            f_vals = [factors_dict[name].loc[stock, date] for name in factors_dict]
            ret = ret_df.loc[stock, date]
            if np.isnan(ret) or any(np.isnan(f_vals)):
                continue
            data.append([date, stock, ret] + f_vals)
    columns = ['date', 'stock', 'ret'] + list(factors_dict.keys())
    return pd.DataFrame(data, columns=columns)

dates_ordered = sorted(common_dates_ret)
df_long = panel_to_long(factors, future_ret, dates_ordered, common_stocks)
print(f"长格式数据形状: {df_long.shape}")

# ==================== 10. 滚动预测 + 动态交叉验证 + 正则化 + 特征重要性记录 ====================
n_fill = 6
pred_scores = []
unique_dates = dates_ordered

# 超参数搜索空间（保持原样，不扩大）
param_grid = {
    'num_leaves': [15, 20],
    'learning_rate': [0.03, 0.04, 0.05],
    'reg_lambda': [0.5, 1.0],
    'reg_alpha': [0.0, 0.1, 0.3]
}

# 基础参数：加入固定 gamma=0.1（正则化）
base_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'bagging_freq': 5,
    'verbosity': -1,
    'n_estimators': 100,
    'gamma': 0.1
}

importance_records = []

for i, test_date in enumerate(unique_dates):
    if i < n_fill:
        pred_series = score_hist[test_date].copy() if test_date in score_hist.columns else pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        importance_records.append([np.nan] * len(factors))
        continue

    train_df = df_long[df_long['date'] < test_date].copy()
    if train_df.empty:
        pred_series = pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        importance_records.append([np.nan] * len(factors))
        continue

    test_df = df_long[df_long['date'] == test_date]
    if test_df.empty:
        pred_series = pd.Series(np.nan, index=common_stocks, name=test_date)
        pred_scores.append(pred_series)
        importance_records.append([np.nan] * len(factors))
        continue

    X_train_all = train_df[list(factors.keys())].values
    y_train_all = train_df['ret'].values
    unique_train_months = train_df['date'].nunique()

    if unique_train_months < 6:
        best_params = base_params.copy()
        best_params.update({'num_leaves': 20, 'learning_rate': 0.03, 'reg_lambda': 1.0, 'reg_alpha': 0.3,
                            'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7})
    else:
        tscv = TimeSeriesSplit(n_splits=3)
        best_ic = -np.inf
        best_params = base_params.copy()
        best_params.update({'num_leaves': 20, 'learning_rate': 0.03, 'reg_lambda': 1.0, 'reg_alpha': 0.3,
                            'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7})
        # 注意：以下循环中固定了 mcs=20, sub=1.0, col=1.0（与原代码一致）
        # 这里保持原代码的硬编码，不改变搜索空间大小
        mcs = 20
        sub = 1.0
        col = 1.0
        for nl in param_grid['num_leaves']:
            for lr in param_grid['learning_rate']:
                for rl in param_grid['reg_lambda']:
                    for ra in param_grid['reg_alpha']:
                        params = base_params.copy()
                        params.update({
                            'num_leaves': nl, 'learning_rate': lr, 'reg_lambda': rl,
                            'reg_alpha': ra, 'min_child_samples': mcs,
                            'subsample': sub, 'colsample_bytree': col
                        })
                        fold_ics = []
                        for train_idx, val_idx in tscv.split(X_train_all):
                            X_cv_train, X_cv_val = X_train_all[train_idx], X_train_all[val_idx]
                            y_cv_train, y_cv_val = y_train_all[train_idx], y_train_all[val_idx]
                            model = lgb.LGBMRegressor(**params, random_state=42)
                            model.fit(X_cv_train, y_cv_train)
                            pred_val = model.predict(X_cv_val)
                            ic = spearmanr(pred_val, y_cv_val)[0]
                            if not np.isnan(ic):
                                fold_ics.append(ic)
                        avg_ic = np.mean(fold_ics) if fold_ics else -np.inf
                        if avg_ic > best_ic:
                            best_ic = avg_ic
                            best_params = params.copy()

    # ========== 最终模型训练：使用早停 ==========
    train_dates = sorted(train_df['date'].unique())
    if len(train_dates) >= 2:
        val_date = train_dates[-1]
        train_no_val = train_df[train_df['date'] < val_date]
        val_df = train_df[train_df['date'] == val_date]
        if not train_no_val.empty and not val_df.empty:
            X_train_final = train_no_val[list(factors.keys())].values
            y_train_final = train_no_val['ret'].values
            X_val = val_df[list(factors.keys())].values
            y_val = val_df['ret'].values
            eval_set = [(X_val, y_val)]
            early_stop_rounds = 20
        else:
            X_train_final, y_train_final = X_train_all, y_train_all
            eval_set = None
            early_stop_rounds = None
    else:
        X_train_final, y_train_final = X_train_all, y_train_all
        eval_set = None
        early_stop_rounds = None

    final_params = best_params.copy()
    final_params['n_estimators'] = 500
    final_params['early_stopping_rounds'] = early_stop_rounds
    final_model = lgb.LGBMRegressor(**final_params, random_state=42)

    if eval_set is not None:
        final_model.fit(X_train_final, y_train_final, eval_set=eval_set,
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False)])
    else:
        final_model.fit(X_train_final, y_train_final)

    # 提取特征重要性（gain）
    importance_gain = final_model.booster_.feature_importance(importance_type='gain')
    importance_records.append(importance_gain)

    X_test = test_df[list(factors.keys())].values
    y_pred = final_model.predict(X_test)
    pred_series = pd.Series(y_pred, index=test_df['stock'].values, name=test_date)
    pred_scores.append(pred_series)

# 合并得分
score_ml = pd.concat(pred_scores, axis=1).reindex(columns=common_dates_ret)
print("机器学习合成因子形状:", score_ml.shape)

# ==================== 11. 特征重要性可视化 ====================
importance_df = pd.DataFrame(importance_records, index=unique_dates, columns=list(factors.keys()))
importance_df_clean = importance_df.iloc[n_fill:].dropna(how='all')

plt.figure(figsize=(12, 6))
for factor in importance_df_clean.columns:
    plt.plot(importance_df_clean.index, importance_df_clean[factor], marker='o', label=factor)
plt.title('LightGBM 特征重要性（gain）随时间变化（有正则化: gamma=0.1, 早停）')
plt.xlabel('月份')
plt.ylabel('重要性（累计增益）')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 归一化图
plt.figure(figsize=(12, 6))
for factor in importance_df_clean.columns:
    series = importance_df_clean[factor]
    if series.max() > series.min():
        normalized = (series - series.min()) / (series.max() - series.min())
    else:
        normalized = series
    plt.plot(importance_df_clean.index, normalized, marker='.', label=factor)
plt.title('特征重要性（归一化，有正则化）')
plt.xlabel('月份')
plt.ylabel('归一化重要性')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 排名变化图
rank_df = importance_df_clean.rank(axis=1, ascending=False, method='min')
plt.figure(figsize=(12, 6))
for factor in rank_df.columns:
    plt.plot(rank_df.index, rank_df[factor], marker='.', label=factor)
plt.title('特征重要性排名变化（有正则化）')
plt.xlabel('月份')
plt.ylabel('排名')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==================== 12. 机器学习回测与绩效 ====================
group_ml, ls_ml = group_backtest(score_ml, future_ret, n_groups=5)

metrics_eq = calc_performance(ls_eq.dropna())
metrics_hist = calc_performance(ls_hist.dropna())
metrics_ml = calc_performance(ls_ml.dropna())

print("\n===== 策略绩效对比（完整区间）=====")
print(f"等权组合        : 年化收益={metrics_eq[0]:.2%}, 夏普={metrics_eq[2]:.2f}, 最大回撤={metrics_eq[3]:.2%}")
print(f"滚动IC加权组合  : 年化收益={metrics_hist[0]:.2%}, 夏普={metrics_hist[2]:.2f}, 最大回撤={metrics_hist[3]:.2%}")
print(f"LightGBM组合    : 年化收益={metrics_ml[0]:.2%}, 夏普={metrics_ml[2]:.2f}, 最大回撤={metrics_ml[3]:.2%}")

# ==================== 13. 可视化（原有4图） ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

(1 + group_ml).cumprod(axis=1).T.plot(ax=axes[0,0])
axes[0,0].set_title('LightGBM组合 - 分组累计净值（5组）')
axes[0,0].legend(title='分组', labels=['组1(低)','组2','组3','组4','组5(高)'], loc='best')

(1 + group_hist).cumprod(axis=1).T.plot(ax=axes[0,1])
axes[0,1].set_title('滚动IC加权组合 - 分组累计净值（5组）')
axes[0,1].legend(title='分组', labels=['组1(低)','组2','组3','组4','组5(高)'], loc='best')

(1 + ls_eq).cumprod().plot(ax=axes[1,0], label='等权')
(1 + ls_hist).cumprod().plot(ax=axes[1,0], label='滚动IC加权')
(1 + ls_ml).cumprod().plot(ax=axes[1,0], label='LightGBM', linestyle='--')
axes[1,0].axhline(1, color='gray', linestyle='--')
axes[1,0].set_title('完整区间多空累计净值（前6个月LightGBM=滚动IC加权）')
axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(ls_eq.index, ls_eq.values, marker='o', label='等权')
axes[1,1].plot(ls_hist.index, ls_hist.values, marker='s', label='IC加权')
axes[1,1].plot(ls_ml.index, ls_ml.values, marker='^', label='LightGBM')
axes[1,1].axhline(0, color='black', linestyle='--')
axes[1,1].set_title('多空月度收益率对比')
axes[1,1].legend(); axes[1,1].grid(True)

plt.tight_layout()
plt.show()

# ==================== 14. 保存结果 ====================
try:
    import tushare as ts
    pro = ts.pro_api()
    stock_info = pro.stock_basic(fields='ts_code,name')
    name_map = dict(zip(stock_info['ts_code'], stock_info['name']))
    df_ml = score_ml.reset_index().rename(columns={'index': 'stock_code'})
    df_ml.insert(0, 'stock_name', df_ml['stock_code'].map(name_map))
    df_ml.to_csv('combined_factor_lightgbm_regularized_fixed_gamma.csv', index=False)
    print("机器学习合成因子已保存")
except:
    score_ml.to_csv('4hyper_combined_factor_lightgbm_regularized_fixed_gamma.csv', index=True)

importance_df.to_csv('feature_importance_gain_timeseries_with_reg.csv', index=True)
print("有正则化重要性时序数据已保存")