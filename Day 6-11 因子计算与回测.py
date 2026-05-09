import tushare as ts
import numpy as np
import pandas as pd
import time
ts.set_token("  ")
pro = ts.pro_api()


#获取20230101到20260331一直在沪深300里的股票名单，并输出到沪深300_202301_202603.csv
'''
start_date = '20230101'
end_date = '20260331'

# 生成每个月第一天的日期列表
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y%m%d').tolist()

# 1. 获取每个月的成分股
all_stocks = set()
monthly_data = {}
for date in monthly_dates:
    # 获取当月数据，trade_date需为当月某一天
    df = pro.index_weight(index_code='399300.SZ', start_date=date, end_date=date)
    if not df.empty:
        stocks = set(df['con_code'].tolist())
        monthly_data[date] = stocks
        all_stocks.update(stocks)
# 2. 找出一直在的成分股
persistent_stocks = []
for stock in all_stocks:
    is_always_in = True
    for date, stocks in monthly_data.items():
        if stock not in stocks:
            is_always_in = False
            break
    if is_always_in:
        persistent_stocks.append(stock)
print(f"一直在沪深300中的股票数量: {len(persistent_stocks)}")
series_persistent_stocks=pd.Series(persistent_stocks)
series_persistent_stocks.to_csv("沪深300_202301_202603.csv")
'''
#resample重采样 rolling滚动计算
'''
#将日期列设为索引、重采样、频率转换、移位和收益率计算
dates = pd.date_range('20230101', periods=100, freq='D')   #这样生成的index是时间戳索引
#df = pro.daily(ts_code='600593.SH', start_date='20230101', end_date='20260315').sort_values('trade_date').set_index('trade_date')
#df.to_csv("圣亚.csv")
df=pd.read_csv("圣亚.csv")       #日线处理方法：1.按交易日期排序 2.转换成时间戳index 3.注意使用后复权数据
df.trade_date=pd.to_datetime(df.trade_date,format="%Y%m%d")
df=df.set_index('trade_date')
print(df.loc['2023-01-13'])
#print(df.loc['2023-02'])      #选出来每天或者每个月的数据

#重采样（Resampling）是指将时间序列数据从一种频率转换为另一种频率的过程。在 Pandas 中，resample() 用于重采样。
# 可以指定目标频率（比如按周、按月、按季度等），然后对每个时间段内的数据进行聚合计算（如求和、均值、最大值等）。
df.resample('W').mean() #该序列每周的均值 M Q A分别是月度 季度 年度   每周的均值数据  注意 这不能计算周线 只是每周各个值的均值
dd=df.resample('W-FRI').last()   #每周最后一个交易日  注意 数值是对的，但是日期，未考虑节假日
df.resample('M').last()
df.shift(1)  #整体下移一行  -1就是上移一行
df.close.pct_change()  #手动计算收盘的计算百分比变化
# 计算过去5天累计收益率（滚动5日收益，即今天相对于5天前的收益率）
df['close'].pct_change(periods=5)    #periods是调整pct_change相对之前多少期的具体参数

#rolling滚动计算
df['pct_chg'].rolling(window=5).mean()
df['pct_chg'].rolling(window=20).std()  #滚动单日收益率相关指标
df['open'].rolling(window=20).corr(df['close']) #实际中一般是两个股票的滚动相关系数
df['pct_chg'].rolling(window=20).std()*np.sqrt(252) #20日滚动年化波动率
df['pct_chg'].transform(lambda p:p.rolling(20).std() * np.sqrt(252)) #单一序列时，完全一样，多序列groupby时只能用transform

#多序列计算  大连圣亚 长白山
df2 = pro.daily(ts_code='600593.SH,603099.SH', start_date='20260101', end_date='20260315').sort_values('trade_date').set_index('trade_date')
#滚动相关系数
df2[df2['ts_code']=='600593.SH'].pct_chg.rolling(window=20).corr(df2[df2['ts_code']=='603099.SH'].pct_chg)

#分别算波动率    添加成新的列 方便分别查看
df2['vol_20d'] = df2.groupby('ts_code')['pct_chg'].transform(
    lambda x: x.rolling(20).std() * np.sqrt(252)
)
# 查看股票 A 的结果     分别看调出来的各个股票的波动率
df2[df2['ts_code'] == '600593.SH'][['pct_chg', 'vol_20d']]
# 查看股票 B 的结果
df2[df2['ts_code'] == '603099.SH'][['pct_chg', 'vol_20d']]
'''

#计算筛选出来的股票的动量值
'''
# 1. 读取一直存在的股票列表
stock_df = pd.read_csv('沪深300_202301_202603.csv', index_col=0)
stock_list = stock_df.iloc[:, 0].tolist()
print(f"共{len(stock_list)}只股票")

# ========== 3. 获取沪深300指数月线作为交易日历 ==========
print("获取沪深300指数月线，用作交易日历...")
index_df = ts.pro_bar(ts_code='000300.SH', asset='I', freq='M',
                      start_date='20230101', end_date='20260331')
if index_df.empty:
    raise ValueError("无法获取沪深300指数月线数据")
index_df['trade_date'] = pd.to_datetime(index_df['trade_date'])
calendar = sorted(index_df['trade_date'].unique())
print(f"日历共 {len(calendar)} 个月份")

# ========== 4. 获取每只股票的月线后复权数据，对齐日历并前向填充 ==========
all_monthly = []
for i, code in enumerate(stock_list):
    if i % 20 == 0:
        print(f"正在处理第 {i+1}/{len(stock_list)} 只股票...")
    try:
        df = ts.pro_bar(ts_code=code, adj='hfq', freq='M',
                        start_date='20230101', end_date='20260331')
        if df is None or df.empty:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df[['trade_date', 'close']].copy()
        df.set_index('trade_date', inplace=True)
        df = df.reindex(calendar)
        df['close'] = df['close'].fillna(method='ffill')
        df.dropna(subset=['close'], inplace=True)
        df['ts_code'] = code
        all_monthly.append(df.reset_index().rename(columns={'index': 'trade_date'}))
        time.sleep(0.1)
    except Exception as e:
        print(f"处理 {code} 时出错: {e}")
        continue

if not all_monthly:
    raise ValueError("没有成功获取任何股票数据")

monthly_all = pd.concat(all_monthly, ignore_index=True)
print(f"成功获取 {monthly_all['ts_code'].nunique()} 只股票的完整月线序列")

# ========== 5. 计算动量因子（12-1） ==========
pivot = monthly_all.pivot(index='ts_code', columns='trade_date', values='close')
pivot = pivot.sort_index(axis=1)

momentum_wide = pd.DataFrame(index=pivot.index)
month_dates = pivot.columns.tolist()
for idx in range(13, len(month_dates)):
    close_lag1 = pivot.iloc[:, idx-1]
    close_lag13 = pivot.iloc[:, idx-13]
    factor = (close_lag1 / close_lag13) - 1
    momentum_wide[month_dates[idx]] = factor

# ========== 6. 筛选目标时间范围（2024年2月 至 2026年3月） ==========
target_start = pd.Timestamp('2024-02-29')
target_end = pd.Timestamp('2026-03-31')
target_dates = [d for d in month_dates if target_start <= d <= target_end]
if not target_dates:
    target_dates = [d for d in month_dates if d >= pd.Timestamp('2024-02-01') and d <= pd.Timestamp('2026-03-31')]

momentum_target = momentum_wide[target_dates]
print(f"目标月份数: {len(target_dates)}")

# ========== 7. 准备结果DataFrame ==========
# 原股票列表（只有股票代码）
result = pd.DataFrame({stock_df.columns[0]: stock_list})  # 列名如 0
result.set_index(result.columns[0], inplace=True)        # 索引为股票代码
result.index.name = 'stock_code'
result = result.join(momentum_target, how='left')
result.reset_index(inplace=True)  # 此时列名为 ['index', 其他日期列...]
result.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in result.columns]

#input_file = '动量因子_2024_2026.csv'   # 请修改为你的实际文件名

df = result

# 3. 获取所有股票代码（假设第一列列名为 'index' 或实际是 'Unnamed: 0'？需要确认）
# 根据之前输出，第一列可能是 'index'（重置索引后的列名），也可能是股票代码列名。
# 我们假定第一列的名字是 'index'，其中存储的是股票代码。但更保险的方式：取第一列。
first_col = df.columns[0]   # 获取第一列的列名
stock_codes = df[first_col].tolist()

print(f"共 {len(stock_codes)} 只股票，获取名称中...")
pivot.to_csv('monthly_close_wide.csv')
# 4. 调用 tushare 获取股票名称
# 注意：stock_basic 接口每次最多返回约 5000 条，足够覆盖所有 A 股
stock_info = pro.stock_basic(fields='ts_code,name')
# 创建代码到名称的映射字典
name_map = dict(zip(stock_info['ts_code'], stock_info['name']))

# 5. 添加股票名称列
df.insert(0, 'stock_name', df[first_col].map(name_map))

# 6. 保存到新文件（或覆盖原文件）
output_file = '动量因子_2024_2026_with_names.csv'
df.to_csv(output_file, index=False)
print(f"已保存至 {output_file}，新列 '股票名称' 已添加在最前面。")
'''
#计算因子值 账面市值比
'''
from datetime import datetime
# ========== 2. 读取股票列表 ==========
stock_df = pd.read_csv('沪深300_202301_202603.csv', index_col=0)
stock_list = stock_df.iloc[:, 0].tolist()
print(f"共 {len(stock_list)} 只股票")

# ========== 3. 获取交易日历（每月最后一个交易日） ==========
print("获取交易日历...")
cal_df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20260331')
cal_df = cal_df[cal_df['is_open'] == 1]
cal_df['cal_date'] = pd.to_datetime(cal_df['cal_date'])
cal_df['year_month'] = cal_df['cal_date'].dt.to_period('M')
calendar = cal_df.groupby('year_month')['cal_date'].max().tolist()
print(f"共 {len(calendar)} 个月份")

# ========== 4. 获取 pb 数据（仅限你的股票列表） ==========
all_pb_data = []
trade_dates_needed = [d.strftime('%Y%m%d') for d in calendar]

# 方法：逐只股票获取（避免全市场数据），因为你的股票只有242只，可以接受
for code in stock_list:
    print(f"正在获取 {code} 的pb数据...")
    try:
        # 一次性获取该股票在所有需要日期的 pb
        df = pro.daily_basic(ts_code=code, start_date='20240101', end_date='20260331',
                             fields='trade_date,pb')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['ts_code'] = code
            all_pb_data.append(df)
    except Exception as e:
        print(f"获取 {code} pb数据失败: {e}")
    time.sleep(0.5)  # 控制频率

if not all_pb_data:
    raise ValueError("没有获取到任何pb数据")

pb_all = pd.concat(all_pb_data, ignore_index=True)

# ========== 5. 筛选出每月最后一个交易日的pb值 ==========
# 只保留 calendar 中的日期
pb_monthly = pb_all[pb_all['trade_date'].isin(calendar)]

# ========== 6. 转换为宽表（行=股票，列=月份，值=pb） ==========
pivot_pb = pb_monthly.pivot(index='ts_code', columns='trade_date', values='pb')

# ========== 7. 计算价值因子 BP = 1 / PB ==========
pivot_bp = 1 / pivot_pb

# ========== 8. 确保只包含你的股票列表（理论上已经包含，但做一次索引对齐） ==========
pivot_bp = pivot_bp.reindex(stock_list)

# ========== 9. 添加股票名称 ==========
stock_info = pro.stock_basic(fields='ts_code,name')
name_map = dict(zip(stock_info['ts_code'], stock_info['name']))
result = pivot_bp.reset_index()
result.rename(columns={result.columns[0]: 'stock_code'}, inplace=True)
result.insert(0, 'stock_name', result['stock_code'].map(name_map).fillna(''))

# ========== 10. 保存结果 ==========
result.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in result.columns]
result.to_csv('价值因子_BP_2024_2026_仅股票列表.csv', index=False)
print("价值因子计算完成，已保存至 价值因子_BP_2024_2026_仅股票列表.csv")
print(result.head())
'''

#计算规模因子 市值的倒数
'''
stock_df = pd.read_csv('沪深300_202301_202603.csv', index_col=0)
stock_list = stock_df.iloc[:, 0].tolist()
print(f"共 {len(stock_list)} 只股票")

# ========== 2. 获取交易日历（每月最后一个交易日） ==========
print("获取交易日历...")
cal_df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20260331')
cal_df = cal_df[cal_df['is_open'] == 1]
cal_df['cal_date'] = pd.to_datetime(cal_df['cal_date'])
cal_df['year_month'] = cal_df['cal_date'].dt.to_period('M')
calendar = cal_df.groupby('year_month')['cal_date'].max().tolist()
print(f"共 {len(calendar)} 个月份")

# ========== 3. 获取总市值数据（每月最后一个交易日） ==========
all_mv_data = []
for code in stock_list:
    print(f"正在获取 {code} 市值数据...")
    try:
        df = pro.daily_basic(ts_code=code, start_date='20240101', end_date='20260331',
                             fields='trade_date,total_mv')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['ts_code'] = code
            all_mv_data.append(df)
    except Exception as e:
        print(f"获取 {code} 失败: {e}")
    time.sleep(0.3)   # 控制请求频率

if not all_mv_data:
    raise ValueError("没有获取到任何市值数据")

mv_all = pd.concat(all_mv_data, ignore_index=True)

# ========== 4. 筛选每月最后一个交易日 ==========
mv_monthly = mv_all[mv_all['trade_date'].isin(calendar)]

# ========== 5. 转换为宽表（行=股票，列=月份，值=总市值，单位：万元） ==========
pivot_mv = mv_monthly.pivot(index='ts_code', columns='trade_date', values='total_mv')
pivot_mv = pivot_mv.reindex(stock_list)   # 确保股票顺序与列表一致

pivot_mv_filled = pivot_mv.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
# ========== 6. 计算市值倒数 ==========
# 注意：市值可能为0或缺失，先处理缺失值（填充NaN，后面去极值时会处理）
pivot_inv_mv = 1.0 / pivot_mv_filled   # 得到市值倒数，数值可能极大

# ========== 7. 去极值（，逐月截面） ==========
def winsorize_mad(series, n=5):
    """MAD法去极值：中位数 ± n * MAD"""
    median = series.median()
    mad = (series - median).abs().median()
    upper = median + n * mad
    lower = median - n * mad
    return series.clip(lower, upper)

# apply axis=1 表示按行（每个日期截面）应用函数
pivot_inv_mv_winsor = pivot_inv_mv.apply(lambda x: winsorize_mad(x), axis=1)

# ========== 8. 标准化（Z-score，逐月截面） ==========
def standardize(series):
    """截面标准化：减去均值除以标准差"""
    s_clean = series.dropna()
    if len(s_clean) < 5:
        return series
    mean = s_clean.mean()
    std = s_clean.std()
    return (series - mean) / std

pivot_size_factor = pivot_inv_mv_winsor.apply(lambda x: standardize(x), axis=1)

# 此时 pivot_size_factor 的值：越大表示市值越小（小盘股得分高），可直接用于多因子合成（正向选股）

# ========== 9. 添加股票名称并保存 ==========
stock_info = pro.stock_basic(fields='ts_code,name')
name_map = dict(zip(stock_info['ts_code'], stock_info['name']))
result = pivot_size_factor.reset_index()
result.rename(columns={result.columns[0]: 'stock_code'}, inplace=True)
result.insert(0, 'stock_name', result['stock_code'].map(name_map).fillna(''))

# 将列名中的 Timestamp 格式化为字符串
result.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in result.columns]

result.to_csv('规模因子_市值倒数_标准化_2024_2026.csv', index=False)
print("规模因子（市值倒数）处理完成，已保存至 规模因子_市值倒数_标准化_2024_2026.csv")
print(result.head())
'''
#因子的评估 多空组合的收益指标
'''
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取动量因子宽表（已包含股票名称和代码）
momentum = pd.read_csv('价值因子_BP_processed.csv', index_col='stock_code')
momentum=momentum.fillna(0)
# 删除前两列（股票名称和股票代码列名，注意调整）
# 假设 momentum 的前两列是 'stock_name' 和 'stock_code'，日期列从第三列开始
date_cols = [col for col in momentum.columns if col not in ['stock_name']]
momentum_wide = momentum[date_cols]  # 纯因子宽表

# 2. 获取未来收益（假设你已经有 pivot 表）
# 如果还没有 pivot，需要重新构建（参考之前动量因子的代码，获取所有股票月线后复权收盘价）
# 这里假设 pivot 已经存在，且列名与 momentum_wide 一致
pivot = pd.read_csv('monthly_close_wide.csv', index_col='stock_code')

# 计算未来收益
future_ret = pd.DataFrame(index=pivot.index)
for i in range(len(pivot.columns)-1):
    current = pivot.columns[i]
    next_ = pivot.columns[i+1]
    future_ret[current] = (pivot[next_] / pivot[current]) - 1

momentum_wide.columns = pd.to_datetime(momentum_wide.columns)
future_ret.columns = pd.to_datetime(future_ret.columns)

# 3. 对齐月份（只取两者都有的月份）
common_months = momentum_wide.columns.intersection(future_ret.columns)
momentum_wide = momentum_wide[common_months]
future_ret = future_ret[common_months]

# 4. 分组分析
group_returns = []
for month in common_months:
    factor = momentum_wide[month].dropna()  # 该月所有股票的因子值
    ret = future_ret[month].loc[factor.index]  # 对应的未来收益
    # 按因子值降序排序，取前43个索引
    #top43_idx = factor.sort_values(ascending=False).head(43).index
    # 从 factor 和 ret 中剔除这些股票
    #factor = factor.drop(top43_idx)
    ret = future_ret[month].loc[factor.index]
    # 分组
    try:
        groups = pd.qcut(factor.rank(method='first'), 5, labels=False) + 1
    except:
        groups = pd.cut(factor, 5, labels=False) + 1
    group_mean = ret.groupby(groups).mean()
    group_mean.name = month
    group_returns.append(group_mean)

group_ret_df = pd.DataFrame(group_returns).T

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：5条分组净值曲线
group_cumulative = (1 + group_ret_df).cumprod(axis=1)   # 行=组别，列=月份
group_cumulative.T.plot(ax=ax1)   # 转置后行=月份，列=组别
ax1.set_title('分组累计净值曲线')
ax1.set_xlabel('月份')
ax1.set_ylabel('累计净值')
ax1.legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'])
# 右图：多空组合累计净值曲线
long_short = group_ret_df.loc[5] - group_ret_df.loc[1]
cumulative = (1 + long_short).cumprod()
cumulative.plot(ax=ax2)
ax2.set_title('多空组合累计净值')
ax2.set_xlabel('截面日期')
ax2.set_ylabel('累计净值')
plt.tight_layout()
plt.show()

# 假设 long_short 是多空组合的月度收益率序列（Series，索引为月份）
# 年化收益率
annual_return = (1 + long_short).prod() ** (12 / len(long_short)) - 1
print(annual_return)
# 年化波动率
annual_vol = long_short.std() * (12 ** 0.5)
print(annual_vol)
# 夏普比率（假设无风险利率为0）
sharpe = annual_return / annual_vol
print(sharpe)
# 最大回撤
cumulative = (1 + long_short).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
print(max_drawdown)
# 胜率（收益为正的月份占比）
win_rate = (long_short > 0).mean()
print(win_rate)
'''
# 可视化因子分布
'''
df = pd.read_csv('价值因子_BP_processed.csv', index_col='stock_code')
# 去掉股票名称列（如果存在）
if 'stock_name' in df.columns:
    df = df.drop('stock_name', axis=1)
# 确保列名是 datetime 类型（便于筛选）
df.columns = pd.to_datetime(df.columns)
plot_months = df.columns[-12:]   # 最后12个月
plot_data = []
for m in plot_months:
    tmp = df[m].dropna().to_frame(name='factor')
    tmp['date'] = m.strftime('%Y-%m')
    plot_data.append(tmp)
plot_df = pd.concat(plot_data)

plt.figure(figsize=(14, 6))
sns.boxplot(data=plot_df, x='date', y='factor')
plt.xticks(rotation=45)
plt.title('动量因子分布随时间变化（箱线图）')
plt.xlabel('月份')
plt.ylabel('因子值')
plt.show()
'''
#查找20260129全市场上涨2.69个点的股票
'''
target_date = '20260129'  # 注意格式：YYYYMMDD
target_pct = 2.69
df_daily = pro.daily(trade_date=target_date)
result = df_daily[abs(df_daily['pct_chg'] - target_pct) < 0.005]  # 允许±0.005的误差

if result.empty:
    print(f"没有找到在 {target_date} 涨跌幅精确等于 {target_pct}% 的股票。")
    # 可选：查看最接近的几只
    closest = df_daily.iloc[(df_daily['pct_chg'] - target_pct).abs().argsort()[:5]]
    print("\n涨跌幅最接近的5只股票：")
    print(closest[['ts_code', 'name', 'pct_chg']])
else:
    print(f"\n找到 {len(result)} 只符合条件的股票：")
    # 显示股票代码、名称、涨跌幅、收盘价等关键信息
    print(result[['ts_code', 'pct_chg', 'close', 'vol']])
'''