import tushare as ts
import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
import alphalens as al
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
ts.set_token(" ")
pro = ts.pro_api()

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


#标准化等预处理之后的因子
import statsmodels.api as sm
from scipy.stats import rankdata
import warnings

warnings.filterwarnings('ignore')

# 假设你已经有了 pivot_bp (DataFrame, index=股票代码, columns=日期, values=BP因子)
# 如果还没有，可以按你之前的代码读取保存的CSV文件：
pivot_bp = pd.read_csv('价值因子_BP_2024_2026_仅股票列表.csv', index_col='stock_code')
# 但注意你保存时第一列是stock_code，第二列是stock_name，需要处理
# 建议直接使用之前生成的 pivot_bp 变量，或者重新构建


stock_basic = pro.stock_basic(fields='ts_code,industry')
# 注意：tushare的industry是申万一级行业，可能部分股票缺失，可改用其他接口
industry_map = dict(zip(stock_basic['ts_code'], stock_basic['industry']))

# 获取市值数据（每月最后一个交易日）
# 需要获取每只股票在calendar日期的总市值（total_mv）
# 你可以复用之前获取pb的日历列表 calendar (list of Timestamp)
# 注意：市值需要从 daily_basic 获取 total_mv 字段

# 方法：逐只股票获取市值（类似之前获取pb的方式），然后透视成宽表
all_mv_data = []
for trade_date in calendar:  # calendar 是你的月末交易日列表
    date_str = trade_date.strftime('%Y%m%d')
    print(f"获取 {date_str} 市值数据...")
    try:
        df = pro.daily_basic(
            trade_date=date_str,
            fields='ts_code,total_mv'
        )
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(date_str)
            all_mv_data.append(df)
    except Exception as e:
        print(f"获取 {date_str} 市值数据失败: {e}")
    time.sleep(0.3)  # 频率控制

mv_all = pd.concat(all_mv_data, ignore_index=True)
# 筛选每月最后一个交易日（calendar 中日期）
mv_monthly = mv_all[mv_all['trade_date'].isin(calendar)]
# 透视：行=股票，列=日期，值=total_mv
pivot_mv = mv_monthly.pivot(index='ts_code', columns='trade_date', values='total_mv')
pivot_mv = pivot_mv.reindex(pivot_bp.index)  # 对齐股票顺序


# ========== 2. 预处理函数 ==========
def process_monthly(factor_series, mv_series, industry_map, method='3sigma'):
    """
    对单个月份的截面因子进行处理
    factor_series: Series, index=股票代码, values=因子值（如BP）
    mv_series: Series, index=股票代码, values=总市值（单位万元）
    industry_map: dict, 股票代码 -> 行业名称
    return: 处理后的因子Series (index同输入)
    """
    # 合并数据
    df = pd.DataFrame({'factor': factor_series, 'mv': mv_series})
    df['industry'] = df.index.map(industry_map)
    # 剔除因子或市值缺失的样本（中性化前必须完整）
    df = df.dropna(subset=['factor', 'mv'])
    # 剔除行业缺失的样本
    df = df[df['industry'].notna()]
    if len(df) < 10:
        return pd.Series(index=factor_series.index, dtype=float)  # 样本太少返回空

    # ---------- 1) 3σ 去极值 ----------
    mean = df['factor'].mean()
    std = df['factor'].std()
    upper = mean + 3 * std
    lower = mean - 3 * std
    df['factor_wins'] = df['factor'].clip(lower, upper)

    # ---------- 2) Z-score 标准化 ----------
    df['factor_std'] = (df['factor_wins'] - df['factor_wins'].mean()) / df['factor_wins'].std()

    # ---------- 3) 双重中性化（回归取残差） ----------
    # 自变量：行业哑变量 + 对数市值
    # 行业哑变量
    industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
    # 对数市值
    df['log_mv'] = np.log(df['mv'])
    X = pd.concat([industry_dummies, df[['log_mv']]], axis=1)
    X = sm.add_constant(X)
    y = df['factor_std']
    # 回归
    model = sm.OLS(y, X, missing='drop').fit()
    resid = model.resid  # 残差即中性化后的因子值

    # ---------- 4) 再次标准化 ----------
    resid_std = (resid - resid.mean()) / resid.std()

    # 将结果映射回原始索引（未处理的股票填NaN）
    result = pd.Series(index=factor_series.index, dtype=float)
    result.update(resid_std)
    return result


# ========== 3. 逐月处理 ==========
processed_list = []
valid_dates = []
for col in pivot_bp.columns:
    try:
        # 尝试转换为 Timestamp，成功则认为是日期
        pd.to_datetime(col)
        valid_dates.append(col)
    except:
        print(f"跳过非日期列: {col}")

# 重新赋值 dates
dates = valid_dates

# 同时确保 pivot_mv 也只保留这些日期列
pivot_mv = pivot_mv[dates]
for date in dates:
    print(f"处理 {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date} ...")
    factor_series = pivot_bp[date]  # 该月份所有股票的因子值
    mv_series = pivot_mv[date]  # 对应月份的市值
    processed_series = process_monthly(factor_series, mv_series, industry_map)
    processed_series.name = date
    processed_list.append(processed_series)

# 合并为面板：行=股票，列=日期
processed_panel = pd.concat(processed_list, axis=1)
processed_panel.index = pivot_bp.index  # 确保股票顺序一致

# ========== 4. 保存结果 ==========
# 可选：将列名格式化为字符串
processed_panel.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else col for col in
                           processed_panel.columns]
processed_panel.to_csv('价值因子_BP_processed.csv')
print("处理完成，结果保存至 价值因子_BP_processed.csv")
print(processed_panel.head())
'''

#计算因子值 盈利质量
'''
from datetime import datetime
# ========== 1. 读取股票列表 ==========
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

# ========== 3. 获取盈利指标（净资产收益率 ROE） ==========
all_fin_data = []
for code in stock_list:
    print(f"正在获取 {code} 的财务指标...")
    try:
        df = pro.fina_indicator(ts_code=code, start_date='20230101', end_date='20260331',
                                fields='ts_code,end_date,ann_date,roe')
        if df is not None and not df.empty:
            df['ts_code'] = code
            df['end_date'] = pd.to_datetime(df['end_date'])
            df = df.sort_values('ann_date')
            all_fin_data.append(df)
    except Exception as e:
        print(f"获取 {code} 失败: {e}")
    time.sleep(0.8)

if not all_fin_data:
    raise ValueError("没有获取到任何财务指标数据")

fin_all = pd.concat(all_fin_data, ignore_index=True)

# ========== 4. 对齐到每月末（取每个调仓日之前最新报告期的 ROE） ==========
pivot_roe = pd.DataFrame(index=stock_list, columns=calendar)
for code in stock_list:
    code_data = fin_all[fin_all['ts_code'] == code].copy()
    if code_data.empty:
        continue
    for trade_date in calendar:
        eligible = code_data[code_data['end_date'] <= trade_date]
        if not eligible.empty:
            latest = eligible.loc[eligible['end_date'].idxmax()]
            pivot_roe.loc[code, trade_date] = latest['roe']

# ---------- 缺失值填充：前向填充（与规模因子相同） ----------
print("进行前向填充...")
pivot_roe_filled = pivot_roe.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
print(f"填充前缺失值总数: {pivot_roe.isna().sum().sum()}")
print(f"填充后缺失值总数: {pivot_roe_filled.isna().sum().sum()}")

# ========== 5. 准备行业和市值数据 ==========
stock_basic = pro.stock_basic(fields='ts_code,industry')
industry_map = dict(zip(stock_basic['ts_code'], stock_basic['industry']))

# 获取市值数据（每月最后一个交易日）
all_mv_data = []
for trade_date in calendar:
    date_str = trade_date.strftime('%Y%m%d')
    print(f"获取 {date_str} 市值数据...")
    try:
        df = pro.daily_basic(trade_date=date_str, fields='ts_code,total_mv')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(date_str)
            all_mv_data.append(df)
    except Exception as e:
        print(f"获取 {date_str} 市值数据失败: {e}")
    time.sleep(0.3)

mv_all = pd.concat(all_mv_data, ignore_index=True)
mv_monthly = mv_all[mv_all['trade_date'].isin(calendar)]
pivot_mv = mv_monthly.pivot(index='ts_code', columns='trade_date', values='total_mv')
pivot_mv = pivot_mv.reindex(stock_list)

# ---------- 市值面板同样进行前向填充 ----------
pivot_mv_filled = pivot_mv.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

# ========== 6. 预处理函数（与价值因子完全一致） ==========
def process_monthly(factor_series, mv_series, industry_map):
    df = pd.DataFrame({'factor': factor_series, 'mv': mv_series})
    df['industry'] = df.index.map(industry_map)
    df = df.dropna(subset=['factor', 'mv'])
    df = df[df['industry'].notna()]
    if len(df) < 10:
        return pd.Series(index=factor_series.index, dtype=float)

    # 1) 3σ 去极值
    mean = df['factor'].mean()
    std = df['factor'].std()
    upper = mean + 3 * std
    lower = mean - 3 * std
    df['factor_wins'] = df['factor'].clip(lower, upper)

    # 2) Z-score 标准化
    df['factor_std'] = (df['factor_wins'] - df['factor_wins'].mean()) / df['factor_wins'].std()

    # 3) 双重中性化（行业 + 对数市值）
    industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
    df['log_mv'] = np.log(df['mv'])
    X = pd.concat([industry_dummies, df[['log_mv']]], axis=1)
    X = sm.add_constant(X)
    y = df['factor_std']
    model = sm.OLS(y, X, missing='drop').fit()
    resid = model.resid

    # 4) 再次标准化
    resid_std = (resid - resid.mean()) / resid.std()

    result = pd.Series(index=factor_series.index, dtype=float)
    result.update(resid_std)
    return result

# ========== 7. 逐月处理 ==========
processed_list = []
dates = calendar  # 直接使用 calendar，所有列均为日期

for date in dates:
    print(f"处理 {date.strftime('%Y-%m-%d')} ...")
    factor_series = pivot_roe_filled[date]        # 使用填充后的因子数据
    mv_series = pivot_mv_filled[date]            # 使用填充后的市值数据
    processed_series = process_monthly(factor_series, mv_series, industry_map)
    processed_series.name = date
    processed_list.append(processed_series)

processed_panel = pd.concat(processed_list, axis=1)
processed_panel.index = pivot_roe_filled.index

# ========== 8. 保存结果 ==========
processed_panel.columns = [col.strftime('%Y-%m-%d') for col in processed_panel.columns]
processed_panel.to_csv('盈利因子_ROE_processed.csv')
print("处理完成，结果保存至 盈利因子_ROE_processed.csv")
print(processed_panel.head())
'''

#计算因子值 投资因子
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

# ========== 3. 获取资产负债表数据（总资产） ==========
all_bs_data = []
for code in stock_list:
    print(f"正在获取 {code} 的资产负债表数据...")
    try:
        df = pro.balancesheet(ts_code=code, start_date='20230101', end_date='20260331',
                              fields='ts_code,end_date,ann_date,total_assets')
        if df is not None and not df.empty:
            df['ts_code'] = code
            df['end_date'] = pd.to_datetime(df['end_date'])
            # 按公告日期排序，确保时效性
            df = df.sort_values('ann_date')
            all_bs_data.append(df)
    except Exception as e:
        print(f"获取 {code} 失败: {e}")
    time.sleep(3.1)

if not all_bs_data:
    raise ValueError("没有获取到任何资产负债表数据")

bs_all = pd.concat(all_bs_data, ignore_index=True)

# ========== 4. 计算总资产增长率 ==========
# 先为每只股票计算增长率（按报告期排序）
growth_list = []
for code in stock_list:
    code_data = bs_all[bs_all['ts_code'] == code].copy()
    if code_data.empty:
        continue
    code_data = code_data.sort_values('end_date')
    code_data['asset_growth'] = code_data['total_assets'].pct_change() * 100   # 百分比
    growth_list.append(code_data)

growth_all = pd.concat(growth_list, ignore_index=True)

# ========== 5. 对齐到每月末 ==========
pivot_growth = pd.DataFrame(index=stock_list, columns=calendar)
for code in stock_list:
    code_data = growth_all[growth_all['ts_code'] == code].copy()
    if code_data.empty:
        continue
    for trade_date in calendar:
        eligible = code_data[code_data['end_date'] <= trade_date]
        if not eligible.empty:
            latest = eligible.loc[eligible['end_date'].idxmax()]
            pivot_growth.loc[code, trade_date] = latest['asset_growth']

# 缺失值前向填充
print("进行前向填充...")
pivot_growth_filled = pivot_growth.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
print(f"填充前缺失值总数: {pivot_growth.isna().sum().sum()}")
print(f"填充后缺失值总数: {pivot_growth_filled.isna().sum().sum()}")

# ========== 6. 准备行业和市值数据 ==========
stock_basic = pro.stock_basic(fields='ts_code,industry')
industry_map = dict(zip(stock_basic['ts_code'], stock_basic['industry']))

# 获取市值数据（每月最后一个交易日）
all_mv_data = []
for trade_date in calendar:
    date_str = trade_date.strftime('%Y%m%d')
    print(f"获取 {date_str} 市值数据...")
    try:
        df = pro.daily_basic(trade_date=date_str, fields='ts_code,total_mv')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(date_str)
            all_mv_data.append(df)
    except Exception as e:
        print(f"获取 {date_str} 市值数据失败: {e}")
    time.sleep(0.3)

mv_all = pd.concat(all_mv_data, ignore_index=True)
mv_monthly = mv_all[mv_all['trade_date'].isin(calendar)]
pivot_mv = mv_monthly.pivot(index='ts_code', columns='trade_date', values='total_mv')
pivot_mv = pivot_mv.reindex(stock_list)

# 市值面板前向填充
pivot_mv_filled = pivot_mv.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

# ========== 7. 预处理函数（MAD去极值 + 标准化 + 双重中性化 + 再次标准化 + 取相反数） ==========
def process_monthly(factor_series, mv_series, industry_map):
    df = pd.DataFrame({'factor': factor_series, 'mv': mv_series})
    df['industry'] = df.index.map(industry_map)
    df = df.dropna(subset=['factor', 'mv'])
    df = df[df['industry'].notna()]
    if len(df) < 10:
        return pd.Series(index=factor_series.index, dtype=float)

    # 1) MAD 去极值（中位数 ± 5 * MAD）
    median = df['factor'].median()
    mad = (df['factor'] - median).abs().median()
    upper = median + 5 * mad
    lower = median - 5 * mad
    df['factor_wins'] = df['factor'].clip(lower, upper)

    # 2) Z-score 标准化
    df['factor_std'] = (df['factor_wins'] - df['factor_wins'].mean()) / df['factor_wins'].std()

    # 3) 双重中性化（行业 + 对数市值）
    industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
    df['log_mv'] = np.log(df['mv'])
    X = pd.concat([industry_dummies, df[['log_mv']]], axis=1)
    X = sm.add_constant(X)
    y = df['factor_std']
    model = sm.OLS(y, X, missing='drop').fit()
    resid = model.resid

    # 4) 再次标准化
    resid_std = (resid - resid.mean()) / resid.std()

    # 5) 取相反数（高增长率 → 投资激进 → 未来收益低 → 因子取负）
    investment_factor = -resid_std

    result = pd.Series(index=factor_series.index, dtype=float)
    result.update(investment_factor)
    return result

# ========== 8. 逐月处理 ==========
processed_list = []
dates = calendar

for date in dates:
    print(f"处理 {date.strftime('%Y-%m-%d')} ...")
    factor_series = pivot_growth_filled[date]
    mv_series = pivot_mv_filled[date]
    processed_series = process_monthly(factor_series, mv_series, industry_map)
    processed_series.name = date
    processed_list.append(processed_series)

processed_panel = pd.concat(processed_list, axis=1)
processed_panel.index = pivot_growth_filled.index

# ========== 9. 保存结果 ==========
processed_panel.columns = [col.strftime('%Y-%m-%d') for col in processed_panel.columns]
processed_panel.to_csv('投资因子_CMA_processed.csv')
print("处理完成，结果保存至 投资因子_CMA_processed.csv")
print(processed_panel.head())
'''

#计算IC IR序列
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== 1. 读取股票列表 ==========
stock_df = pd.read_csv('沪深300_202301_202603.csv', index_col=0)
stock_list = stock_df.iloc[:, 0].tolist()
print(f"共 {len(stock_list)} 只股票")

# ========== 2. 读取五个因子面板并统一列名为 Timestamp ==========
def load_factor(filepath, index_col='stock_code'):
    df = pd.read_csv(filepath, index_col=index_col)
    if 'stock_name' in df.columns:
        df = df.drop('stock_name', axis=1)
    df.columns = pd.to_datetime(df.columns)
    return df

mom_raw = load_factor('动量因子_2024_2026_with_names.csv')
size_raw = load_factor('规模因子_市值倒数_标准化_2024_2026.csv')
value_raw = load_factor('价值因子_BP_processed.csv')
roe_raw = load_factor('盈利因子_ROE_processed.csv')
cma_raw = load_factor('投资因子_CMA_processed.csv')

# 取所有因子日期的交集
common_dates = (mom_raw.columns.intersection(size_raw.columns)
                .intersection(value_raw.columns)
                .intersection(roe_raw.columns)
                .intersection(cma_raw.columns))
common_dates = common_dates.sort_values()
print(f"对齐后共 {len(common_dates)} 个月份")

mom = mom_raw[common_dates]
size = size_raw[common_dates]
value = value_raw[common_dates]
roe = roe_raw[common_dates]
cma = cma_raw[common_dates]

# 股票对齐（取五个因子都有数据的股票）
common_stocks = (mom.index.intersection(size.index)
                 .intersection(value.index)
                 .intersection(roe.index)
                 .intersection(cma.index))
mom = mom.loc[common_stocks]
size = size.loc[common_stocks]
value = value.loc[common_stocks]
roe = roe.loc[common_stocks]
cma = cma.loc[common_stocks]

print(f"对齐后股票数量: {len(common_stocks)}")

# ========== 3. 获取月线后复权收盘价并计算未来收益 ==========
start_date = common_dates.min().strftime('%Y%m%d')
end_date = common_dates.max().strftime('%Y%m%d')
print(f"获取月线数据: {start_date} ~ {end_date}")

all_monthly = []
for code in tqdm(stock_list, desc="获取月线"):
    try:
        df = ts.pro_bar(ts_code=code, adj='hfq', freq='M',
                        start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            df['ts_code'] = code
            all_monthly.append(df[['trade_date', 'ts_code', 'close']])
    except Exception as e:
        print(f"获取 {code} 失败: {e}")

monthly_all = pd.concat(all_monthly, ignore_index=True)
monthly_all['trade_date'] = pd.to_datetime(monthly_all['trade_date'])
price_wide = monthly_all.pivot(index='ts_code', columns='trade_date', values='close')
price_wide = price_wide.reindex(stock_list)

# 未来收益率 = (下月收盘 / 本月收盘 - 1)，列名保持本月日期
future_ret = price_wide.pct_change(axis=1).shift(-1, axis=1)
future_ret = future_ret.loc[common_stocks, common_dates]   # 对齐因子

# ========== 4. 手动计算每期 IC（Spearman 秩相关系数） ==========
def compute_ic_series(factor_df, ret_df):
    ic_list = []
    for date in factor_df.columns:
        f = factor_df[date]
        r = ret_df[date]
        mask = f.notna() & r.notna()
        if mask.sum() < 5:
            ic_list.append(np.nan)
            continue
        ic = spearmanr(f[mask], r[mask])[0]
        ic_list.append(ic)
    return pd.Series(ic_list, index=factor_df.columns, name='IC')

ic_mom = compute_ic_series(mom, future_ret)
ic_size = compute_ic_series(size, future_ret)
ic_value = compute_ic_series(value, future_ret)
ic_roe = compute_ic_series(roe, future_ret)
ic_cma = compute_ic_series(cma, future_ret)

# ========== 5. 计算 ICIR 和滚动 ICIR ==========
def compute_icir(ic_series, window=12):
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic != 0 else np.nan
    rolling_icir = ic_series.rolling(window).apply(lambda x: x.mean() / x.std() if x.std() != 0 else np.nan)
    return mean_ic, std_ic, icir, rolling_icir

factor_names = ['动量', '规模', '价值', '盈利', '投资']
ic_series_list = [ic_mom, ic_size, ic_value, ic_roe, ic_cma]

stats = {}
for name, ic in zip(factor_names, ic_series_list):
    mean_ic, std_ic, icir, roll_icir = compute_icir(ic)
    stats[name] = {'mean_IC': mean_ic, 'std_IC': std_ic, 'IR': icir, 'rolling_IR': roll_icir}
    print(f"{name}因子: 平均IC = {mean_ic:.4f}, IC标准差 = {std_ic:.4f}, IR = {icir:.4f}")
# ========== 6. 可视化（5行2列） ==========
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
plt.subplots_adjust(hspace=0.5, wspace=0.3)        # 增加子图间距
colors = ['blue', 'green', 'red', 'orange', 'purple']
for i, (name, ic, roll_ir) in enumerate(zip(factor_names, ic_series_list,
                                             [stats[k]['rolling_IR'] for k in factor_names])):
    # 左列：IC 时间序列
    axes[i, 0].plot(ic.index, ic.values,  alpha=0.7)
    axes[i, 0].axhline(0, color='red', linestyle='--')
    axes[i, 0].axhline(stats[name]['mean_IC'], color='blue', linestyle='-',
                       label=f"Mean IC = {stats[name]['mean_IC']:.4f}")
    axes[i, 0].set_title(f'{name}因子 IC 序列')
    axes[i, 0].legend()
    axes[i, 0].grid(True)

    # 右列：滚动 IR
    axes[i, 1].plot(roll_ir.index, roll_ir.values)
    axes[i, 1].axhline(0.5, color='red', linestyle='--', label='IR = 0.5')
    axes[i, 1].axhline(0, color='gray', linestyle='--')
    axes[i, 1].set_title(f'{name}因子 滚动 IR (12个月)')
    axes[i, 1].legend()
    axes[i, 1].grid(True)

plt.tight_layout()
plt.show()
# 保存 IC 序列供后续使用
ic_mom.to_csv('ic_momentum.csv')
ic_size.to_csv('ic_size.csv')
ic_value.to_csv('ic_value.csv')
ic_roe.to_csv('ic_roe.csv')
ic_cma.to_csv('ic_cma.csv')
'''

#实现等权/ICIR加权的因子组合 这个ICIR加权有前视误差，因为IR是整体信息算出来的
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取五个因子面板（已预处理，无缺失值） ==========
def load_factor(filepath, index_col='stock_code'):
    df = pd.read_csv(filepath, index_col=index_col)
    if 'stock_name' in df.columns:
        df = df.drop('stock_name', axis=1)
    df.columns = pd.to_datetime(df.columns)
    return df

# 请根据实际文件路径修改
mom = load_factor('动量因子_2024_2026_with_names.csv')
size = load_factor('规模因子_市值倒数_标准化_2024_2026.csv')
value = load_factor('价值因子_BP_processed.csv')
roe = load_factor('盈利因子_ROE_processed.csv')
cma = load_factor('投资因子_CMA_processed.csv')

# 对齐日期和股票（取五个因子的共同日期和共同股票）
common_dates = mom.columns.intersection(size.columns).intersection(value.columns).intersection(roe.columns).intersection(cma.columns)
common_dates = common_dates.sort_values()
common_stocks = mom.index.intersection(size.index).intersection(value.index).intersection(roe.index).intersection(cma.index)

mom = mom.loc[common_stocks, common_dates]
size = size.loc[common_stocks, common_dates]
value = value.loc[common_stocks, common_dates]
roe = roe.loc[common_stocks, common_dates]
cma = cma.loc[common_stocks, common_dates]

print(f"因子面板对齐：{len(common_stocks)} 只股票 × {len(common_dates)} 个月份")

# 因子字典
factors = {
    '动量': mom,
    '规模': size,
    '价值': value,
    '盈利': roe,
    '投资': cma
}

# ========== 2. 准备未来收益数据 ==========
# 方法：从月线后复权收盘价计算未来收益率（t月因子对应t+1月收益）
# 需要提前准备好 monthly_close_wide.csv（股票×日期，值为后复权收盘价）
pivot = pd.read_csv('monthly_close_wide.csv', index_col='stock_code')
pivot.columns = pd.to_datetime(pivot.columns)
# 对齐股票和日期
pivot = pivot.loc[common_stocks, common_dates]
# 计算未来收益率：下月收盘/本月收盘 - 1
future_ret = pivot.pct_change(axis=1).shift(-1, axis=1)
# 删除最后一期（无未来收益）
future_ret = future_ret.iloc[:, :-1]
common_dates = future_ret.columns  # 更新有效日期
# 重新对齐因子面板到有效日期
for name in factors:
    factors[name] = factors[name][common_dates]

print(f"未来收益对齐后：{len(common_dates)} 个月份")

# ========== 3. 计算各因子的全样本 IC 和 IR ==========
def compute_icir(factor_df, ret_df):
    ic_list = []
    for date in factor_df.columns:
        f = factor_df[date].dropna()
        r = ret_df[date].loc[f.index]
        mask = f.notna() & r.notna()
        if mask.sum() < 5:
            continue
        ic = spearmanr(f[mask], r[mask])[0]
        ic_list.append(ic)
    ic_series = pd.Series(ic_list)
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic != 0 else np.nan
    return mean_ic, std_ic, ir

icir_stats = {}
for name, f_df in factors.items():
    mean_ic, std_ic, ir = compute_icir(f_df, future_ret)
    icir_stats[name] = {'mean_IC': mean_ic, 'std_IC': std_ic, 'IR': ir}
    print(f"{name}因子: 平均IC = {mean_ic:.4f}, IC标准差 = {std_ic:.4f}, IR = {ir:.4f}")

# ========== 4. 构建多因子组合得分 ==========
# 等权组合：每个因子权重 1/5
weights_equal = {name: 1/len(factors) for name in factors}

# ICIR 加权组合：直接使用 IR 作为权重（可正可负）
# 为避免极端权重，可对绝对值归一化（可选）
ir_values = {name: icir_stats[name]['IR'] for name in factors}
total_abs_ir = sum(abs(v) for v in ir_values.values())
weights_icir = {name: ir / total_abs_ir for name, ir in ir_values.items()}

print("\n等权权重：", weights_equal)
print("ICIR权重（绝对值归一化）：", weights_icir)

# 计算综合得分
score_equal = sum(weights_equal[name] * factors[name] for name in factors)
score_icir = sum(weights_icir[name] * factors[name] for name in factors)

# 填充可能的缺失值（用0）
score_equal = score_equal.fillna(0)
score_icir = score_icir.fillna(0)

# ========== 5. 分组回测函数 ==========
def group_backtest(score_df, ret_df, n_groups=5):
    """
    对综合得分进行分层回测
    返回：
        group_ret_df: 分组收益矩阵（组别 × 月份）
        long_short: 多空组合收益序列
    """
    group_returns = []
    for month in score_df.columns:
        factor = score_df[month].dropna()
        ret = ret_df[month].loc[factor.index]
        # 等频分组
        try:
            groups = pd.qcut(factor.rank(method='first'), n_groups, labels=False) + 1
        except:
            groups = pd.cut(factor, n_groups, labels=False) + 1
        group_mean = ret.groupby(groups).mean()
        group_mean.name = month
        group_returns.append(group_mean)
    group_ret_df = pd.DataFrame(group_returns).T
    # 多空组合：组5（高） - 组1（低）
    long_short = group_ret_df.loc[n_groups] - group_ret_df.loc[1]
    return group_ret_df, long_short

# 等权组合回测
group_eq, ls_eq = group_backtest(score_equal, future_ret)
# ICIR加权组合回测
group_icir, ls_icir = group_backtest(score_icir, future_ret)

# ========== 6. 绩效指标计算 ==========
def calc_performance(returns_series):
    """计算年化收益、年化波动、夏普、最大回撤、胜率"""
    if returns_series.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    annual_return = (1 + returns_series).prod() ** (12 / len(returns_series)) - 1
    annual_vol = returns_series.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    cum = (1 + returns_series).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (returns_series > 0).mean()
    return annual_return, annual_vol, sharpe, max_drawdown, win_rate

metrics_eq = calc_performance(ls_eq.dropna())
metrics_icir = calc_performance(ls_icir.dropna())

print("\n等权组合：年化收益={:.2%}, 年化波动={:.2%}, 夏普={:.2f}, 最大回撤={:.2%}, 胜率={:.2%}".format(*metrics_eq))
print("ICIR加权：年化收益={:.2%}, 年化波动={:.2%}, 夏普={:.2f}, 最大回撤={:.2%}, 胜率={:.2%}".format(*metrics_icir))

# ========== 7. 可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 等权组合的分组净值曲线
group_cum_eq = (1 + group_eq).cumprod(axis=1)
group_cum_eq.T.plot(ax=axes[0, 0])
axes[0, 0].set_title('等权组合 - 分组累计净值')
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('累计净值')
axes[0, 0].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'])

# ICIR加权组合的分组净值曲线
group_cum_icir = (1 + group_icir).cumprod(axis=1)
group_cum_icir.T.plot(ax=axes[0, 1])
axes[0, 1].set_title('ICIR加权组合 - 分组累计净值')
axes[0, 1].set_xlabel('月份')
axes[0, 1].set_ylabel('累计净值')
axes[0, 1].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'])

# 多空组合累计净值对比
cum_eq = (1 + ls_eq).cumprod()
cum_icir = (1 + ls_icir).cumprod()
cum_eq.plot(ax=axes[1, 0], label='等权组合', linewidth=2)
cum_icir.plot(ax=axes[1, 0], label='ICIR加权', linewidth=2)
axes[1, 0].axhline(1, color='gray', linestyle='--')
axes[1, 0].set_title('多空组合累计净值对比')
axes[1, 0].set_xlabel('月份')
axes[1, 0].set_ylabel('累计净值')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 多空组合月度收益柱状图（可选）
# 右下角：改用折线图
ax = axes[1, 1]
ax.plot(ls_eq.index, ls_eq.values, marker='o', linestyle='-', linewidth=1.2, markersize=3, label='等权')
ax.plot(ls_icir.index, ls_icir.values, marker='s', linestyle='-', linewidth=1.2, markersize=3, label='ICIR加权')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title('多空组合月度收益率对比')
ax.set_xlabel('月份')
ax.set_ylabel('收益率')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)   # 为旋转标签留出空间
plt.show()
'''

#实现等权/IC加权的因子组合 IC用的是历史均值（最长12期），不用当期的IC（因为当期IC含未来信息），用的方法没有信息泄露
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取五个因子面板（已预处理） ==========
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

# 对齐日期和股票
common_dates = mom.columns.intersection(size.columns).intersection(value.columns).intersection(roe.columns).intersection(cma.columns)
common_dates = common_dates.sort_values()
common_stocks = mom.index.intersection(size.index).intersection(value.index).intersection(roe.index).intersection(cma.index)

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

# ========== 2. 读取已保存的 IC 序列 ==========
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

# ========== 3. 准备未来收益数据 ==========
pivot = pd.read_csv('monthly_close_wide.csv', index_col='stock_code')
pivot.columns = pd.to_datetime(pivot.columns)
pivot = pivot.loc[common_stocks, common_dates]
future_ret = pivot.pct_change(axis=1).shift(-1, axis=1)
future_ret = future_ret.iloc[:, :-1]          # 去掉最后一期无未来收益
common_dates_ret = future_ret.columns
for name in factors:
    factors[name] = factors[name][common_dates_ret]
ic_df = ic_df.loc[common_dates_ret]

# ========== 4. 定义两种权重 ==========
# 4.1 等权
weights_equal = pd.DataFrame(1/len(factors), index=common_dates_ret, columns=list(factors.keys()))

# 4.2 滚动历史 IC 均值（无未来信息）
# 将 IC 序列向后平移一期，使得 t 时刻可用的是 t-1 及之前的 IC
ic_shifted = ic_df.shift(1)
# 计算过去 12 个月的历史 IC 均值（不含当期，min_periods=1 表示至少有一个历史 IC 就计算）
rolling_ic_mean = ic_shifted.rolling(window=12, min_periods=1).mean()
# 归一化（保留符号，按绝对值归一化）
abs_sum = rolling_ic_mean.abs().sum(axis=1)
weights_hist = rolling_ic_mean.div(abs_sum, axis=0)
# 第一期 shift 后全部为 NaN，rolling 后仍为 NaN，用等权填充（但根据要求初期用历史均值，实际第一期无历史，只能用等权）
# 不过第一期本身就无历史，等权是合理的
weights_hist = weights_hist.fillna(1/len(factors))

# ========== 5. 合成组合得分 ==========
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

# ========== 6. 分组回测函数 ==========
def group_backtest(score_df, ret_df, n_groups=10):
    group_returns = []
    for month in score_df.columns:
        factor = score_df[month].dropna()
        ret = ret_df[month].loc[factor.index]
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

group_eq, ls_eq = group_backtest(score_equal, future_ret)
group_hist, ls_hist = group_backtest(score_hist, future_ret)

# ========== 7. 绩效指标计算 ==========
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

metrics_eq = calc_performance(ls_eq.dropna())
metrics_hist = calc_performance(ls_hist.dropna())

print("等权组合（无未来信息）：")
print(f"  年化收益={metrics_eq[0]:.2%}, 年化波动={metrics_eq[1]:.2%}, 夏普={metrics_eq[2]:.2f}, 最大回撤={metrics_eq[3]:.2%}, 胜率={metrics_eq[4]:.2%}")
print("\n滚动历史IC均值（无未来信息）：")
print(f"  年化收益={metrics_hist[0]:.2%}, 年化波动={metrics_hist[1]:.2%}, 夏普={metrics_hist[2]:.2f}, 最大回撤={metrics_hist[3]:.2%}, 胜率={metrics_hist[4]:.2%}")

# ========== 8. 可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 等权组合的分组净值曲线
group_cum_eq = (1 + group_eq).cumprod(axis=1)
group_cum_eq.T.plot(ax=axes[0,0])
axes[0,0].set_title('等权组合 - 分组累计净值')
axes[0,0].set_xlabel('月份')
axes[0,0].set_ylabel('累计净值')
axes[0,0].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'])

# 滚动历史IC均值组合的分组净值曲线
group_cum_hist = (1 + group_hist).cumprod(axis=1)
group_cum_hist.T.plot(ax=axes[0,1])
axes[0,1].set_title('滚动历史IC均值 - 分组累计净值')
axes[0,1].set_xlabel('月份')
axes[0,1].set_ylabel('累计净值')
axes[0,1].legend(title='分组', labels=['组1(低)', '组2', '组3', '组4', '组5(高)'])

# 多空组合累计净值对比
cum_eq = (1 + ls_eq).cumprod()
cum_hist = (1 + ls_hist).cumprod()
ax = axes[1,0]
cum_eq.plot(ax=ax, label='等权')
cum_hist.plot(ax=ax, label='滚动历史IC均值')
ax.axhline(1, color='gray', linestyle='--')
ax.set_title('多空组合累计净值对比')
ax.set_xlabel('月份')
ax.set_ylabel('累计净值')
ax.legend()
ax.grid(True)

# 多空组合月度收益率折线图
ax = axes[1,1]
ax.plot(ls_eq.index, ls_eq.values, marker='o', linestyle='-', linewidth=1, markersize=3, label='等权')
ax.plot(ls_hist.index, ls_hist.values, marker='s', linestyle='-', linewidth=1, markersize=3, label='滚动历史IC均值')
ax.axhline(0, color='black', linestyle='--')
ax.set_title('多空组合月度收益率对比')
ax.set_xlabel('月份')
ax.set_ylabel('收益率')
ax.legend()
ax.grid(True)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()

# ========== 9. 保存合成因子值到 CSV ==========
# 等权组合
score_equal.to_csv('含另类_5组_combined_factor_equal_weight.csv', index=True)
# 滚动历史 IC 加权组合
score_hist.to_csv('含另类_5组_combined_factor_ic_weighted.csv', index=True)

stock_info = pro.stock_basic(fields='ts_code,name')
name_map = dict(zip(stock_info['ts_code'], stock_info['name']))

# 等权组合带名称
df_eq = score_equal.reset_index()
df_eq.rename(columns={'index': 'stock_code'}, inplace=True)
df_eq.insert(0, 'stock_name', df_eq['stock_code'].map(name_map))
df_eq.to_csv('含另类_5组_combined_factor_equal_weight_with_name.csv', index=False)

# 滚动历史 IC 加权组合带名称
df_hist = score_hist.reset_index()
df_hist.rename(columns={'index': 'stock_code'}, inplace=True)
df_hist.insert(0, 'stock_name', df_hist['stock_code'].map(name_map))
df_hist.to_csv('含另类_5组_combined_factor_ic_weighted_with_name.csv', index=False)
