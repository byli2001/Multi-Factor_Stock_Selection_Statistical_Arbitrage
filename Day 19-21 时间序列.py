import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import akshare as ak
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import warnings
from tqdm import tqdm
ts.set_token(" ")
warnings.filterwarnings('ignore')
pro = ts.pro_api()
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#ARIMA分析股价 定阶发现 海尔智家股价收益率平稳 几乎服从随机游走
'''
# Step 1: 获取股票日线数据
print("\n[Step 1] 正在获取日线数据...")
df = pro.daily(ts_code='600000.SH', start_date='2023-01-01', end_date='2025-12-31')

# 按日期升序排列
df = df.sort_values('trade_date')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)

print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
print(f"总交易日数: {len(df)}")
print(df.head())

# ============================================================
# Step 2: 计算对数收益率并可视化
# ============================================================

print("\n[Step 2] 计算对数收益率...")
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
returns = df['log_return'].dropna()  # 删除NaN

print(f"收益率序列长度: {len(returns)}")

# 绘制收盘价和收益率
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(df.index, df['close'], color='blue')
axes[0].set_title('海尔智家收盘价', fontsize=12)
axes[0].set_ylabel('价格（元）')
axes[0].grid(True, alpha=0.3)

axes[1].plot(returns.index, returns, color='green', alpha=0.7)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('对数收益率序列', fontsize=12)
axes[1].set_ylabel('收益率')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Step 3: 平稳性检验（ADF检验）
# ============================================================

print("\n[Step 3] ADF平稳性检验...")
result = adfuller(returns, autolag='AIC')
print(f"ADF统计量: {result[0]:.6f}")
print(f"p值: {result[1]:.10f}")
print(f"临界值:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("\n结论: p值 < 0.05，拒绝原假设，序列平稳，无需差分（d=0）")
else:
    print("\n结论: 序列非平稳，需要差分处理")

# 可选：若p>0.05，尝试一阶差分
if result[1] >= 0.05:
    print("\n尝试一阶差分...")
    diff_returns = returns.diff().dropna()
    diff_result = adfuller(diff_returns)
    print(f"一阶差分后p值: {diff_result[1]:.10f}")
# Step 4: ACF/PACF图定阶（确定p和q）
# ============================================================

print("\n[Step 4] 绘制ACF和PACF图，确定p和q...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(returns, ax=axes[0], lags=20, title='自相关函数（ACF）')
plot_pacf(returns, ax=axes[1], lags=20, title='偏自相关函数（PACF）')
plt.tight_layout()
plt.show()

print("\n定阶指引:")
print("- ACF拖尾（缓慢衰减） + PACF截尾（某一阶后为0）→ AR模型，p = PACF截尾位置")
print("- ACF截尾 + PACF拖尾 → MA模型，q = ACF截尾位置")
print("- 两者均拖尾 → ARMA模型，用AIC/BIC选择最优参数")
print("\n观察上图，初步判断p=1, q=0或1，后续用AIC选择最优模型。")

# ============================================================
# Step 5: 用AIC/BIC自动选择最优ARIMA阶数
print("\n[Step 5] 使用AIC/BIC选择最优ARIMA模型...")
def find_best_arima(series, p_range=range(0, 4), d=0, q_range=range(0, 4)):
    """遍历p,d,q组合，返回AIC最小的模型"""
    best_aic = np.inf
    best_order = None
    best_model = None
    results_list = []

    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                aic = fitted.aic
                results_list.append({'p': p, 'q': q, 'AIC': aic})
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = fitted
            except:
                continue

    print(f"最优阶数: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")
    print(f"最优AIC: {best_aic:.4f}")
    return best_model, best_order, results_list


# 收益率已平稳，d=0
best_model, best_order, aic_results = find_best_arima(returns, p_range=range(0, 4), d=0, q_range=range(0, 4))

# 显示AIC对比表
print("\n各模型AIC对比（前10个）:")
aic_df = pd.DataFrame(aic_results).sort_values('AIC').head(10)
print(aic_df)

# ============================================================
# Step 6: 拟合最优ARIMA模型并输出摘要

print(f"\n[Step 6] 拟合ARIMA{best_order}模型...")
model_fit = best_model
print(model_fit.summary())

# ============================================================
# Step 7: 模型诊断 - 残差分析
# ============================================================

print("\n[Step 7] 模型诊断：残差分析...")

# 获取残差
residuals = model_fit.resid

# 1. 绘制标准诊断图（标准化残差、直方图、QQ图、ACF）
fig = model_fit.plot_diagnostics(figsize=(12, 10))
plt.suptitle(f'ARIMA{best_order}模型诊断图', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 2. Ljung-Box检验（残差白噪声检验）
print("\nLjung-Box白噪声检验:")
lb_test = acorr_ljungbox(residuals, lags=[5, 10, 15], return_df=True)
print(lb_test)

# 解读检验结果
print("\n残差白噪声检验解读:")
for lag in [5, 10, 15]:
    p_val = lb_test.loc[lag, 'lb_pvalue']
    if p_val > 0.05:
        print(f"  滞后{lag}阶: p值={p_val:.4f} > 0.05，无法拒绝原假设 → 残差为白噪声")
    else:
        print(f"  滞后{lag}阶: p值={p_val:.4f} < 0.05，拒绝原假设 → 残差仍存在自相关")

# 3. 计算残差平方的Ljung-Box检验（判断是否需要GARCH）
print("\n残差平方的Ljung-Box检验（ARCH效应检验）:")
residuals_sq = residuals ** 2
lb_sq_test = acorr_ljungbox(residuals_sq, lags=[5, 10, 15], return_df=True)
print(lb_sq_test)

print("\nARCH效应判断:")
need_garch = False
for lag in [5, 10, 15]:
    p_val = lb_sq_test.loc[lag, 'lb_pvalue']
    if p_val < 0.05:
        print(f"  滞后{lag}阶: p值={p_val:.4f} < 0.05，存在ARCH效应，建议使用GARCH建模")
        need_garch = True
if not need_garch:
    print("  所有滞后阶p值均>0.05，不存在ARCH效应，仅ARIMA即可充分建模")

# ============================================================
# Step 8: 收益率预测（未来20个交易日）
print("\n[Step 8] 预测未来20个交易日的对数收益率...")

# 预测未来20步
forecast_steps = 20
forecast = model_fit.forecast(steps=forecast_steps)

print(f"未来{forecast_steps}个交易日对数收益率预测值:")
for i, val in enumerate(forecast, 1):
    print(f"  T+{i:2d}: {val:.6f}")

# 将预测的对数收益率转换为价格预测（可选）
# 注：价格预测仅供参考，误差会累积
last_price = df['close'].iloc[-1]
forecast_price = [last_price]
for ret in forecast:
    forecast_price.append(forecast_price[-1] * np.exp(ret))
forecast_price = forecast_price[1:]  # 移除初始值

# 生成未来日期索引（交易日，简化处理）
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# 绘制预测结果
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 收益率预测图
axes[0].plot(returns.index[-100:], returns.iloc[-100:], label='历史收益率', color='blue', alpha=0.7)
axes[0].plot(future_dates, forecast, label='预测收益率', color='red', linestyle='--', marker='o', markersize=4)
axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].set_title('ARIMA收益率预测', fontsize=12)
axes[0].set_ylabel('对数收益率')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 价格预测图
axes[1].plot(df.index[-100:], df['close'].iloc[-100:], label='历史收盘价', color='blue', alpha=0.7)
axes[1].plot(future_dates, forecast_price, label='预测收盘价', color='red', linestyle='--', marker='o', markersize=4)
axes[1].set_title('基于ARIMA预测的价格走势（仅供参考，误差会累积）', fontsize=12)
axes[1].set_ylabel('价格（元）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# 总结输出
# ============================================================
print("\n" + "=" * 60)
print("=" * 60)
print(f"✓ 数据获取: 海尔智家 {len(df)} 个交易日")
print(f"✓ 平稳性检验: p值={result[1]:.10f} → 序列{'平稳' if result[1] < 0.05 else '非平稳'}")
print(f"✓ ARIMA定阶: AIC自动选择 → ARIMA{best_order}")
print(f"✓ 模型拟合完成")
print(f"✓ 残差诊断: Ljung-Box检验完成")
if need_garch:
    print("✓ ARCH效应: 存在，建议继续学习GARCH模型")
else:
    print("✓ ARCH效应: 不存在，ARIMA已充分拟合")
print(f"✓ 收益率预测: 未来{forecast_steps}个交易日")
'''

#协整与配对交易回测 中信建投 中金公司

stock_A = '601066.SH'  # 中信建投
stock_B = '601995.SH'  # 中金公司

start_date = '20240101'
end_date = '20260420'

# 回测参数
beta_window = 30  # 估计对冲比率的滚动窗口长度（必须足够长以保证回归稳定）
zscore_window = 30  # 计算Z-Score的滚动窗口长度
entry_z = 2.0
exit_z = 0.5


# ==================== 2. 获取数据（前复权） ====================
def get_price(ts_code, start, end):
    df = ts.pro_bar(ts_code=ts_code, start_date=start, end_date=end, adj='qfq', asset='E')
    df = df[['trade_date', 'close']].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    return df['close']


price_a = get_price(stock_A, start_date, end_date)
price_b = get_price(stock_B, start_date, end_date)
prices = pd.DataFrame({'A': price_a, 'B': price_b}).dropna()
print(f"数据范围：{prices.index[0].date()} 至 {prices.index[-1].date()}，共{len(prices)}个交易日")

# ==================== 3. 协整检验（全样本，仅供参考） ====================
score, pvalue, _ = coint(prices['A'], prices['B'])
print(f"\n全样本协整检验 p-value: {pvalue:.6f}")
if pvalue < 0.05:
    print("结论：存在协整关系，适合配对交易。")
else:
    print("结论：不存在协整关系，请更换股票对。")
    # 为演示继续，实际应退出

# ==================== 4. 滚动估计对冲比率 β ====================
# 初始化一个空列存放每日的 β（从beta_window天之后开始有值）
prices['beta'] = np.nan

# 滚动回归：从第beta_window个数据点开始，每次取前beta_window天回归
for i in range(beta_window, len(prices)):
    window_data = prices.iloc[i - beta_window:i]  # 历史窗口（不含当前日）
    # 回归 A = α + β * B
    reg = LinearRegression()
    reg.fit(window_data['B'].values.reshape(-1, 1), window_data['A'].values.reshape(-1, 1))
    prices.iloc[i, prices.columns.get_loc('beta')] = reg.coef_[0][0]

# 前beta_window天无法计算β，删除这些缺失行（或从第beta_window天开始）
prices = prices.iloc[beta_window:].copy()

# ==================== 5. 计算价差与滚动Z-Score ====================
# 价差 = A - β * B（使用当日β）
prices['spread'] = prices['A'] - prices['beta'] * prices['B']

# 滚动计算价差的均值和标准差（用于Z-Score）
prices['spread_mean'] = prices['spread'].rolling(window=zscore_window).mean()
prices['spread_std'] = prices['spread'].rolling(window=zscore_window).std()
prices['zscore'] = (prices['spread'] - prices['spread_mean']) / prices['spread_std']

# ==================== 6. 生成交易信号 ====================
prices['position'] = 0
# 开仓条件
prices.loc[prices['zscore'] > entry_z, 'position'] = -1  # 做空价差
prices.loc[prices['zscore'] < -entry_z, 'position'] = 1  # 做多价差
# 平仓条件
prices.loc[abs(prices['zscore']) < exit_z, 'position'] = 0
# 仓位向前填充（直到平仓信号出现）
prices['position'] = prices['position'].replace(0, np.nan).ffill().fillna(0)

# ==================== 7. 计算策略收益 ====================
prices['ret_A'] = prices['A'].pct_change()
prices['ret_B'] = prices['B'].pct_change()
# 策略收益率 = 前一日仓位 * (A收益 - β * B收益)  注意：β需使用前一日值
prices['strategy_ret'] = prices['position'].shift(1) * (prices['ret_A'] - prices['beta'].shift(1) * prices['ret_B'])
# 累计净值
prices['strategy_nav'] = (1 + prices['strategy_ret']).cumprod()
prices['benchmark_nav'] = (1 + prices['ret_A']).cumprod()

# ==================== 8. 绩效评估 ====================
rets = prices['strategy_ret'].dropna()
if len(rets) == 0:
    print("无交易信号，策略收益为空。")
else:
    total_return = prices['strategy_nav'].iloc[-1] - 1
    days = (prices.index[-1] - prices.index[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252)
    cummax = prices['strategy_nav'].expanding().max()
    drawdown = (prices['strategy_nav'] - cummax) / cummax
    max_drawdown = drawdown.min()

    print(f"\n===== 滚动β回测绩效统计 =====")
    print(f"累计收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_drawdown:.2%}")

# ==================== 9. 绘图 ====================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(prices.index, prices['A'], label='中信建投')
axes[0].plot(prices.index, prices['B'], label='中金公司')
axes[0].set_ylabel('价格（前复权）')
axes[0].legend()
axes[0].grid(True)
axes[1].plot(prices.index, prices['zscore'], label='Z-Score', color='purple')
axes[1].axhline(entry_z, linestyle='--', color='red', label=f'开仓阈值 ±{entry_z}')
axes[1].axhline(-entry_z, linestyle='--', color='red')
axes[1].axhline(exit_z, linestyle=':', color='gray', label=f'平仓阈值 ±{exit_z}')
axes[1].axhline(-exit_z, linestyle=':', color='gray')
axes[1].fill_between(prices.index, 0, prices['position'], where=prices['position'] != 0,
                     alpha=0.3, color='green', label='持仓')
axes[1].set_ylabel('Z-Score')
axes[1].legend()
axes[1].grid(True)
axes[2].plot(prices.index, prices['strategy_nav'], label='配对策略净值（滚动β）', color='green')
axes[2].plot(prices.index, prices['benchmark_nav'], label='中信建投净值（基准）', color='blue', alpha=0.6)
axes[2].set_ylabel('累计净值')
axes[2].set_xlabel('日期')
axes[2].legend()
axes[2].grid(True)
plt.tight_layout()
plt.show()
