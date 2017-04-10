# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:33:07 2017

@author: lh
"""

import numpy as np
import pandas as pd
from WindPy import w
import statsmodels.api as sm
w.start()


def winsorize_series(se):
    q = se.quantile([0.025, 0.975])
    if isinstance(q, pd.Series) and len(q) == 2:
        se[se < q.iloc[0]] = q.iloc[0]
        se[se > q.iloc[1]] = q.iloc[1]
    return se


def winsorize(factor):
    return factor.apply(winsorize_series, axis=1)


def standarlize(factor):
    factor = factor.dropna(how='all')
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std


def ic_caculate(factor, pctChange, period):
    ic = []
    index = factor.index
    for i in range(factor.shape[0] - period):
        ic_value = factor.ix[index[i]].corr(pctChange.ix[index[i + period]])
        ic.append(ic_value)
    return pd.Series(ic, index=index[:len(index) - period])


def rankic(factor, pctchange, period):
    factor_rank = factor.rank(axis=1)
    pctchange_rank = pctchange.rank(axis=1)
    return ic_caculate(factor_rank, pctchange_rank, period)


def factor_handle(factor):
    factor = winsorize(factor)
    factor = standarlize(factor)
    return factor


def group_backtest(factor, volume, close, group_num, quantile, fee):
    pct_chg = close.pct_change()
    stockpool = pd.Series(np.zeros(factor.shape[1]), index=factor.columns)
    cash = 1.0
    net_value = []
    for i in range(1, factor.shape[0]):
        date = factor.index[i]
        factor_today = factor.ix[factor.index[i - 1]].sort_values().dropna()
        close_today = close.ix[date]
        pct_chg_today = pct_chg.ix[date]
        vol_today = volume.ix[date]
        inteval_len = factor_today.shape[0] / group_num
        tobuy = factor_today[
                (quantile - 1) * inteval_len:quantile * inteval_len].index
        tosell = stockpool[stockpool > 0].index
        first_sell = list(set(tosell) - set(tobuy))
        for stock in first_sell:
            if pct_chg_today[stock] > -0.099 and vol_today[stock] > 0:
                cash += close_today[stock] * stockpool[stock] * (1 - fee)
                stockpool[stock] = 0.0
        last_buy = list(set(tobuy) - set(tosell))
        buy_num = len(last_buy)
        per_money = cash / (buy_num + 0.0)
        for stock in last_buy:
            if pct_chg_today[stock] < 0.99 and vol_today[stock] > 0:
                stockpool[stock] += per_money / close_today[stock] * (1 - fee)
                cash -= per_money

        pool = stockpool[stockpool > 0]
        net_value.append((pool * close_today[pool.index]).sum() + cash)
    return pd.Series(net_value, index=factor.index[1:])


def group_result(pctchange, dic, period):
    net_value = dict()
    for s in dic.keys():
        group = dic[s]
        zz = pd.Series([pctchange.iloc[i][dic[s][i - period]].mean() for i in range(period, len(pctchange))]) + 1
        net_value[s] = zz.cumprod()
    res = pd.DataFrame(net_value)
    res.index = pctchange.index[period:]
    return res


def generate_group(factor, group_num):
    dic = dict()
    for i in range(1, group_num + 1):
        dic[i] = []
    for line in range(factor.shape[0]):
        temp = factor.iloc[line].copy()
        temp = temp.dropna().sort_values()
        interval = len(temp) / group_num
        for quantile in range(1, group_num + 1):
            dic[quantile].append(temp[(quantile - 1) * interval:quantile * interval].index)
    return dic


def mean_return(factor, pctchange, group_num, period):
    dic = generate_group(factor, group_num)
    group_return = dict()
    pctchange = pctchange.ix[factor.index]
    for key in dic.keys():
        group_return[key] = pd.Series([pctchange.iloc[i][dic[key][i - period]].mean()
                                       for i in range(period, len(pctchange))]).mean()
    return pd.Series(group_return)


def quick_test(factor, pctChange, group_num, period):
    dic = generate_group(factor, group_num)
    pctChange = pctChange.ix[factor.index]
    res = group_result(pctChange, dic, period)
    return res


def hedge_curve(net_value, benchmark_code):
    benchmark = w.wsd(benchmark_code, "pct_chg", net_value.index[0], net_value.index[-1], "PriceAdj=B")
    benchmark = pd.Series(benchmark.Data[0], index=net_value.index) / 100.0
    net_value_pct = net_value.pct_change()
    hedge_res = (net_value_pct.apply(lambda x: x - benchmark) + 1).cumprod()
    return hedge_res


def returns_sta(net_value):
    grouped = net_value.groupby(lambda x: x.split('-')[0])
    return grouped.apply(lambda x: x.iloc[-1]/x.iloc[0]-1)


def cap_neutral(factor, mkt_value):
    new_factor = factor.copy().dropna(how='all')
    for i in range(new_factor.shape[0]):
        a = new_factor.iloc[i].dropna()
        mkt = mkt_value.iloc[i].ix[a.index]
        resduies = sm.OLS(a, mkt).fit().resid
        new_factor.iloc[i].ix[a.index] = resduies
    return new_factor



