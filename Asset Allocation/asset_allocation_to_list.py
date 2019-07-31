for _ in range(1):
    # The usual stuff
    import os
    import sys
    import numpy as np
    import pandas as pd
    import datetime
    from dateutil.relativedelta import relativedelta
    import time
    import math
    import shutil
    import collections
    import copy
    import zipfile
    import cvxpy as cp
    # from workalendar.asia import SouthKorea
    # import itertools
    # import multiprocessing as mp
    from scipy.optimize import minimize

    # our main tool for optimization

    # from analytical_sharpe_max import analytical_sharpe_weight

    # plotting
    # import plotly
    # import plotly.offline as offline  # offline.plot(fig, image='png')
    # from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    # import plotly.plotly as py
    # import plotly.graph_objs as go
    # import cufflinks  # pd.DataFrame.iplot()

    # debugging tool
    import logging

    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s  -  %(message)s')
    logging.disable(logging.DEBUG)

    # current location
    # os.chdir(r'/home/swy/quantec_git/asset allocation/')
    # os.chdir('/home/incantator/quantec_git/asset allocation/')
    # os.chdir(r'C:\Users\Incantator\Documents\quantec_git\asset allocation')
    # os.chdir(r'C:\quantec_git\asset allocation')
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # DownloadsDir = r'C:\Users\quantec\Downloads'
    # os.chdir(r'C:\Users\swy_2\Documents\quantec_git\asset allocation')

    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        os.chdir(sys._MEIPASS)
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # I wanna see each and every column rather than giving me the abbreviations '...'
    pd.set_option('expand_frame_repr', False)

    # pandas display floating precision to 10; default is 6
    # https://pandas.pydata.org/pandas-docs/stable/options.html
    pd.reset_option('display.precision', 10)

    # SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  # default='warn'

####################################################################################################################
# Retrieving Data from the Database
####################################################################################################################
# Receiving arguments
# The validity should be checked before the query, including the datetime and the bounds.
# sys.argv[0] is always the name of the program
# sys.argv = ['NAME', '9', '20160129', '20190514', 'Sharpe', 'D', '5', 'SPY', '삼성전자', 'IPAY', '신한지주', 'UUP', '0.05', '1.', '0.05', '1.', '0.05', '1.', '0.05', '1.', '0.05', '1.', 'SPY', 1000000, 100000, 3, 1]
# sys.argv = ['NAME', '12', '20120330', '20151010', 'Sharpe', 0, '1', 'SPY', '0', '1', 'SPY', 1000000, 0]
# sys.argv = ['NAME', '9', '20150129', '20190424', 'Sharpe', 'D', '1', 'SPY',  '0.05', '1', 'SPY', '100000', '0']
# python asset_allocation_to_list.py 12 20100101 20151010 Sharpe W 3 ACWV AOR BSCK 0 1 .1 .5 .1 .6 SPY 1000000 3
if len(sys.argv) < 13:
    print('You need at least 13 arguments (lookback date1 date2 method interval num strategy min_ratio max_ratio bm acc_fund fee)')
    sys.exit()

lookback = int(sys.argv[1])
date_start = sys.argv[2]
date_end = sys.argv[3]
method = sys.argv[4]  # Sharpe, Max
interval = sys.argv[5]
num = int(sys.argv[6])  # 종목수
data_arg = sys.argv[7:7 + num]  # 종목명
bounds = sys.argv[7 + num:7 + 3 * num]
bounds = np.array(list(map(float, bounds))).reshape(int(len(bounds) / 2), 2)
bm_arg = [sys.argv[-5]]
curr = int(sys.argv[-1])


try:
    acc_fund = float(sys.argv[-4])  # 거치금액
except:
    acc_fund = 'yo'

try:
    acc = float(sys.argv[-3])  # 월적립액
except:
    acc_fund = 'yeah'

fee = float(sys.argv[-2])  # always should be a valid number, just type in 0 if necessary

#if interval == 'W':
 #   fee = fee * 0.01 / 52
#else:
fee = fee * 0.01 / 252

zf_us = zipfile.ZipFile('daily_price_ForKorETF_kor.zip')
dp_us = pd.read_csv(zf_us.open("daily_price_ForKorETF_kor.csv"), index_col='date')
us_stock_list = dp_us.columns

zf_kr = zipfile.ZipFile('daily_price_KOSPI_KOSDAQ_kor.zip')
dp_kr = pd.read_csv(zf_kr.open('daily_price_KOSPI_KOSDAQ_kor.csv'), index_col='date')
kr_stock_list = dp_kr.columns

us_stock, kr_stock = [], []
for i in range(num):
    if data_arg[i] in us_stock_list:
        us_stock.append(data_arg[i])
    elif data_arg[i] in kr_stock_list:
        kr_stock.append(data_arg[i])

#data_arg = us_stock + kr_stock
if bm_arg[0] in us_stock_list:
    bm_dp = dp_us[bm_arg].dropna()
    bm_dp.index = pd.to_datetime(bm_dp.index)
if bm_arg[0] in kr_stock_list:
    bm_dp = dp_kr[bm_arg].dropna()
    bm_dp.index = pd.to_datetime(bm_dp.index)



dp_us = dp_us[us_stock].dropna()
dp_kr = dp_kr[kr_stock].dropna()

dp_us.index = pd.to_datetime(dp_us.index)
dp_kr.index = pd.to_datetime(dp_kr.index)



zf_cur = zipfile.ZipFile('Cur.zip')
cur = pd.read_csv(zf_cur.open("Cur.csv"), header=None)  # csv fileì˜ ì²«ë²ˆì§¸ rowë¥¼ column nameìœ¼ë¡œ ì½ì–´ì˜¤ì§€ ì•Šê²Œ header= None
cur.columns = ['date', 'cur']
cur.index = cur['date']
del cur['date']
cur.index = pd.to_datetime(cur.index)

# Datetime input checking
date_func = lambda x: datetime.datetime(year=int(x[0:4]), month=int(x[4:6]), day=int(x[6:8]))

if date_start == 0:
    date_start = max(dp_us.index[0], dp_kr.index[0], cur.index[0])  # Timestamp('2015-07-16 00:00:00')
else:
    date_start = date_func(date_start)  # datetime.datetime(2016, 1, 29, 0, 0)
if date_end == 0:
    date_end = min(dp_us.index[-1], dp_kr.index[-1], cur.index[-1])
else:
    date_end = date_func(date_end)

dp_us = dp_us[(date_start <= dp_us.index) & (dp_us.index <= date_end)]  # It will be cut further by lookback later
dp_kr = dp_kr[(date_start <= dp_kr.index) & (dp_kr.index <= date_end)]


dp = dp_us.merge(dp_kr, how='outer', on='date', sort = True).fillna(method='ffill').dropna()

bm_dp = bm_dp.reindex(dp.index).fillna(method='ffill').dropna()

cur = cur.reindex(dp.index).fillna(method = 'ffill').dropna()

dp = dp[(max(dp.index[0], bm_dp.index[0], cur.index[0]) <= dp.index ) & (dp.index <= min(dp.index[-1], bm_dp.index[-1], cur.index[-1])) ]
bm_dp = bm_dp[(max(dp.index[0], bm_dp.index[0], cur.index[0]) <= bm_dp.index ) & (bm_dp.index <= min(dp.index[-1], bm_dp.index[-1], cur.index[-1])) ]
cur = cur[(max(dp.index[0], bm_dp.index[0], cur.index[0]) <= cur.index ) & (cur.index <= min(dp.index[-1], bm_dp.index[-1], cur.index[-1])) ]



for col in dp.columns:
    dp[col] = dp[col].apply(lambda x: str(x).replace(',',''))
    dp[col] = dp[col].apply(lambda x: float(x))

for col in bm_dp.columns:
    bm_dp[col] = bm_dp[col].apply(lambda x: str(x).replace(',', ''))
    bm_dp[col] = bm_dp[col].apply(lambda x: float(x))

for col in cur.columns:
    cur[col] = cur[col].apply(lambda x: str(x).replace(',', ''))
    cur[col] = cur[col].apply(lambda x: float(x))

'''
def date_push_back(dp):
    """

    :param dp:
    :return: monthly_index is the index of points including the last datetime.
    mr_mr only considers up to the date where rebalancing happens.
    The criterion is
    """
    index = dp.index
    base_dt = index[0]
    last_dt = index[-1]
    length = len(index.to_period('M').drop_duplicates()) - 1
    dt = base_dt
    monthly_index = [base_dt]
    for _ in range(length):
        dt = dt + relativedelta(months=1)  # We assume that a month has at least a single day in dp.
        month = dt.month
        if dt in index:  # The exact day of the next month.
            monthly_index.append(dt)
        else:  # Start looking for a later day.
            # Detected an error! after February the day becomes 28, so not good!
            dt_var = dt + relativedelta(days=1)
            while dt_var.month == month:
                if dt_var in index:
                    monthly_index.append(dt_var)
                    break
                else:
                    dt_var = dt_var + relativedelta(days=1)
            else:  # Start looking for an earlier day.
                dt_var = dt - relativedelta(days=1)
                while True:
                    if dt_var in index:
                        monthly_index.append(dt_var)
                        break
                    else:
                        dt_var = dt_var - relativedelta(days=1)

    logging.debug(f'{dt} {monthly_index[-1]} {last_dt}')
    if monthly_index[-1] < last_dt:
        return monthly_index + [last_dt], dp.reindex(monthly_index).pct_change().dropna()
    else:  # monthly_index[-1] == last_dt
        return monthly_index, dp.reindex(monthly_index[:-1]).pct_change().dropna()
'''


def date_push_back(dp):
    """

    :param dp:
    :return: monthly_index is the index of points including the last datetime.
    mr_mr only considers up to the date where rebalancing happens.
    The criterion is
    """
    index = dp.index
    base_dt = index[0]  # Timestamp('2016-01-29 00:00:00')
    last_dt = index[-1]  # Timestamp('2019-04-12 00:00:00')
    base_day = base_dt.day
    length = len(index.to_period('M').drop_duplicates()) - 1
    dt = base_dt  # Timestamp('2016-01-29 00:00:00')
    monthly_index = [base_dt]
    for _ in range(length):
        dt = dt + relativedelta(months=1) + relativedelta(
            day=base_day)  # We assume that a month has at least a single day in dp.
        month = dt.month
        if dt in index:  # The exact day of the next month.
            monthly_index.append(dt)
        else:  # Start looking for a later day.
            dt_var = dt + relativedelta(days=1)
            while dt_var.month == month:
                if dt_var in index:
                    monthly_index.append(dt_var)
                    break
                else:
                    dt_var = dt_var + relativedelta(days=1)
            else:  # Start looking for an earlier day.
                dt_var = dt - relativedelta(days=1)
                while True:
                    if dt_var in index:
                        monthly_index.append(dt_var)
                        break
                    else:
                        dt_var = dt_var - relativedelta(days=1)

    logging.debug(f'{dt}, {monthly_index[-1]}, {last_dt}')
    if monthly_index[-1] < last_dt:  # The last date is omitted in the monthly_index.
        return monthly_index + [last_dt], dp.reindex(monthly_index).pct_change().dropna()
    else:  # monthly_index[-1] == last_dt, no real transaction.
        # More precisely, a real transaction might occur on that day, but the ROR does not change.
        return monthly_index, dp.reindex(monthly_index[:-1]).pct_change().dropna()


# Monthly rebalancing monthly ROR data.
monthly_index, mr_mr = date_push_back(dp)

dp_cur = pd.DataFrame()
for stock in us_stock:
    values = dp[stock]*cur['cur']
    dp_cur = pd.concat([dp_cur,values],axis=1)
dp_cur.columns = us_stock

bm_cur = pd.DataFrame()
if bm_arg[0] in us_stock_list:
    values = bm_dp[bm_arg[0]]*cur['cur']
    bm_cur = pd.concat([bm_cur,values],axis=1)
else:
    bm_cur = bm_dp
bm_cur.columns = bm_arg



dp_cur = pd.concat([dp_cur,dp[kr_stock]],axis =1 )

mr_mr_cur = date_push_back(dp_cur)[1]



# data_arg = ['ACWV', 'AOR', 'BSCK']
# data_arg = ['AAXJ', 'ACWI', 'ACWV', 'AGG', 'AMJ', 'AMLP', 'ANGL', 'AOA', 'AOM', 'AOR']
# AAXJ ACWI ACWV AGG AMJ AMLP ANGL AOA AOM AOR
# mr_mr = pd.read_csv("monthly_rebalancing_monthly_ROR_foreignETF.csv", index_col='date')[data_arg].dropna()
# mr_mr.index = pd.to_datetime(mr_mr.index)
# raw daily price data
# no-rebalancing daily ROR data
# dp = (dp - dp.iloc[0]) / dp.iloc[0]
# df = pd.DataFrame(index=dp.index)
# for i in dp.columns:
#     A = dp[i].dropna()
#     A = A.apply(lambda x: (x - A[0]) / A[0])
#     df = df.join(A)
# df.to_csv("no_rebalancing_daily_ROR.csv", index=True, header=True)
# if date_start >= date_end or date_start > dp.index[-1] or date_end < dp.index[0]:
#     print('date_start or date_end overflow.')
#     sys.exit()
# dp = dp[(date_start <= dp.index) & (dp.index <= date_end)]

# mr_mr is used to calculate the weight whihc is used in the next month, hence retrieving 1 month
# mr_mr = mr_mr[(date_start <= mr_mr.index) & (mr_mr.index < date_end + relativedelta(day=1))]


# dp = pd.read_csv(os.path.join(os.getcwd(), 'foreign_ETF.csv'))
# dp['date'] = pd.to_datetime(dp['date'])
# dp[(dp['name'].isnull())]
# dp[((dp['name'].isnull())) & (dp['num']==1)]
# Count = dp.groupby(['name'])['num'].count().to_frame('count')
# Count.to_csv(os.path.join(os.getcwd(), 'count.csv'), index=True, header=True, encoding='euc-kr')
# del dp['num']
# dp['name'] = dp['name'].ffill()
# dp = dp.pivot_table(index='date', columns='name', values='price')
# dp = dp[data_arg].dropna()
# dp = dp[['ACWV', 'AOR', 'BSCK']].dropna()

# dp.index = dp.index.astype(str)
# Date_Func = lambda x: datetime.datetime(year=int(x[0:4]), month=int(x[4:6]), day=int(x[6:8]))
# dp.index = dp.index.map(Date_Func)
# dp[dp.index == 20180912].isnull().values.any()
# B = dp[dp.index == 20180912]
# B.columns[B.isnull().any()].tolist()
# AMS@BBCA 20180808
# dp.groupby(pd.TimeGrouper('M')).nth(0)
# mr_mr = dp.resample('M').first().pct_change().iloc[:-1]
# mr_mr = mr_mr.dropna()
# Date = mr_mr.index
# mr_mr.index = Date[:-1]

# mr_mr = pd.read_excel(os.path.join(os.getcwd(), 'ROR.xlsx'), 'monthly_ror', index_col=0, converters={'date': pd.to_datetime})
# mr_mr = mr_mr[sys.argv[2:]]
# dp = pd.read_excel(os.path.join(os.getcwd(), 'ROR.xlsx'), 'daily_twror', index_col=0, converters={'date': pd.to_datetime})
# dp = dp[sys.argv[2:]]
####################################################################################################################


def sharpe_weight(mr_mr=mr_mr, lookback=lookback, bounds=bounds):
    
    data_length = len(mr_mr.columns)
    loop_depth = len(mr_mr.index) - lookback + 1  # Running from lookback to len(mr_mr.index)
    # We can start from lookback - 1 if we want to take the most logical approach.
    if data_length == 1:
        return [[1.] for _ in range(loop_depth)]

    initial_value = np.repeat(1 / data_length, data_length)

    # store all the weights
    weight_list = []
   
    start = 0
    r_f = 0.02/12  # risk free interest rate
    # optimization only using mr_mr
    while start < loop_depth:
        data_lookback = mr_mr[start:start + lookback]

        # the objective function we want to minimize/maximize
        # given the ratio, we calculate the portfolio's variance / Sharpe ratio etc
        # currently, the objective function is fixed to Sharpe ratio alone
        mean = data_lookback.mean().values  # return of assets
        covar = np.cov(np.transpose(data_lookback))  # covariance of assets

        # mean = x @ pd.DataFrame(data_lookback.mean())
        # std = (x.T @ risk_factored_data.cov() @ x) ** 0.5
        # return (- mean / std)[0]
        # def data_lookback_std1(x):
        #     return (x.T @ data_lookback.cov() @ x) ** 0.5 - 0.1
        # def data_lookback_std2(x):
        #     return -(x.T @ data_lookback.cov() @ x) ** 0.5 + 0.45
        # cons = ({'type': 'eq', 'fun': weight_sum_constraint},
        #         {'type': 'ineq', 'fun': data_lookback_std1},
        #         {'type': 'ineq', 'fun': data_lookback_std2})

        # ineq = nonnegative
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        mean_sort = []
        for i, x in enumerate(mean):
            mean_sort.append([x, i])
        mean_sort = sorted(mean_sort)[::-1]  # from max return to min return
        weight_max = [0] * data_length  # find the weight when a combination of portfolio's return is maximum
        weight_sum = 0  # weight sum of weight_max
        for i in range(data_length):
            if bounds[mean_sort[i][1]][1] <= 1 - weight_sum:
                weight_max[mean_sort[i][1]] = bounds[mean_sort[i][1]][1]
                weight_sum += weight_max[mean_sort[i][1]]
            elif bounds[mean_sort[i][1]][0] <= 1 - weight_sum:
                weight_max[mean_sort[i][1]] = 1 - weight_sum
                weight_sum = 1  # weight sum이 1이 된다
            else:
                for j in range(i, data_length):
                    weight_max[mean_sort[j][1]] = bounds[mean_sort[j][1]][0]  # 뒤에 수익률 낮은 자산들에게 최소비중 던져준다
                    weight_sum += weight_max[mean_sort[j][1]]
                excess = weight_sum - 1  # 비중의 합이 1이 넘는 경우 초과분
                for j in range(i - 1, -1, -1):
                    if weight_max[mean_sort[j][1]] - excess >= bounds[mean_sort[j][1]][0]:
                        weight_max[mean_sort[j][1]] -= excess
                        break
                    else:
                        excess -= weight_max[mean_sort[j][1]] - bounds[mean_sort[j][1]][0]
                        weight_max[mean_sort[j][1]] = bounds[mean_sort[j][1]][0]
                break
        # if S < 1:
        #    shortage = 1 - S
        #    for j in range(len(mean_sort)):
        #        if w[mean_sort[j][1]] + shortage <= bounds[mean_sort[j][1]][1]:
        #            w[mean_sort[j][1]] += shortage
        #            break
        #        else:
        #            shortage -= bounds[mean_sort[j][1]][1] - w[mean_sort[j][1]]
        #            w[mean_sort[j][1]] = bounds[mean_sort[j][1]][1]

        mu_hat = mean - r_f
        if mean @ weight_max > r_f:  # 포트폴리오 조합중에 최대 수익률이 무위험 이자수익률보다 좋을때
            x = cp.Variable(data_length)

            obj = cp.Minimize(cp.quad_form(x, covar))
            pos_ineq_cons_G = np.reshape([-bounds[i][1] for i in range(data_length)] + [bounds[i][0] for i in range(data_length)],(2 * data_length, 1)) + np.concatenate((np.identity(data_length), -np.identity(data_length)), axis=0)

            constraints = [mu_hat@x == 1,  pos_ineq_cons_G@x <= 0]

            prob = cp.Problem(obj, constraints)
            prob.solve()

            cvxpy_w = [y / sum(x.value) for y in x.value]  # it should be transformed like the left
            if np.isnan(np.array(cvxpy_w).any()):
                weight_list.append(initial_value)  # nan 값이 나올때 그냥 동일비중을 던져준다
            else:
                weight_list.append(cvxpy_w)

        else:  # 어떠한 경우에도 무위험 이자수익률보다 낮을때
            # http://cvxopt.org/userguide/solvers.html 참고
            #x = cp.Variable(data_length)
            #var_x = cp.quad_form(x, covar)
            #ret_x = -mu_hat @ x
            #
            # #obj = cp.Minimize(ret_x*ret_x*var_x)
            # #constraints = [sum(x) == 1,  x <= [bounds[i][1] for i in range(len(bounds))],  x >= [bounds[i][0] for i in range(len(bounds))]]
            #prob = cp.Problem(obj, constraints)
            #prob.solve()
            #mod_cvxpy_w = [y for y in x.value]
            def objective(x):  # -modified Sharpe ratio
                x = np.array(x)
                return -(mu_hat @ x) * (x @ covar @ x) ** 0.5

            result = minimize(fun=objective,
                              x0=initial_value,
                              method='SLSQP',
                              constraints={'type': 'eq', 'fun': lambda x: sum(x) - 1},
                              options={'ftol': 1e-7, 'maxiter': 100},
                              bounds=bounds)
            slsqp_w = [y for y in result.x]
            if np.isnan(np.array(slsqp_w).any()):
                weight_list.append(initial_value)
            else:
                weight_list.append(slsqp_w)
        start += 1
        
    return weight_list


weight_list = sharpe_weight(mr_mr, lookback, bounds)

def wf_analysis(mr_mr, dp, bm_dp, acc_fund,acc, fee, plot=False):
    """ 

    :param mr_mr: Monthly ROR data.
    :param dp: Daily time-weighted ROR data.
    :param lookback:
    :param bounds:
    :return:
    """
    # directory = os.path.join(".", f"Walk-Forward Sharpe ratio {lookback} {lower_bound:.2f} {upper_bound:.2f}")
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    weight = pd.DataFrame(weight_list, columns=mr_mr.columns, index=monthly_index[lookback + 1:]).round(4)
    # More precisely, index = mr_mr.index[lookback:], but honestly index = monthly_index[lookback:] is the best.
    # But then again, this method of calculating return is flawed from the start.
    # dp_cut = dp[dp.index >= monthly_index[lookback]]
    dp_cut = dp[dp.index >= weight.index[0]]  # Cutting dp by the lookback

    cur_cut = cur[(cur.index >= weight.index[0]) & (cur.index <= dp.index[-1])]

    weight_fill = weight.reindex(dp_cut.index, method='bfill')  # Filling the data backwardly, hence monthly_index

    # group date based on weight.index
    def group_date(x):
        windex = weight.index
        length = len(windex) - 1
        if x == windex[-1]:  # the last one should belong to the previous group instead of being a separate one
            return length - 1
        for i in range(length):
            if (x >= windex[i]) & (x < windex[i + 1]):
                return i

    # group_date(pd.to_datetime('2018-09-12'))

    dp_cut['date_group'] = dp_cut.index.map(group_date)



    # no-rebalancing daily ror, groupbed by month
    # nr_dr_gm = dp_cut.groupby(pd.Grouper(freq="M")).apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))
    nr_dr_gm = dp_cut.groupby('date_group').apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))  # 종목별 리밸런싱 기간 수익률

    cur_ret = cur_cut.apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))  # 첫날 기준 리밸런싱 기간 수익률

    #cur_ret_pseudo = cur_cut.groupby('date_group').apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))  # 리밸런싱 기간별 환수익률

    del nr_dr_gm['date_group']


    weighted_nr_dr_gm = pd.DataFrame(weight_fill * nr_dr_gm).sum(axis=1).to_frame('walk_forward_ROR')  # The basic data of which everything is derived.

    # carry-over ROR for each rebalancing time
    rebalancing_ror = weighted_nr_dr_gm.groupby(dp_cut['date_group']).last() + 1  # 리밸런싱 기간별 수익률 즉 리밸런싱 날을 기준으로 다음 리밸런싱 전날까지 수익률

    rebalancing_ror = pd.DataFrame(index=weight.index[:-1],
                                   data=np.insert(rebalancing_ror.values[:-1], 0, 1),
                                   columns=['walk_forward_ROR']).cumprod()

    rebalancing_ror_fill = rebalancing_ror.reindex(nr_dr_gm.index, method='ffill')

    weighted_mr_dr = ((weighted_nr_dr_gm + 1) * rebalancing_ror_fill - 1)  # 비적립식 투자 첫날 기준 수익률

    # As for the logic behind the acc_fund ROR, check the 'acc_daily_explanation' excel file.
    # if type(acc_fund) == float:  # Ultimately generating weighted_mr_dr
    weighted_nr_dr_gm['acc'] = weighted_nr_dr_gm.index.map(group_date)



    # 포트폴리오 금액가중수익률 계산
    gogo = []
    gg = weighted_nr_dr_gm.groupby("acc")['walk_forward_ROR']
    xxx = acc_fund  # 리밸런싱 시점 투자금액 (리밸런싱 전날까지 모인 누적금액+ acc)

    for i in gg:
        total_acc = i[0] * acc
        df = i[1]
        nn = df.apply(lambda x: xxx * (1 + x) / (acc_fund + total_acc) - 1)
        gogo.extend(list(nn.values))
        xxx = (gogo[-1] + 1) * (acc_fund + total_acc) + acc
    weighted_nr_dr_gm['acc_rate'] = gogo
    weighted_mr_dr_acc = weighted_nr_dr_gm[['acc_rate']]
    weighted_mr_dr_acc.columns = ['walk_forward_ROR']  #고정환율 금액가중수익률
    # else:

    # 환율 금액가중수익률 계산 (원화기준)
    #gogo = []
    #xxx = acc_fund/cur.loc[cur_cut.index[0], 'cur'] - acc/cur.loc[cur_cut.index[0], 'cur']
    #for i in gg:
    #    total_acc = i[0] * acc #총 원화기준 리밸런싱 적립액
    #    df = i[1]
    #    month_dollar_acc = acc / cur.loc[df.index[0], 'cur']
    #    xxx += month_dollar_acc #달러기준 투자액
    #    for j in range(len(df)):
    #        gogo.append(xxx * (1 + df.iloc[j])*cur.loc[df.index[j], 'cur'] / (acc_fund + total_acc) - 1)
    #    xxx = (gogo[-1] + 1) * (acc_fund + total_acc)/cur.loc[df.index[-1], 'cur']
    #cur_cut['acc_rate'] = gogo
    #cur_acc = cur_cut[['acc_rate']]
    #cur_acc.columns = ['walk_forward_ROR'] #환율반영 원화기준 금액가중수익률
    # Fee
    for i in range(len(weighted_mr_dr)):
        weighted_mr_dr.iloc[i,0] = weighted_mr_dr.iloc[i,0] - fee*i
    for i in range(len(weighted_mr_dr_acc)):
        weighted_mr_dr_acc.iloc[i,0] = weighted_mr_dr_acc.iloc[i,0] - fee*i


    #cur_acc = cur_acc - fee
    #cur_acc.iloc[0, 0] = 0



    # PCT
    # Deprecated, most likely...
    weighted_mr_dr_pct = (weighted_mr_dr + 1).pct_change().fillna(0)  # 일별 수익률 (오늘-어제)/오늘

    ror_list = []

    ror_list_acc = []

    ror_list_pseudo = []  # This one also seems like deprecated...
    # not possible with omitted data.
    # the first and the last yet implemented yet.
    #cur_ror_list_acc = []
    # CY = pd.concat([weighted_mr_dr[['walk_forward_ROR']], weighted_mr_dr_acc[['walk_forward_ROR']],bm_non_acc[['SPY']]],axis =1)
    # CY.columns = ['non_acc','acc','benchmark_non_acc']
    # CY.plot(color =['r', 'b','gray'])

    if interval == 'W':
        logging.debug('Weekly interval.')
        first_ror = [weighted_mr_dr.index[0], 0]  # 비적립식 첫날기준 수익률
        last_ror = [weighted_mr_dr.index[-1], weighted_mr_dr.iloc[-1][0]]
        first_ror_acc = [weighted_mr_dr_acc.index[0], 0]  # 적립식 첫날기준 수익률
        last_ror_acc = [weighted_mr_dr_acc.index[-1], weighted_mr_dr_acc.iloc[-1][0]]
        first_ror_pseudo = [weighted_nr_dr_gm.index[0], 0]  # 리밸런싱 기간별 수익률
        last_ror_pseudo = [weighted_nr_dr_gm.index[-1], weighted_nr_dr_gm.iloc[-1][0]]

        #first_cur_acc = [cur_acc.index[0], 0]
        #last_cur_acc = [cur_acc.index[-1], cur_acc.iloc[-1][0]]



        weighted_mr_dr['YW'] = [int(str(i.year) + str(i.week).zfill(2)) for i in weighted_mr_dr.index]
        weighted_mr_dr_acc['YW'] = [int(str(i.year) + str(i.week).zfill(2)) for i in weighted_mr_dr_acc.index]
        weighted_nr_dr_gm['YW'] = [int(str(i.year) + str(i.week).zfill(2)) for i in weighted_nr_dr_gm.index]
        #cur_acc['YW'] = [int(str(i.year) + str(i.week).zfill(2)) for i in cur_acc.index]

        YW_group = weighted_mr_dr.groupby('YW')
        YW_acc_group = weighted_mr_dr_acc.groupby('YW')
        YW_group_pseudo = weighted_nr_dr_gm.groupby('YW')

        #YW_cur_acc = cur_acc.groupby('YW')

        for i in YW_group:
            sub_table = i[1].iloc[::-1]
            ror_list.append([sub_table.index[0], sub_table['walk_forward_ROR'][sub_table.index[0]]])

        for i in YW_acc_group:
            sub_table = i[1].iloc[::-1]
            ror_list_acc.append([sub_table.index[0], sub_table['walk_forward_ROR'][sub_table.index[0]]])

        for i in YW_group_pseudo:
            sub_table = i[1].iloc[::-1]
            ror_list_pseudo.append([sub_table.index[0], sub_table['walk_forward_ROR'][sub_table.index[0]]])

        #for i in YW_cur_acc:
            #sub_table = i[1].iloc[::-1]
            #cur_ror_list_acc.append([sub_table.index[0], sub_table['walk_forward_ROR'][sub_table.index[0]]])


        ror_list = collections.deque(sorted(ror_list))  # Default by first element (key=lambda x: x[0]), due to weird .week method in the last month.
        ror_list_acc = collections.deque(sorted(ror_list_acc))
        ror_list_pseudo = collections.deque(sorted(ror_list_pseudo))

        #cur_ror_list_acc = collections.deque(sorted(cur_ror_list_acc))

        # for i in ror_list:
        #     if i[0].day_name() != 'Friday':
        #         print(i[0], i[0].day_name())

        if ror_list[0][0] != first_ror[0]: ror_list.appendleft(first_ror)
        if ror_list[-1][0] != last_ror[0]: ror_list.append(last_ror)
        if ror_list_acc[0][0] != first_ror_acc[0]: ror_list_acc.appendleft(first_ror_acc)
        if ror_list_acc[-1][0] != last_ror_acc[0]: ror_list_acc.append(last_ror_acc)
        if ror_list_pseudo[0][0] != first_ror_pseudo[0]: ror_list_pseudo.appendleft(first_ror_pseudo)
        if ror_list_pseudo[-1][0] != last_ror_pseudo[0]: ror_list_pseudo.append(last_ror_pseudo)

        #if cur_ror_list_acc[0][0] != first_cur_acc[0]: cur_ror_list_acc.appendleft(first_cur_acc)
        #if cur_ror_list_acc[-1][0] != last_cur_acc[0]: cur_ror_list_acc.append(last_cur_acc)


        ror_list = list(ror_list)
        ror_list_acc = list(ror_list_acc)
        ror_list_pseudo = list(ror_list_pseudo)
        #cur_ror_list_acc = list(cur_ror_list_acc)

        final_dt = pd.to_datetime([x[0] for x in ror_list])
        #plt.plot(ror_list, ror_list_acc)
        #plt.show()
        # first_dt = w_index.popleft()
        # last_dt = w_index.pop()
        # # In python, unilke in JavaScript, 0 = Monday, and new week starts from Monday.
        # new_index = collections.deque(dt for dt in w_index if dt.day_name() == 'Friday')
        # new_index.appendleft(first_dt)
        # new_index.append(last_dt)
        # weighted_mr_dr = weighted_mr_dr.loc[new_index]
    else:
        final_dt = weighted_mr_dr.index
        for i in range(len(weighted_mr_dr.index)):
            ror_list.append([weighted_mr_dr.index[i], weighted_mr_dr['walk_forward_ROR'][i]])
            ror_list_acc.append([weighted_mr_dr_acc.index[i], weighted_mr_dr_acc['walk_forward_ROR'][i]])
            #cur_ror_list_acc.append([cur_acc.index[i], cur_acc['walk_forward_ROR'][i]])
        for i in range(len(weighted_nr_dr_gm.index)):
            ror_list_pseudo.append([weighted_nr_dr_gm.index[i], weighted_nr_dr_gm['walk_forward_ROR'][i]])

    weighted_mr_dr_pct = weighted_mr_dr_pct.loc[final_dt]
    www = []
    for i in range(len(weighted_mr_dr_pct.index)):
        www.append([weighted_mr_dr_pct.index[i], weighted_mr_dr_pct.iloc[i, 0]])

    if plot:
        # always saved as plot_image.png in the default download folder of the default browser
        # also temp-plot.html is created in the current directory
        fig = weighted_mr_dr.iplot(kind='scatter', asFigure=True)
        offline.plot(fig, image='png')

        time.sleep(3)
        # os.path.join(self.directory, self.name + '.png') causes an error when dealing with sharpe_ratio
        shutil.move(os.path.join(os.getcwd(), 'temp-plot.html'), os.path.join(os.getcwd(), 'plot.html'))

    w_list = []
    for i in range(len(weight)):
        ii = weight.iloc[[i], :]
        name = []
        for nn in ii.columns:
            name.append(nn)
            name.append(ii[nn][0])
        w_list.append(ii.index[0])
        w_list.append(str(name))

    #bm_list = []
    bm_list_acc = []
    bm_list_non_acc = []
    bm_dp = bm_dp[bm_dp.index >= weighted_mr_dr.index[0]]

    # deepcopy... cannot help it at this point.
    # bm is not buy and hold, check 'bm_is_not_buy_and_hold' excel file for the calculation.
    bm_dp_copy = copy.deepcopy(bm_dp)
    bm_non_acc = bm_dp_copy.apply(lambda x: (x - x[0]) / x[0])

    # if type(acc_fund) == float:

    # Forcing acc_fund, cannot help it.
    bm_dp_name = bm_dp.columns[0]
    bm_dp['acc'] = bm_dp.index.map(group_date)
    bm = bm_dp.groupby('acc')[bm_dp_name].apply(lambda x: (x - x.iloc[0]) / x.iloc[0]).to_frame(bm_dp_name)  # 리밸런싱 기간별 수익률
    bm['acc'] = bm.index.map(group_date)
    # r_bm = bm.groupby(bm['date_group']).last()
    # r_bm = pd.DataFrame(index=weight.index[:-1],
    #                     data=np.insert(r_bm.to_numpy()[:-1], 0, 0),
    #                     columns=[bm_dp_name])
    # r_bm_fill = r_bm.reindex(bm.index, method='ffill')
    # del bm['date_group']
    # bm = bm + r_bm_fill


    #고정환율 금액가중수익률
    gogo = []
    gg = bm.groupby("acc")[bm_dp_name]
    xxx = acc_fund
    for i in gg:
        total_acc = i[0] * acc
        df = i[1]
        nn = df.apply(lambda x: xxx * (1 + x) / (acc_fund + total_acc) - 1)
        gogo.extend(list(nn.values))
        xxx = (gogo[-1] + 1) * (acc_fund + total_acc) + acc
    bm['acc_rate'] = gogo
    bm = bm[['acc_rate']]
    bm.columns = [bm_dp_name]
    # else:
    #     bm = bm_dp.apply(lambda x: (x-x[0])/x[0])

    #환율반영 금액가중수익률
    #gogo = []
    #xxx = acc_fund / cur.loc[cur_cut.index[0], 'cur'] - acc / cur.loc[cur_cut.index[0], 'cur']
    #for i in gg:
    #    df = i[1]
    #    total_acc = i[0] * acc  # 총 원화기준 리밸런싱 적립액
    #    month_dollar_acc = acc / cur.loc[df.index[0], 'cur']
    #    xxx += month_dollar_acc  # 달러기준 투자액
    #    for j in range(len(df)):
    #        gogo.append(xxx * (1 + df.iloc[j]) * cur.loc[df.index[j], 'cur'] / (acc_fund + total_acc) - 1)
    #    xxx = (gogo[-1] + 1) * (acc_fund + total_acc) / cur.loc[df.index[-1], 'cur']
    #bm['cur_acc_rate'] = gogo
    #cur_bm = bm[['cur_acc_rate']]
    #del bm['cur_acc_rate']
    #cur_bm.columns = [bm_dp_name]



    # Fee
    #for i in range(len(bm)):
    #    bm.iloc[i,0] = bm.iloc[i,0]-fee*i
    #for i in range(len(bm_non_acc)):
    #    bm_non_acc.iloc[i,0] = bm_non_acc.iloc[i,0]-fee*i

    #cur_bm = cur_bm - fee
    #cur_bm.iloc[0, 0] = 0

    #cur_bm_list_acc = []

    if interval == 'W':
        bm = bm.loc[final_dt]
        bm_non_acc = bm_non_acc.loc[final_dt]
        #cur_bm = cur_bm.loc[final_dt]

        # for i in range(len(final_dt)):
        #     bm_list.append([bm.index[i], bm.iloc[i, 0]])
        # bm_dp['date_group'] = bm_dp.index.map(group_date)
        # no-rebalancing daily ror, groupbed by month
        # nr_dr_gm = dp_cut.groupby(pd.Grouper(freq="M")).apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))
        # bm_dp_gm = bm_dp.groupby('date_group').apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))
        # del bm_dp_gm['date_group']
        # bm_dp_gm = bm_dp_gm.loc[final_dt]
        for i in range(len(final_dt)):
            bm_list_acc.append([bm.index[i], bm.iloc[i, 0]])
            bm_list_non_acc.append([bm_non_acc.index[i], bm_non_acc.iloc[i, 0]])
            #cur_bm_list_acc.append([cur_bm.index[i], cur_bm.iloc[i, 0]])
    else:
        for i in range(len(bm.index)):
            bm_list_acc.append([bm.index[i], bm.iloc[i, 0]])
            bm_list_non_acc.append([bm_non_acc.index[i], bm_non_acc.iloc[i, 0]])
            #cur_bm_list_acc.append([cur_bm.index[i], cur_bm.iloc[i, 0]])

    # return ror_list, ror_list_pseudo, w_list, bm_list_pseudo

    #cur_ror_list = []  # 환율 고려한 비적립식 기준 수익률
    #cur_bm_list_non_acc = []  # 환율 고려한 벤치마크 비적립식 수익률




    #for i in range(len(ror_list)):
    #    cur_ror_list.append([ror_list[i][0], (1 + ror_list[i][1]) * (1 + cur_ret.loc[ror_list[i][0], 'cur']) - 1])  # 환율적용 비적립식 수익률

    #for i in range(len(bm_list_non_acc)):
    #    cur_bm_list_non_acc.append([bm_list_non_acc[i][0], (1 + bm_list_non_acc[i][1]) * (1 + cur_ret.loc[bm_list_non_acc[i][0], 'cur']) - 1])  # 환율적용 비적립식 벤치마크 수익률



    return ror_list, ror_list_acc, w_list, bm_list_non_acc, bm_list_acc


'''
def worker(start, mr_mr, dp, lookback, bnds, initial_value):
    data_lookback = mr_mr[start:start + lookback]

    # the objective function we want to minimize/maximize
    # given the ratio, we calculate the portfolio's variance / Sharpe ratio etc
    # currently, the objective function is fixed to Sharpe ratio alone
    def objective(x):
        mean = data_lookback.mean()
        return (np.dot(mean, x) - (0.02 / 12)) / (data_lookback @ np.transpose(x)).std()

        # mean = x @ pd.DataFrame(data_lookback.mean())
        # std = (x.T @ risk_factored_data.cov() @ x) ** 0.5
        # return (- mean / std)[0]

    # def data_lookback_std1(x):
    #     return (x.T @ data_lookback.cov() @ x) ** 0.5 - 0.1
    # def data_lookback_std2(x):
    #     return -(x.T @ data_lookback.cov() @ x) ** 0.5 + 0.45
    # cons = ({'type': 'eq', 'fun': weight_sum_constraint},
    #         {'type': 'ineq', 'fun': data_lookback_std1},
    #         {'type': 'ineq', 'fun': data_lookback_std2})

    # ineq = nonnegative
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    result = minimize(fun=objective,
                      x0=initial_value,
                      method='SLSQP',
                      constraints={'type': 'eq', 'fun': weight_sum_constraint},
                      options={'ftol': 1e-20, 'maxiter': 800},
                      bounds=bnds)

    if not result.success:  # type(result.success) == numpy.bool_ -> one shouldn't use 'is False', '==' works fine
        logging.debug(f'{start}th Walk Foward Failed!, {result.message}, {objective(result.x)}, {result.nit}')
    else:
        logging.debug(f"{start}th Walk Forward Successful, {objective(result.x)}, {result.nit}")

    if math.isnan(objective(result.x)):
        logging.debug("Warning: weight was not calculated correctly, falling back to the equal rate.")
        return initial_value
    else:
        return result.x
'''

'''
def wf_analysis_mp(mr_mr, dp, lookback=lookback, bounds=bounds, plot=False):
    """

    :param mr_mr: Monthly ROR data.
    :param dp: Daily time-weighted ROR data.
    :param lookback:
    :param bounds:
    :return:
    """
    data_length = len(mr_mr.columns)
    loop_depth = len(mr_mr.index) - lookback + 1  # running from lookback to len(mr_mr.index)
    initial_value = np.repeat(1 / data_length, data_length)
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    bnds = (bounds, ) * data_length  # currently all the strategies face the same lower bound condition

    # store all the weights
    weight_list = []
    # failed list; currently not functioning
    failed_list = []
    # success list; currently not functioning
    success_list = []

    start_list = np.arange(0, loop_depth)

    pool = mp.Pool()
    result = pool.starmap(worker, zip(start_list, itertools.repeat(mr_mr), itertools.repeat(dp), itertools.repeat(lookback), itertools.repeat(bnds), itertools.repeat(initial_value)))
    pool.close()
    pool.join()

    directory = os.path.join(".", f"Walk-Forward Sharpe ratio {lookback} {lower_bound:.2f} {upper_bound:.2f}")
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    weight = pd.DataFrame(weight_list, columns=mr_mr.columns, index=monthly_index[lookback + 1:]).round(4)
    dp_cut = dp[dp.index >= weight.index[0]]  # cutting dp by the lookback
    weight_fill = weight.reindex(dp_cut.index, method='bfill')  # filling the data backwardly, hence monthly_index

    # group date based on weight.index
    def group_date(x):
        windex = weight.index
        length = len(windex) - 1
        if x == windex[-1]:  # the last one should belong to the previous group instead of being a separate one
            return length - 1
        for i in range(length):
            if (x >= windex[i]) & (x < windex[i + 1]):
                return i
    # group_date(pd.to_datetime('2018-09-12'))

    dp_cut['date_group'] = dp_cut.index.map(group_date)

    # no-rebalancing daily ror, groupbed by month
    # nr_dr_gm = dp_cut.groupby(pd.Grouper(freq="M")).apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))
    nr_dr_gm = dp_cut.groupby('date_group').apply((lambda x: (x - x.iloc[0]) / x.iloc[0]))
    del nr_dr_gm['date_group']
    weighted_nr_dr_gm = pd.DataFrame(weight_fill * nr_dr_gm).sum(axis=1).to_frame('walk_forward_ROR')

    # carry-over ROR for each rebalancing time
    rebalancing_ror = weighted_nr_dr_gm.groupby(dp_cut['date_group']).last()
    rebalancing_ror = pd.DataFrame(index=weight.index[:-1],
                                   data=np.insert(rebalancing_ror.values[:-1], 0, 0),
                                   columns=['walk_forward_ROR']).cumsum()
    rebalancing_ror_fill = rebalancing_ror.reindex(nr_dr_gm.index, method='ffill')
    weighted_mr_dr = weighted_nr_dr_gm + rebalancing_ror_fill

    # not possible with omitted data
    if interval == 'W':
        if len(weighted_mr_dr) % 7 == 1:
            weighted_mr_dr = weighted_mr_dr[::7]
        else:
            weighted_mr_dr = weighted_mr_dr[::7].append(weighted_mr_dr.iloc[-1, [0]])

    ror_list = []
    for i in range(len(weighted_mr_dr.index)):
        ror_list.append([weighted_mr_dr.index[i], weighted_mr_dr['walk_forward_ROR'][i]])

    if plot:
        # always saved as plot_image.png in the default download folder of the default browser
        # also temp-plot.html is created in the current directory
        fig = weighted_mr_dr.iplot(kind='scatter', asFigure=True)
        offline.plot(fig, image='png')

        time.sleep(3)
        # os.path.join(self.directory, self.name + '.png') causes an error when dealing with sharpe_ratio
        shutil.move(os.path.join(os.getcwd(), 'temp-plot.html'), os.path.join(os.getcwd(), 'plot.html'))

    return ror_list, ';' + str(list(weight.iloc[-1]))
'''
# mr_mr.to_csv('mr_mr.csv')

if __name__ == "__main__":
    # t0 = time.time()wf_analysis(mr_mr, dp, bm_dp, acc_fund, fee, plot=False)
    if curr == 0:
        A, B, C, D, E = wf_analysis(mr_mr, dp, bm_dp, acc_fund, acc, fee)
        print(A) #거치식
        print(B) #적립식
        print(C) #비중값
        print(D) #거치식 벤치마크
        print(E)
    else:
        F, G, H, I, J = wf_analysis(mr_mr_cur, dp_cur, bm_cur, acc_fund, acc, fee)
        print(F) #환율반영 거치식
        print(G) #환율반영 적립식
        print(H) #비중값
        print(I) #환율반영 거치식 벤치마크
        print(J) #환율반영 적립식 벤치마크
    # print(time.time() - t0)
    # t1 = time.time()
    # wf_analysis_mp(mr_mr, dp, lookback=int(sys.argv[1]))
    # print(time.time() - t1)
