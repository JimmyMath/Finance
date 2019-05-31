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
    
    from scipy.optimize import minimize
    import logging

    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s  -  %(message)s')
    logging.disable(logging.DEBUG)

  

def wf_analysis(mr_mr, dp, bm_dp, acc_fund, fee, lookback=lookback, bounds=bounds, plot=False):
  
    data_length = len(mr_mr.columns)
    loop_depth = len(mr_mr.index) - lookback + 1  # Running from lookback to len(mr_mr.index)
    # We can start from lookback - 1 if we want to take the most logical approach.
    initial_value = np.repeat(1 / data_length, data_length)

    # store all the weights
    weight_list = []

    start = 0
    r_f = 0.02 / 12  # risk free interest rate
    # optimization only using mr_mr
    while start < loop_depth:
        data_lookback = mr_mr[start:start + lookback]

        mean = data_lookback.mean().values  # return of assets
        covar = np.cov(np.transpose(data_lookback))  # covariance of assets

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
      
        mu_hat = mean - r_f
        if mean @ weight_max > r_f:  # 포트폴리오 조합중에 최대 수익률이 무위험 이자수익률보다 좋을때
            def objective(x):  # -modified Sharpe ratio
                x = np.array(x)
                return x @ covar @ x

            pos_ineq_cons_G = np.reshape([-bounds[i][1] for i in range(data_length)] + [bounds[i][0] for i in range(data_length)],(2 * data_length, 1)) + np.concatenate((np.identity(data_length), -np.identity(data_length)), axis=0)

            result_slsqp = minimize(fun=objective,
                              x0=initial_value,
                              method='SLSQP',
                              constraints = [{'type': 'eq', 'fun': lambda x: mu_hat@x - 1},
                                             {'type':'ineq','fun': lambda x: -pos_ineq_cons_G @ x}],
                              options={'ftol': 1e-7, 'maxiter': 100})

            slsqp_w = [y / sum(result_slsqp.x) for y in result_slsqp.x]  # it should be transformed like the left
            if np.isnan(np.array(slsqp_w).any()):
                weight_list.append(initial_value)  # nan 값이 나올때 그냥 동일비중을 던져준다
            else:
                weight_list.append(slsqp_w)
        else:  # 어떠한 경우에도 무위험 이자수익률보다 낮을때
            # http://cvxopt.org/userguide/solvers.html 참고

            def objective(x):  # -modified Sharpe ratio
                x = np.array(x)
                return -(mu_hat @ x) * (x @ covar @ x) ** 0.5

            mod_result = minimize(fun=objective,
                              x0=initial_value,
                              method='SLSQP',
                              constraints={'type': 'eq', 'fun': lambda x: sum(x) - 1},
                              options={'ftol': 1e-7, 'maxiter': 100},
                              bounds=bounds)
            mod_slsqp_w = [y for y in mod_result.x]
            if np.isnan(mod_slsqp_w.any()):
                weight_list.append(initial_value)
            else:
                weight_list.append(mod_slsqp_w)
        start += 1
  

if __name__ == "__main__":
    A, B, C, D, E, F, G, H, I = wf_analysis(mr_mr, dp, bm_dp, acc_fund, fee)
    print(A) #거치식
    print(B) #적립식
    print(C) #비중값
    print(D) #거치식 벤치마크
    print(E) #적립식 벤치마크
    print(F) #환율반영 거치식
    print(G) #환율반영 적립식
    print(H) #환율반영 거치식 벤치마크
    print(I) #환율반영 적립식 벤치마크
