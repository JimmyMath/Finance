import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
start_time = time.time()

df = pd.read_excel("sp500.xlsx")
df = df.drop("Date",axis =1)
B = np.transpose(df.values)

C = np.cov(B) #Covariance matrix

mu = np.array([np.mean(B[i]) for i in range(len(B))]) #expected returns

r_f = 0.005

m = 0.001
M = 0.8

initial_value = [1./len(B)]*len(B)


Q =  np.linalg.solve(C, np.identity(len(B))) #Inverse of Covariance matrix
mu_1 = mu - r_f

t = [0]*len(B)

for i in range(len(B)):
    for j in range(len(B)):
        t[i] += Q[i][j]*mu_1[j]

t = t/np.sum(t)


bounds = tuple((m,M) for _ in range(len(B)))


def objective(x):
    x = np.array(x)
    return (r_f-x@mu)/ (x@C@x)**0.5


result = minimize(fun=objective,
                  x0=initial_value,
                  method='SLSQP',
                  constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                  options={'ftol': 1e-7, 'maxiter': 1000},
                  bounds=bounds)
print(result)
print("SLSQP:",result.x)
print("Sharpe ratio:",-objective(result.x))

#print("이론값:",t,-objective(t))


print("start_time", start_time) #출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.
print("Excecution time: --- %s seconds ---" %(time.time() - start_time))