#Sharpe ratio maximization using SLSQP

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
start_time = time.time()

df = pd.read_excel("sp500.xlsx") #150 equities of S&P500
df = df.drop("Date",axis =1)
B = np.transpose(df.values)

C = np.cov(B) #Covariance matrix

mu = np.array([np.mean(B[i]) for i in range(len(B))]) #expected returns

r_f = 0.005 #risk free interest rate

m = 0.001 #minimum bound
M = 0.8 #maximum bound

initial_value = [1./len(B)]*len(B)

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

print("SLSQP:",result.x)
print("Sharpe ratio:",-objective(result.x))
print("Excecution time: --- %s seconds ---" %(time.time() - start_time))
