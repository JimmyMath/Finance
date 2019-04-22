#Sharpe ratio maximization using CVXOPT

import numpy as np
import pandas as pd
import time
from cvxopt import matrix
from cvxopt.solvers import qp
start_time = time.time()

df = pd.read_excel("sp500.xlsx") #150 equities of S&P500
df = df.drop("Date",axis =1)

B = np.transpose(df.values)

C = np.cov(B) #Covariance matrix

Q =  np.linalg.solve(C, np.identity(len(B))) #Inverse of Covariance matrix

mu = np.array([np.mean(B[i]) for i in range(len(B))]) #expected returns


r_f = 0.005
mu_1 = mu - r_f


def objective(x):
    x = np.array(x)
    return (r_f-x@mu)/ (x@C@x)**0.5

m = 0.001 #minimun bound
M = 0.8 #Maximum bound

G = np.transpose([[-M]*len(B)+[m]*len(B) for _ in range(len(B))])-np.concatenate((-np.identity(len(B)), np.identity(len(B)) ), axis=0)

C = matrix(C)
q = matrix([0.0]*len(B))

G = matrix(G)

h =matrix([0.0]*(2*len(B)) )

mu_bar = mu-r_f

A_ = matrix(np.reshape(mu_bar,(1 , len(B))   ))

b = matrix(1.0)

sol = qp( C, q, G , h, A_ , b )

z = [y/sum(sol['x']) for y in sol['x']]

print("CVXOPT:",z)
print("Share ratio:",-objective(z))
print("Execution time: --- %s seconds ---" %(time.time() - start_time))
