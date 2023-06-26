# from KF_Plot_concurrent import *
from KF_Plot import *

from tqdm import tqdm

N = 40
F = 8

k = 80 # number of steps      
m = N # dimension of X
j = 20 # dimension of Y

dt = 0.05

def L96(x):
    """Lorenz 96"""
    d = np.zeros(N)
    for i in range(N):
        d[i] = (x[((i + 1) % N)][0] - x[i - 2][0]) * x[i - 1][0] - x[i][0] + F
    return np.array(d).reshape((m,1))

def RK4(x):
    """Fourth Order Runge-Kutta"""    
    k1 = L96(x)
    k2 = L96(x + 0.5*k1*dt)
    k3 = L96(x + 0.5*k2*dt)
    k4 = L96(x + k3*dt)
    return x + dt*(k1+2*k2+2*k3+k4)/6 
    
def M(x): # Model Operator
    return RK4(x)

def H(x): # Observation Operator
    return x[[i for i in range(m) if i%2==0],:].reshape((j,1))

"""Localization operator varies in different systems"""
def L(m): # Localization Operator 
    # This example only works j=m/2 
    # AND 
    # even variables have observation. 
    # Localization distance = 1 grid point
    if (m%2==0):
        u = int(m/2)
        if (m == N-2):
            b = [u-1,u,0]
        else:
            b = [u-1,u,u+1]
    else:
        if (m == (N-1)):
            b = [int((N-1)/2),0] 
        else:
            b=[int((m)/2),int((m+1)/2)] 
    return b

R = np.diag([0.1]*j) # Observation Error

# true states 
       # starting point
xt0 = F * np.ones((N,1))
xt0[0] += 0.1
xt = xt0

Ys = np.array([np.inf]*j) # observations

c = 1 # observation Frequency

for i in range(k):
    x = M(xt[:,-1].reshape((N,1))) 
    xt = np.column_stack((xt,x))
P = np.diag([1]*N) # initial Covariance 
e = np.random.multivariate_normal([0.]*N, P, size=(1)).T # initial error
X0 = xt0 + e # initial X

if __name__ == '__main__':
    n = 10
    letkf = LETKF(m, j, X0, P, M, R, H, L, n=n, rho=1.14)

    for i in tqdm(range(k),desc="Filtering"):
        if i % c == 0:
            letkf.leForecast()
            mu = np.random.multivariate_normal([0]*j, R).reshape((j,1))
            y = H(xt[:,i+1].reshape((m,1))) + mu # observations with error
            letkf.leAnalyze_Parallel(y)
            Ys = np.column_stack((Ys, y))
        else:
            letkf.leForward()
            Ys = np.column_stack((Ys, np.array([np.inf]*j)))
        letkf.RMSDSave(letkf.X_cStack, xt)

    li=[i for i in range(20) if i%2==0]
    letkf.plot_some(li, xt, has_obs=li, Ys=Ys, plotXm=False, show=False)
    letkf.plot_RMSD(show=True)
