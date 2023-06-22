from KF_Plot import *
from tqdm import tqdm

N = 40
F = 8

k = 80 # number of steps      
m = N # dimension of X
j = 40 # dimension of Y

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
    return x[[i for i in range(0,j)],:].reshape((j,1))

h = [[0 for i in range(m)] for o in range(j)]
u = 0
for i in range(j):
    h[i][u] = 1
    u += 1    
h = np.array(h)

def H_j(x):
    return h

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

n=50
enkf = EnKF(m, j, X0, P, M, R, H, H_j, n=n)
etkf = ETKF(m, j, X0, P, M, R, H, n=n)
enkf.enX = etkf.enX.copy() # using identical initial ensembles
for i in tqdm(range(k),desc="Filtering"):
    if i % c == 0:
        etkf.etForecast()
        enkf.enForecast()
        mu = np.random.multivariate_normal([0]*j, R).reshape((j,1))
        y = h @ xt[:,i+1].reshape((m,1)) + mu # observations with error
        etkf.etAnalyze(y)
        enkf.enAnalyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        etkf.etForward()
        enkf.enForward()
        Ys = np.column_stack((Ys, np.array([np.inf]*j)))
    etkf.RMSDSave(etkf.X_cStack, xt)
    enkf.RMSDSave(enkf.X_cStack, xt)

li = [i for i in range(10)]
etkf.plot_some(li, xt, has_obs=li, Ys=Ys, plotXm=False, show=False, filters=[enkf])
etkf.plot_RMSD(filters=[enkf], show=True)

