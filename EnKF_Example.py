from KF_Plot import *
from tqdm import tqdm

k = 500 # number of iterations      
m = 3 # dimension of X
j = 1 # dimension of Y

dt = 0.1 # measure every 5 seconds
M = np.array([[1, dt, dt**2/2], [0,1,dt], [0,0,1]]) # Model Matrix
def enM(x):
    return M @ x
H = np.array([[1,0,0]]) # Observation Matrix
def enH(x): # Observation Operator
    return np.array([x[0]])
def enH_j(x):
    return np.array([[1,0,0]])
# Q = np.zeros((m,m)) # Model error (assume no model error)
Q = np.diag([0.001,0.001,0.001])
R = np.array([[1]]) # Observation Error

# true states 
       # initial  pos vel acc
xt0 = np.array([[0],[40],[-1]])
xt = xt0

Ys = np.random.normal(xt[0], R) # observations

c = 5 # observation frequency

for i in range(k):
    x = M @ xt[:,-1]
    xt = np.column_stack((xt,x))

q = [50, 10, 1] # Variance of each variable
P = np.diag(q) # background Covariance 
e = np.random.multivariate_normal([0, 0, 0], np.sqrt(P), size=(1)).T # background error
X0 = xt0 + e # background X

enkf = StochasticEnKF(m, j, X0, P, enM, Q, R, enH, enH_j, n=100)
kf = KF(m, j, X0, P, M, Q, R, H)
for i in tqdm(range(k), desc='Filtering'):
    if i % c == 0:
        enkf.enForecast()
        kf.forecast()
        y = np.random.normal(xt[:,i+1][0], R).reshape((j,1)) # observations with error
        enkf.enAnalyze(y)
        kf.analyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        enkf.enForward()
        kf.forward()
        Ys = np.column_stack((Ys, np.inf))

# enkf.plot_all(xt,has_obs=[0],Ys=Ys)
# enkf.plot_one(0, xt, 0, Ys)
# enkf.plot_two(0, 1, xt, ym1=0, Ys=Ys)
enkf.compare_two(kf, 0, 1, xt, ym1=0, Ys=Ys) 

# plt.show()

