from KF_Plot import *
from tqdm import tqdm

k = 4000 # number of steps      
m = 3 # dimension of X
j = 1 # dimension of Y

dt = 0.01
def L63(x):
    """Lorenz 63"""
    s=10
    r=28
    b=8/3
    dxdt = s * (x[1] - x[0])
    dydt = x[0] * (r - x[2]) - x[1]
    dzdt = x[0] * x[1] - b * x[2]
    return np.array([dxdt,dydt,dzdt])

def RK4(x):
    """Fourth Order Runge-Kutta"""    
    k1 = L63(x)
    k2 = L63(x + 0.5*k1*dt)
    k3 = L63(x + 0.5*k2*dt)
    k4 = L63(x + k3*dt)
    return x + dt*(k1+2*k2+2*k3+k4)/6 

def M(x): # Model Operator
    return RK4(x)

def H(x): # Observation Operator
    return np.array([x[1]])

def H_j(x):
    return np.array([[0,1,0]])

# Q = np.zeros((m,m)) # Model error 
Q = np.diag([0.001,0.001,0.001])
# Q = np.diag([0.1,0.1,0.1])
R = np.diag([0.1]) # Observation Error

# true states 
       # starting point
xt0 = np.array([[0],[1],[1.05]])
xt = xt0

Ys = np.array([np.inf]*j) # observations

c = 10 # observation Frequency

for i in range(k):
    x = M(xt[:,-1]) 
    xt = np.column_stack((xt,x))

P = np.diag([1.,1.,1.]) # initial Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # initial error
X0 = xt0 + e # initial X

# Using different inflation factors
etkf = ETKF(m, j, X0, P, M, R, H, n=10)
etkf2 = ETKF(m, j, X0, P, M, R, H, n=10, rho=1.1)
etkf3 = ETKF(m, j, X0, P, M, R, H, n=10, rho=1.2)
etkf2.enX = etkf.enX.copy() # using identical initial ensembles
etkf3.enX = etkf.enX.copy() 
for i in tqdm(range(k),desc="Filtering"):
    if i % c == 0:
        etkf.etForecast()
        etkf2.etForecast()
        etkf3.etForecast()
        y = np.random.multivariate_normal(H(xt[:,i+1]).T, R).reshape((j,1)) # observations with error
        etkf.etAnalyze(y)
        etkf2.etAnalyze(y)
        etkf3.etAnalyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        etkf.etForward()
        etkf2.etForward()
        etkf3.etForward()
        Ys = np.column_stack((Ys, np.array([np.inf])))
    etkf.RMSDSave(etkf.X_cStack, xt)
    etkf2.RMSDSave(etkf2.X_cStack, xt)
    etkf3.RMSDSave(etkf3.X_cStack, xt)
    
etkf.plot_all(xt, has_obs=[1], Ys=Ys, filters=[etkf2, etkf3], plotXm=False, show=False, cov=True)

etkf.plot_RMSD(filters=[etkf2,etkf3],show=False)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xt[0,:], xt[1,:], xt[2,:], 'k', lw=0.5, label='True')
ax.plot(etkf.get_xk(0),etkf.get_xk(1),etkf.get_xk(2),'r--', lw=0.5, label='Estimated')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.legend(loc='best')
ax.set_title("Lorenz Attractor")

plt.figure()
plt.plot(xt[0,:], xt[1,:], 'r', lw=0.5, label='True')
plt.plot(etkf.get_xk(0),etkf.get_xk(1),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Y")

plt.figure()
plt.plot(xt[0,:], xt[2,:], 'r', lw=0.5, label='True')
plt.plot(etkf.get_xk(0),etkf.get_xk(2),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Z")

plt.show()

