from KF_Plot import *
from tqdm import tqdm

k = 2000 # number of steps      
m = 3 # dimension of X
j = 3 # dimension of Y

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
    return x

def L(m):
    return [0,1,2]
    
R = np.diag([1]*j) # Observation Error

# true states 
       # starting point
xt0 = np.array([[0],[1],[1.05]])
xt = xt0

Ys = np.array([np.inf]*j) # observations

c = 5 # observation Frequency

for i in range(k):
    x = M(xt[:,-1].reshape((m, 1))) 
    xt = np.column_stack((xt,x))

P = np.diag([1.]*m) # initial Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # initial error
X0 = xt0 + e # initial X

etkf = ETKF(m, j, X0, P, M, R, H, n=10, rho=1.14)
letkf = LETKF(m, j, X0, P, M, R, H, L, n=10, rho=1.14)
letkf.enX = etkf.enX.copy() # using identical initial ensembles
for i in tqdm(range(k)):
    if i % c == 0:
        etkf.etForecast()
        letkf.leForecast()
        y = np.random.multivariate_normal(xt[:,i+1], R).reshape((j,1)) # observations with error
        etkf.etAnalyze(y)
        letkf.leAnalyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        etkf.etForward()
        letkf.leForward()
        Ys = np.column_stack((Ys, np.array([np.inf]*j)))
    etkf.RMSDSave(etkf.X_cStack, xt)
    letkf.RMSDSave(letkf.X_cStack, xt)
etkf.plot_all(xt, has_obs=[0,1,2], Ys=Ys, plotXm=False, show=False, filters=[letkf])
etkf.plot_RMSD(filters=[letkf], show=False)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xt[0,:], xt[1,:], xt[2,:], 'k', lw=0.5, label='True')
ax.plot(letkf.get_xk(0),letkf.get_xk(1),letkf.get_xk(2),'r--', lw=0.5, label='Estimated')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.legend(loc='best')
ax.set_title("Lorenz Attractor")

plt.figure()
plt.plot(xt[0,:], xt[1,:], 'r', lw=0.5, label='True')
plt.plot(letkf.get_xk(0),letkf.get_xk(1),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Y")

plt.figure()
plt.plot(xt[0,:], xt[2,:], 'r', lw=0.5, label='True')
plt.plot(letkf.get_xk(0),letkf.get_xk(2),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Z")

plt.show()

