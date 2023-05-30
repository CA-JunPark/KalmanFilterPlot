from KF_Plot import *
from tqdm import tqdm

k = 2000 # number of steps      
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
    return np.array([x[0]])

def H_j(x):
    return np.array([[1,0,0]])

# Q = np.zeros((m,m)) # Model error 
# Q = np.diag([0.0001,0.0001,0.0001])
Q = np.diag([0.1,0.1,0.1])
R = np.diag([0.1]) # Observation Error

# true states 
       # starting point
xt0 = np.array([[0],[1],[1.05]])
xt = xt0

Ys = xt0[0] # observations

c = 10 # observation Frequency

for i in range(k):
    x = M(xt[:,-1]) 
    xt = np.column_stack((xt,x))

P = np.diag([1,1,1]) # background Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # background error
X0 = xt0 + e # background X

enkf = StochasticEnKF(m, j, X0, P, M, Q, R, H, H_j, n=10)

for i in tqdm(range(k),desc="Filtering"):
    if i % c == 0:
        enkf.enForecast()
        y = np.random.multivariate_normal([xt[:,i+1][0]], R).reshape((j,1)) # observations with error
        enkf.enAnalyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        enkf.enForward()
        Ys = np.column_stack((Ys, np.array([np.inf])))
        
enkf.plot_all(xt,has_obs=[0],Ys=Ys,plotXm=False,show=False)
# enkf.plot_one(0, xt, 0, Ys)
# enkf.plot_two(0, 1, xt, ym1=0, Ys=Ys) 

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xt[0,:], xt[1,:], xt[2,:], 'k', lw=0.5, label='True')
ax.plot(enkf.get_xk(0),enkf.get_xk(1),enkf.get_xk(2),'r--', lw=0.5, label='Estimated')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.legend(loc='best')
ax.set_title("Lorenz Attractor")

plt.figure()
plt.plot(xt[0,:], xt[1,:], 'r', lw=0.5, label='True')
plt.plot(enkf.get_xk(0),enkf.get_xk(1),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Y")

plt.figure()
plt.plot(xt[0,:], xt[2,:], 'r', lw=0.5, label='True')
plt.plot(enkf.get_xk(0),enkf.get_xk(2),'b--', lw=0.5, label='Estimated')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc='best')
plt.title("X vs Z")

plt.show()

