from KF_Plot import *
from tqdm import tqdm

"""
Comparing two obs vs one obs
    [[x]
X =  [y]
     [z]]
enkf only observes x
enkf2 observes x and y
They share observations of x
"""

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

def H_j2(x):
    return np.array([[1,0,0],[0,1,0]])

def H2(x):
    return np.array([x[0],x[1]])

# Q = np.zeros((m,m)) # Model error (assume no model error)
Q = np.diag([0.0001,0.0001,0.0001])
R = np.diag([0.5]) # Observation Error
R2 = np.diag([0.5, 0.5])
# true states 
       # starting point
xt0 = np.array([[0],[1],[1.05]])
xt = xt0
Ys = xt0[0] # observations 1
Ys2 = np.array([xt0[0], xt0[2]]) # observations 2 

c = 25 # observation Frequency

for i in range(k):
    x = M(xt[:,-1]) 
    xt = np.column_stack((xt,x))

P = np.diag([1,1,1]) # background Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # background error
X0 = xt0 + e # background X

enkf = StochasticEnKF(m, 1, X0, P, M, Q, R, H, H_j, n=10)
enkf2 = StochasticEnKF(m, 2, X0, P, M, Q, R2, H2, H_j2, n=10)
for i in tqdm(range(k),desc="Filtering"):
    if i % c == 0:
        enkf.enForecast()
        enkf2.enForecast()
        y = np.random.multivariate_normal([xt[:,i+1][0]], R).reshape((1,1)) # observations with error
        y2 = np.row_stack((y, np.random.multivariate_normal([xt[:,i+1][1]], R).reshape((1,1))))
        enkf.enAnalyze(y)
        enkf2.enAnalyze(y2)
        Ys = np.column_stack((Ys, y))
        Ys2 = np.column_stack((Ys2, y2))
    else:
        enkf.enForward()
        enkf2.enForward()
        Ys = np.column_stack((Ys, np.array([np.inf])))
        Ys2 = np.column_stack((Ys2, np.array([[np.inf],[np.inf]])))

# enkf.plot_all(xt,has_obs=[0],Ys=Ys,plotXm=True)
# enkf2.plot_all(xt,has_obs=[0,1], Ys=Ys2, plotXm=True)
# enkf.plot_one(0, xt, 0, Ys)
# enkf.plot_two(0, 1, xt, ym1=0, Ys=Ys) 

enkf.compare_all(enkf2, xt, has_obs=[0,1], Ys=Ys2, plotXm=False)

""" 
R = 0.1
StochasticEnKF 1 Obs: 10.950961691661407
StochasticEnKF 2 Obs: 6.975195175720216

R = 0.5 
StochasticEnKF 1 Obs: 222.36212813968243
StochasticEnKF 2 Obs: 10.236436384657338
"""

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(xt[0,:], xt[1,:], xt[2,:], 'k', lw=0.5, label='True')
# ax.plot(enkf.get_xk(0),enkf.get_xk(1),enkf.get_xk(2),'r--', lw=0.5, label='Estimated')
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.legend(loc='best')
# ax.set_title("Lorenz Attractor")

# plt.figure()
# plt.plot(xt[0,:], xt[1,:], 'r', lw=0.5, label='True')
# plt.plot(enkf.get_xk(0),enkf.get_xk(1),'b--', lw=0.5, label='Estimated')
# plt.xlabel("X Axis")
# plt.ylabel("Y Axis")
# plt.legend(loc='best')
# plt.title("X vs Y")

# plt.figure()
# plt.plot(xt[0,:], xt[2,:], 'r', lw=0.5, label='True')
# plt.plot(enkf.get_xk(0),enkf.get_xk(2),'b--', lw=0.5, label='Estimated')
# plt.xlabel("X Axis")
# plt.ylabel("Y Axis")
# plt.legend(loc='best')
# plt.title("X vs Z")

plt.show()

