from KF_Plot import *
import matplotlib.pyplot as plt
from tqdm import tqdm

""" 
1D-Car twin experiment example
    [[pos]
X =  [vel]
     [acc]]

Y = [[pos]]

Fixed acceleration

Model (fixed a_k is applied to the car) (physics)                                      
pos_k = pos_(k-1) + vel_(k-1) * dt + a_k * dt^2 / 2
vel_k = vel_(k-1) + a_k * dt 
x_k = M * x_(k-1) + q_(k-1)
      [[1 dt dt^2/2]
    =  [0  1   dt  ]  x_(k-1) + q_(k-1)
       [0  0   1   ]]

Has initial Noise (initial error)
"""

k = 200 # number of steps 
m = 3 # dimension of X
j = 1 # dimension of Y

dt = 0.1 # measure every dt seconds
M = np.array([[1, dt, dt**2/2], [0,1,dt], [0,0,1]]) # Model Matrix
H = np.array([[1,0,0]]) # Observation Matrix
Q = np.zeros((m,m)) # Model error (assume no model error)
R = np.array([[0.1]]) # Observation Error

# true states 
       # initial  pos vel acc
xt0 = np.array([[0],[20],[-1]])
xt = xt0

Ys = np.array([np.inf]*j) # observations

c = 1 # observation Frequency

for i in range(k):
    x = M @ xt[:,-1]
    xt = np.column_stack((xt,x))

P = np.diag([5, 5, 1]) # initial Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # initial error
X0 = xt0 + e # initial X
kf = KF(m, j, X0, P, M, Q, R, H)

for i in tqdm(range(k), desc="Filtering"):
    y = np.random.normal(xt[:,i+1][0], R) # observations with error
    if (i % c == 0):
        kf.forecast()
        kf.analyze(y)
        Ys = np.column_stack((Ys, y))
    else:
        kf.forward()
        Ys = np.column_stack((Ys, np.inf))
    
kf.plot_all(xt, has_obs=[0], Ys=Ys) 