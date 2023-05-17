from KF_Plot import *

""" 
1D-Car example
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

Has Background Noise (initial error)
"""

k = 25 # number of iterations      
m = 3 # dimension of X
j = 1 # dimension of Y

dt = 5 # measure every 5 seconds
M = np.array([[1, dt, dt**2/2], [0,1,dt], [0,0,1]]) # Model Matrix
H = np.array([[1,0,0]]) # Observation Matrix
Q = np.zeros((m,m)) # Model error (assume no model error)
R = np.array([[0]]) # Observation Error

# true states 
       # initial  pos vel acc
xt0 = np.array([[1000],[40],[-1]])
xt = xt0
Ys = np.random.normal(xt[0], R)
for i in range(k):
    x = M @ xt[:,-1]
    xt = np.column_stack((xt,x))

q = [100, 10, 1] # Variance of each variable
P = np.diag(q) # background Covariance 
epsilon = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # background error
X0 = xt0 + epsilon # background X

kf = KF(m, j, X0, P, M, Q, R, H)
for i in range(k):
    kf.forecast()
    y = np.random.normal(xt[:,i+1][0], R) # observations with error
    kf.analyze(y)
    Ys = np.column_stack((Ys, y))
    
# kf.plot_all(xt,has_obs=[0],Ys=Ys)
# kf.plot_one(0, xt, 0, Ys)
kf.plot_two(0,1, xt, ym1=0, Ys=Ys) 