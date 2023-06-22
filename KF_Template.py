from KF_Plot import *
from tqdm import tqdm

"""
KF_Plot Twin Experiment Template 

replace "<>" for user's model
add more steps or features if needed

See examples and KF_Plot.py for more details
"""

"""Variables and Operators Set Up"""
k = "<int>" # number of steps      
m = "<int>" # dimension of X
j = "<int>" # dimension of Y

dt = "<float>" # time step

# x is mx1 vector

def M(x): # Model Operator 
    "<>"
    return "<>" # must return mx1 vector

def H(x): # Observation Operator
    "<>"
    return "<>" # must return jxm vector

def H_j(x): # Jacobian of H if H is non-linear
    "<>"
    return "<>" # must return jxm vector


Q = "<mxm matrix>" # Model error 

R = "<jxj matrix>" # Observation Error

# true states 
       # starting point
xt0 = "<mx1> vector"
xt = xt0

Ys = np.array([np.inf]*j) # observations 
# This is initial observation. Observations will be stacked columnwise

c = "<int>" # observation Frequency

for i in range(k): # generate True State
    x = M(xt[:,-1]) 
    xt = np.column_stack((xt,x))

P = "<mxm matrix>" # initial Covariance 
e = np.random.multivariate_normal([0, 0, 0], P, size=(1)).T # initial error
X0 = xt0 + e # initial X


filter = "<Kalman Filter>"
# ex) etkf = ETKF(m, j, X0, P, enM, R, enH, n=10)

# Use correct forecast, analyze, forward. Check the prefixes ex) etforward() != forward()
for i in tqdm(range(k),desc="Filtering"):
    if i % c == 0:
        #"<filter.forecast()>"
        y = np.random.multivariate_normal(H(xt[:,i+1]).T, R).reshape((j,1)) # observations with error
        #"<filter.analyze(y)>"
        Ys = np.column_stack((Ys, y))
    else:
        #<"filter.forward()>"
        Ys = np.column_stack((Ys, np.array([np.inf]*j)))
    # filter.RMSDSave(filter.X_cStack, xt)
    
# "<filter.plot_all(...)>"
# "<filter.plot_RMSD()>"


