# KalmanFilterPlot
## Main File

```KF_Plot.py```
<details>
    <summary>Code</summary>

```KF_Plot.py```
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.widgets import Slider

class KF:
    def __init__(self, dim_x:int, dim_y:int, x0, P, M, Q, R, H):
        """
        Kalman Filter
        Use forecast(), forward(), and analyze()
        
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        x0 (numpy.ndarray): Initial x (mean) mx1 
        
        P (numpy.ndarray): P Covariance mxm
        
        M (numpy.ndarray): M Model Matrix mxm 
        
        Q (numpy.ndarray): Q Covariance of the model error mxm
        
        R (numpy.ndarray): R Covariance of the observation error jxj
        
        H (numpy.ndarray): H Observation Matrix jxm
        """
        if ((dim_x,1) != x0.shape):
            raise ValueError("Wrong dimension (X0)")
        self.filterType = "KF"
        
        self.dim_x = dim_x  # dimension of x (m)
        self.dim_y = dim_y  # dimension of Observation Y (j)
        self.x = x0         # Initial X (mean) mx1 
        self.P = P          # P (Covariance) mxm
        self.M = M          # M (Model Matrix) mxm 
        self.Q = Q          # Q (Covariance of the model error) mxm
        self.R = R          # R (Covariance of the observation error) jxj
        self.H = H          # H (Observation Matrix) jxm
        
        self.Xm = x0        # without KF (process X only with M)
        
        self.X_cStack = x0  # collection of x (mxk, k = number of steps)
        self.Xm_cStack = x0 # collection of Xm (mxk)
        self.RMSDList = []  # root mean squared deviation List
        self.PList = []     # covariance List
        
    def forecast(self):
        """
        Forecast X and P
        Also save Xm to compare with the estimate
        """
        self.x = self.M @ self.x 
        self.P = self.M @ self.P @ self.M.T + self.Q

        self.Xm = self.M @ self.Xm
        
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
    
    def forward(self):
        """Forward X with M"""
        self.forecast()
        self.X_cStack = np.column_stack((self.X_cStack, self.x))
    
    def analyze(self, y):
        """
        Analyze X and P
         
        y (numpy.ndarray): Observation of the true state with observation error
        y = H X_t + mu   where (mu ~ N(0,R))
        """
        if (self.dim_y != y.shape[0]):
            raise ValueError("Wrong dimension (Y)")
        
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R) # Kalman Gain Matrix
        self.x = K @ (y - self.H @ self.x) + self.x # analysis ensemble mean
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P # analysis covariance
        self.PList.append(self.P) # save analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.x))
    
    def get_xk(self, m):
        """
        return row m of X_cStack as a list
        
        m (int): row index
        """
        return list(self.X_cStack[m,:])

    def RMSD(self, X, X_true):
        """
        Return final Root Mean Squared Deviation
        
        X (numpy.ndarray): Column Stacked X
        
        X_true (numpy.ndarray): Column Stacked True States
        """
        return np.sqrt(np.sum((np.sum(X, axis=0) - np.sum(X_true, axis=0))**2)/X.shape[1])
    
    def RMSDSave(self, X, X_true):
        """
        Save current RMSD
        
        X (numpy.ndarray): Column Stacked X
        
        X_true (numpy.ndarray): Column Stacked True States
        """
        X_Length = X.shape[1]
        X_true = X_true[:,np.arange(X_Length)]
        self.RMSDList.append(self.RMSD(X, X_true))
    
    def plot_RMSD(self, filters=[], show=True):
        """ 
        plot evolution of Root Mean Squared Deviation.
        Need to run RMSDSave in the loop.
        
        filters (list, optional): addition filter to compare, Defaults to [].
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        """
        plt.figure()
        plt.plot(self.RMSDList, label=self.filterType)
        if (len(filters) > 0):
            for filter in filters:
                plt.plot(filter.RMSDList, label=filter.filterType)
        plt.legend(loc="best")
        if show:
            plt.show()
        
    def plot_all(self, x_true=np.array([[]]), 
                    has_obs=[], Ys=np.array([[]]), titles=[], plotXm = True, 
                    show=True, rmsd=True, filters=[], cov=False):
        """
        Plot each element (x_i) from X in 2D graph (value vs k)
        
        @param:
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        has_obs (int list, optional): One integer element 'i' represents x_i from X has observations. Defaults to [].
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Must be ascending order. Defaults to np.array([[]]).
        
        titles (string list, optional): Titles of the plots. Must have dim_x titles. Defaults to [].
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of the filters. Defaults to True.
        
        filters (list of Kfs, optional): other filters to compare with. Defaults to [].
        
        cov (boolean, optional): plot covariance matrix. Defaults to False.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        for i in range(self.x.shape[0]):
            plt.figure(figsize=(10,6))
            if (x_true.shape[1] == len(xAxis)):
                plt.plot(xAxis, x_true[i,:][xAxis], 'k', label="True")
            if (i in has_obs):
                plt.plot(xAxis, Ys[has_obs.index(i),:], 'y*', label="Observations")
            if (plotXm):
                plt.plot(xAxis, self.Xm_cStack[i,:][xAxis], 'g', label = "without KF")
            plt.plot(xAxis, self.get_xk(i), 'r--', label = f"{self.filterType} x{i}")
            if (len(filters) > 0):
                for filter in filters:
                    if (filter.filterType == self.filterType):
                        filter.filterType = self.filterType + str(filters.index(filter))
                    plt.plot(xAxis, filter.get_xk(i), '--', label = f"{filter.filterType} x{i}")
            if (len(titles) == self.x.shape[0]):
                plt.title(f"{titles[i]}")
            else:
                plt.title(f"x{i}")
            plt.xlabel("state")
            plt.ylabel("value")
            plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
            if (len(filters) > 0):
                for filter in filters:
                    print(f"{filter.filterType}: ", end = "")
                    print(self.RMSD(filter.X_cStack, x_true))
        if cov:
            self.fig, self.ax = plt.subplots() # Create figure and axis
            self.frame = 0 # Initial frame index
            
            ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03]) # slider
            self.slider = Slider(ax_slider, 'Frame', 0, len(self.PList) - 1, valinit=self.frame, valstep=1)
            
            self.slider.on_changed(self.on_slider_change)
            self.update_COV() 
        if show:
            plt.show()
        
    def plot_some(self, some: list, x_true=np.array([[]]), 
                    has_obs=[], Ys=np.array([[]]), titles=[], plotXm = True, 
                    show=True, rmsd=True, filters=[], cov=False):
        """
        Plot some elements (x_i) from X in 2D graph (value vs k)
        
        @param:  
         
        some (in list): index of the element (x_i) to be plotted 
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        has_obs (int list, optional): One integer element 'i' represents x_i from X has observation at y_i. Defaults to [].
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Must be ascending order. Defaults to np.array([[]]).
        
        titles (string list, optional): Titles of the plots. Must have dim_x titles. Defaults to [].
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of the filters. Defaults to True.
        
        filters (list of Kfs, optional): other filters to compare with. Defaults to [].
        
        cov (boolean, optional): plot covariance matrix. Defaults to False.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        for i in some:
            plt.figure(figsize=(10,6))
            if (x_true.shape[1] == len(xAxis)):
                plt.plot(xAxis, x_true[i,:][xAxis], 'k', label="True")
            if (i in has_obs):
                plt.plot(xAxis, Ys[has_obs.index(i),:], 'y*', label="Observations")
            if (plotXm):
                plt.plot(xAxis, self.Xm_cStack[i,:][xAxis], 'g', label = "without KF")
            plt.plot(xAxis, self.get_xk(i), 'r--', label = f"{self.filterType} x{i}")
            if (len(filters) > 0):
                for filter in filters:
                    if (filter.filterType == self.filterType):
                        filter.filterType = self.filterType + str(filters.index(filter))
                    plt.plot(xAxis, filter.get_xk(i), '--', label = f"{filter.filterType} x{i}")
            if (len(titles) == self.x.shape[0]):
                plt.title(f"{titles[i]}")
            else:
                plt.title(f"x{i}")
            plt.xlabel("state")
            plt.ylabel("value")
            plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
            if (len(filters) > 0):
                for filter in filters:
                    print(f"{filter.filterType}: ", end = "")
                    print(self.RMSD(filter.X_cStack, x_true))
        if cov:
            self.fig, self.ax = plt.subplots() # Create figure and axis
            self.frame = 0 # Initial frame index
            
            ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03]) # slider
            self.slider = Slider(ax_slider, 'Frame', 0, len(self.PList) - 1, valinit=self.frame, valstep=1)
            
            self.slider.on_changed(self.on_slider_change)
            self.update_COV()
        if show:
            plt.show()
    
    def update_COV(self):
        """Update covariance matrix plot"""
        self.ax.clear()
        self.ax.imshow(self.PList[self.frame], cmap='Greys', vmin=-1, vmax=1)
        self.ax.set_title('Covariance Matrix')
           
    def on_slider_change(self, val):
        """slider update function"""
        self.frame = int(val)
        self.update_COV()

class EKF(KF):            
    """Introduction to the principles and methods of data assimilation in the geosciences.pdf"""
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, Q, R, H, M_j, H_j):
        """
        Extended Kalman Filter
        Use eForecast(), eForward(), and eAnalyze()
        *** This KF is not optimal filter. 
            Often diverges from the true state,
            which cause overflow error.
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. Must return (dim_y,1,1) array
        
        M_j (function): M_j (A function return Jacobian of M at xi) mxm
        
        H_j (function): H_j (A function return Jacobian of H at xi) jxm
        """
        KF.__init__(self, dim_x, dim_y, X0, P, M, Q, R, H)
        self.filterType = "EKF"
        self.M_j = M_j
        self.H_j = H_j
    
    def eForecast(self):
        """Forecast EKF"""
        self.x = self.M(self.x) # forecast
        
        Mk = self.M_j(self.x) # jacobian of M at current state x
        self.P = Mk @ self.P @ Mk.T + self.Q # forecast covariance
        
        self.Xm = self.M(self.Xm)
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
        
    def eForward(self):
        self.eForecast()
        self.X_cStack = np.column_stack((self.X_cStack, self.x))
    
    def eAnalyze(self, y):
        """
        Analyze EKF
        
        y (numpy.ndarray): Observation of the true state with observation error
        """
        Hk = self.H_j(self.x) 
        K = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + self.R) # Kalman Gain Matrix
        self.x = K @ (y - self.H(self.x)) + self.x # analysis mean
        self.P = (np.eye(self.dim_x) - K @ Hk) @ self.P # analysis covariance
        self.PList.append(self.P) # save analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.x))
        
class EnKF(KF):
    """Introduction to the principles and methods of data assimilation 
       in the geosciences.pdf(Bocquet)"""
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, R, H, H_j, n=10):
        """
        Stochastic Ensemble Kalman Filter
        Use enForecast(), enForward(), and enAnalyze()
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        n (int): Number of ensemble members
        """
        KF.__init__(self, dim_x, dim_y, X0, P, M, None, R, H)
        self.filterType = "StochasticEnKF"
        self.n = n
        self.H_j = H_j 
        self.sampling()
        
    def sampling(self):
        """Create n ensemble members with initial P"""
        self.enX = []
        for i in range(self.n):
            s = np.random.multivariate_normal([0]*self.dim_x, self.P).reshape((self.dim_x,1)) # background error
            xi = self.M(self.x) + s
            self.enX.append(xi)
    
    def enCov(self):
        """Calculate error covariance P in Ensemble"""
        X = [0 for i in range(self.n)] # ensemble perturbations
        for i in range(self.n):
            X[i] = self.enX[i] - self.x
        X = np.column_stack(X)
        
        self.P = X @ X.T / (self.n - 1) # analysis covariance
        self.PList.append(self.P) # save analysis covariance
        
    def enForecast(self):
        """Forecast EnKF"""
        for i in range(self.n): # forecast each enX[i]
            self.enX[i] = self.M(self.enX[i])
            
        self.x = np.mean(self.enX, axis=0) # forecast ensemble mean 
        
        self.enCov() # forecast covariance
        
        self.Xm = self.M(self.Xm)
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
        
    def enForward(self):
        self.enForecast()
        self.X_cStack = np.column_stack((self.X_cStack, self.x))
        
    def perturbed_obs(self, y):
        """Create n perturbed observations and its Covariance"""
        perturbedY = []
        perturbs = []
        for i in range(self.n):
            perturb = np.random.multivariate_normal([0]*self.dim_y, self.R).reshape((self.dim_y,1))
            perturbs.append(perturb)
            perturbedY.append(y + perturb) 
        
        # make bias = sum(perturb) = 0 to avoid bias
        perturbs_sum = np.sum(perturbs, axis=0)
        bias = perturbs_sum / (len(perturbs))
        for i in range(self.n): 
            perturbs[i] -= bias

        # Covariance
        Ru = np.zeros((self.dim_y, self.dim_y))
        for i in range(self.n):
            Ru += perturbs[i] @ perturbs[i].T
        Ru /= (self.n-1)
        
        return perturbedY, Ru
    
    def enAnalyze(self, y):
        """
        Analyze EnKF
        
        y (numpy.ndarray): Observation of the true state with observation error
        """ 
        if (self.dim_y != y.shape[0]):
            raise ValueError("Wrong dimension (Y)")
        
        perturbedY, Ru = self.perturbed_obs(y) # empirical error covariance
        Hk = self.H_j(self.x)

        K = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + Ru) # Kalman Gain Matrix
        
        for i in range(self.n): # analyze xi
            self.enX[i] = ((K @ (perturbedY[i] - self.H(self.enX[i]))) + self.enX[i])
            
        self.x = np.mean(self.enX, axis=0) # analysis ensemble mean
        
        self.enCov() # analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.x))

class ETKF(EnKF):
    """
    Local Ensemble Transform Kalman Filter 
    Local Ensemble Transform Kalman Filter: An Efficient Scheme for Assimilating Atmospheric Data
    (Harlim and Hunt, 2006)
    """
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, R, H, n=10, rho=1):
        """
        Ensemble Transform Kalman Filter
        Use etForecast(), etForward(), and etAnalyze()d
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        n (int): Number of ensemble members
        
        rho (float): multiplicative inflation factor. must be greater or equal than 1
        """
        EnKF.__init__(self, dim_x, dim_y, X0, P, M, R, H, H, n=n)
        self.filterType = "ETKF"
        self.R_inv = np.linalg.inv(self.R)
        if (rho < 1):
            raise ValueError("inflation factor must be greater than one")
        self.rho = rho
        self.ones = np.ones((1,self.n)) #1xn
        
    def etForecast(self):
        """Forecast ETKF"""
        self.enForecast()

    def etForward(self):
        self.enForward()

    def etAnalyze(self, y):
        """
        Analyze ETKF
        
        y (numpy.ndarray): Observation of the true state with observation error
        """
        X = np.column_stack(self.enX) # stacked ensemble members mxn
        
        Y = [0 for i in range(self.n)] # forecasted perturbations in the observation space
        for i in range(self.n): # project X_a onto observation space
            Y[i] = self.H(self.enX[i])
        Y_mean = np.mean(Y, axis=0)
        Y = np.column_stack((Y)) #jxn
        Y = Y - Y_mean @ self.ones # subtract the mean on each columns of Y

        X = X - self.x @ self.ones # subtract the mean on each columns of X
        
        C = Y.T @ self.R_inv # for computational efficiency nxj
        
        P_tilde = (self.n - 1) * np.eye(self.n) / self.rho + C @ Y 
        P_tilde = np.linalg.inv(P_tilde) #nxn
        
        w = P_tilde @ C @ (y - Y_mean) # weight vector nx1
    
        W = (self.n - 1) * P_tilde 
        W = sp.linalg.fractional_matrix_power(W, 0.5) #nxn
        # sometimes converted to complex numbers.
        # it comes from computational rounding in python
        # imaginary parts are all 0, so neglectable
        W = W.real # transform matrix 
        
        W = W + w @ self.ones #nxn
        
        X = X @ W #mxn
        
        for i in range(self.n): # analysis ensembles
            self.enX[i] = self.x + X[:,i].reshape((self.dim_x, 1)) 
        
        self.x = np.mean(self.enX, axis=0) # analysis ensemble mean
        self.enCov() # analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.x))

class LETKF(ETKF):
    """Efficient data assimilation for spatiotemporal chaos: A local ensemble
       transform Kalman filter (Hunt et al)"""
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, R, H, L, n=10, rho=1):
        """
        Local Ensemble Transform Kalman Filter
        Use leForecast(), leForward(), and leAnalyze()
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        L (function): L(m) (Localization operator. m=index of the grid point.
            must return a list of N indices of the observations)
        
        n (int): Number of ensemble members
        
        rho (float): multiplicative inflation factor. must be greater or equal than 1
        """
        ETKF.__init__(self, dim_x, dim_y, X0, P, M, R, H, n=n, rho=rho)
        self.filterType = "LETKF"
        self.L = L
        
    def leForecast(self):
        self.enForecast()
    
    def leForward(self):
        self.enForward()
    
    def leAnalyze(self, y):
        """
        Analyze LETKF
        
        y (numpy.ndarray): Observation of the true state with observation error
        """
        
        X = np.column_stack(self.enX) # stacked ensemble members
        
        Y = [0 for i in range(self.n)] # forecasted perturbations in the observation space
        for i in range(self.n): # project X_a onto observation space
            Y[i] = self.H(self.enX[i])
        Y_mean = np.mean(Y, axis=0)
        Y = np.column_stack((Y)) 
        Y = Y - Y_mean @ self.ones # subtract the mean on each columns of Y
        
        X = X - self.x @ self.ones # subtract the mean on each columns of X
        
        Xa = [0 for i in range(self.dim_x)]
        for m in range(self.dim_x):
            b = self.L(m) # indices of localized lows of Y
            Y_local = Y[b,:] #Nxn
            
            X_local = X[m,:] #Nxn
            N = len(b)
            
            R_local = np.diag([self.R_inv[z,z] for z in b]) #NxN 
                # Assume R is diagonal (each observation is independent from others)
            C = Y_local.T @ R_local #nxN
            
            P_tilde = self.n * np.eye((self.n)) / self.rho + C @ Y_local #nxn
            P_tilde = np.linalg.inv(P_tilde) #nxn
            
            W = (self.n-1) * P_tilde 
            W = sp.linalg.fractional_matrix_power(W, 0.5) #nxn
            # sometimes converted to complex numbers.
            # it comes from computational rounding in python
            # imaginary parts are all 0, so neglectable
            W = W.real
            
            w = P_tilde @ C @ (y[b,:].reshape((N,1)) - Y_mean[b,:].reshape(N,1)) # weight vector nx1
            
            W = W + w @ self.ones #nxn
            
            X_local = X_local @ W #Nxn
            
            X_local = X_local + self.x[m,:].reshape((1, 1)) @ self.ones #Nxn
            
            Xa[m] = X_local # save analyzed local grid point
            
        Xa = np.row_stack(Xa) #mxn
        
        for i in range(self.n):
            self.enX[i] = Xa[:,i].reshape((self.dim_x, 1)) #mx1
        
        self.x = np.mean(self.enX, axis=0) # analysis ensemble mean
        self.enCov() # analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.x))

```
</details>

### Requirements
numpy
    
    pip install numpy

matplotlib 
    
    pip install matplotlib

scipy 
    
    pip install scipy

tqdm 
    
    pip install tqdmd


