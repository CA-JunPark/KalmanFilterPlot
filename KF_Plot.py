import numpy as np
import matplotlib.pyplot as plt

class KF:
    def __init__(self, dim_x:int, dim_y:int, X0, P=None, M=None, Q=None, R=None, H=None):
        """
        Kalman Filter
        Use forecast(), forward(), and analyze()
        
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (numpy.ndarray): M (Model Matrix) mxm 
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (numpy.ndarray): H (Observation Matrix) jxm
        """
        if ((dim_x,1) != X0.shape):
            raise ValueError("Wrong dimension (X0)")
        self.filterType = "KF"
        
        self.dim_x = dim_x  # dimension of X (m)
        self.dim_y = dim_y  # dimension of Observation Y (j)
        self.X = X0         # Initial X (mean) mx1 
        self.P = np.eye(dim_x) if P is None else P  # P (Covariance) mxm
        self.M = np.eye(dim_x) if M is None else M  # M (Model Matrix) mxm 
        self.Q = np.eye(dim_x) if Q is None else Q  # Q (Covariance of the model error) mxm
        self.R = np.eye(dim_x) if R is None else R  # R (Covariance of the observation error) jxj
        self.H = np.array([[1]*dim_y]) if H is None else H  # H (Observation Matrix) jxm
        
        self.Xm = X0        # without KF (process X only with M)
        
        self.X_cStack = X0  # collection of X (mxk, k = # of iterations)
        self.Xm_cStack = X0 # collection of Xm (mxk)
        self.RMSDList = []
        
    def forecast(self):
        """
        Forecast X and P
        Also save Xm to compare with the estimate
        """
        self.X = self.M @ self.X 
        self.P = self.M @ self.P @ self.M.T + self.Q

        self.Xm = self.M @ self.Xm
        
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
    
    def forward(self):
        """Forward X with M"""
        self.X = self.M @ self.X
        
        self.Xm = self.M @ self.Xm
        
        self.X_cStack = np.column_stack((self.X_cStack, self.X))
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
    
    def analyze(self, Y):
        """
        Analyze X and P
         
        Y (numpy.ndarray): Observation of the true state with observation error
        Y = H X_t + mu   where (mu ~ N(0,R))
        """
        if (self.dim_y != Y.shape[0]):
            raise ValueError("Wrong dimension (Y)")
        
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R) # Kalman Gain Matrix
        self.X = K @ (Y - self.H @ self.X) + self.X
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
        self.X_cStack = np.column_stack((self.X_cStack, self.X))
    
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
        return np.sqrt(np.sum((np.sum(X, axis=0) - np.sum(X_true, axis=0))**2)/10)
    
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
                 has_obs=[], Ys=np.array([[]]), titles=[], plotXm=True, 
                 show=True, rmsd=True):
        """
        Plot each element (x_i) from X in 2D graph (value vs k)
        d
        @param:
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        has_obs (int list, optional): One integer element 'i' represents x_i from X has observations. Defaults to [].
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Must be ascending order Defaults to np.array([[]]).
        
        titles (string list, optional): Titles of the plots. Must have dim_x titles. Defaults to [].
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of this filter. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        for i in range(self.X.shape[0]):
            plt.figure(figsize=(10,6))
            if (x_true.shape[1] == len(xAxis)):
                plt.plot(xAxis, x_true[i,:][xAxis], 'k', label="True")
            if (i in has_obs):
                plt.plot(xAxis, Ys[has_obs.index(i),:], 'y*', label="Observations")
            plt.plot(xAxis, self.get_xk(i), 'r--', label = f"estimated x{i}")
            if (plotXm):
                plt.plot(xAxis, self.Xm_cStack[i,:][xAxis], 'g', label = "without KF")
            if (len(titles) > 0):
                plt.title(f"{titles[i]} x{i}")
            else:
                plt.title(f"x{i}")
            plt.xlabel("state")
            plt.ylabel("value")
            plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
        if show:
            plt.show()
    
    def plot_one(self, m:int, x_true=np.array([[]]), 
                 ym=None, Ys=np.array([[]]), title=None, plotXm=True, 
                 show=True, rmsd=True):
        """
        Plot one element (x_m) from X in 2D graph (value vs k)
        
        @param:
    
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        ym (int, optional): i-th row elements in Ys is corresponding Observation of x_m. Defaults to None.
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Defaults to np.array([[]]).
        
        title (string, optional): Title of the plot. Defaults to None
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of this filter. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        plt.figure(figsize=(10,6))
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym != None):
            plt.plot(xAxis, Ys[ym,:][xAxis], 'y*', label="Observations")
        plt.plot(xAxis, self.get_xk(m), 'r--', label = f"estimated x{m}")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m,:][xAxis], 'g', label = "without KF")
        plotTitle1 = f"x{m}" if title is None else title
        plt.title(plotTitle1)
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
        if show:
            plt.show()
    
    def plot_two(self, m1:int, m2:int, x_true=np.array([[]]), 
                 ym1=None, ym2=None, Ys=np.array([[]]), title1=None, title2=None, plotXm=True, 
                 show=True, rmsd=True):
        """
        Plot two element (x_m1 and x_m2) from X in 2D graph (value vs k)
        
        @param:
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        ym1 (int, optional): ym1-th row elements in Ys is corresponding Observation of x_m1. Defaults to None.
        
        ym2 (int, optional): ym2-th row elements in Ys is corresponding Observation of x_m2. Defaults to None.
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Defaults to np.array([[]]).
        
        title1 & title2 (string, optional): Titles of the plots. Defaults to None.
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of this filter. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m1,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym1 != None):
            plt.plot(xAxis, Ys[ym1,:], 'y*', label="Observations")
        plt.plot(xAxis, self.get_xk(m1), 'r--', label = f"estimated x{m1}")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m1,:][xAxis], 'g', label = "without KF")
        plotTitle = f"x{m1}" if title1 is None else title1
        plt.title(plotTitle)
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        
        plt.subplot(1,2,2)
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m2,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym2 != None):
            plt.plot(xAxis, Ys[ym2,:], 'y*', label="Observations")
        plt.plot(xAxis, self.get_xk(m2), 'r--', label = f"estimated x{m2}")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m2,:][xAxis], 'g', label = "without KF")
        plotTitle = f"x{m2}" if title2 is None else title2
        plt.title(f"x{m2}")
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
        if show:
            plt.show()
        
    def compare_all(self, filter, x_true=np.array([[]]), 
                    has_obs=[], Ys=np.array([[]]), titles=[], plotXm = True, 
                    show=True, rmsd=True):
        """
        Plot each element (x_i) from X in 2D graph (value vs k)
        
        @param:
        
        filter(KF): another filter to compare with
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        has_obs (int list, optional): One integer element 'i' represents x_i from X has observations. Defaults to [].
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Must be ascending order. Defaults to np.array([[]]).
        
        titles (string list, optional): Titles of the plots. Must have dim_x titles. Defaults to [].
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of the filters. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        for i in range(self.X.shape[0]):
            plt.figure(figsize=(10,6))
            if (x_true.shape[1] == len(xAxis)):
                plt.plot(xAxis, x_true[i,:][xAxis], 'k', label="True")
            if (i in has_obs):
                plt.plot(xAxis, Ys[has_obs.index(i),:], 'y*', label="Observations")
            if (plotXm):
                plt.plot(xAxis, self.Xm_cStack[i,:][xAxis], 'g', label = "without KF")
            plt.plot(xAxis, self.get_xk(i), 'r--', label = f"{self.filterType} x{i}")
            plt.plot(xAxis, filter.get_xk(i), 'b--', label = f"{filter.filterType} x{i}")
            if (len(titles) == self.X.shape[0]):
                plt.title(f"{titles[i]}")
            else:
                plt.title(f"x{i}")
            plt.xlabel("state")
            plt.ylabel("value")
            plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
            print(f"{filter.filterType}: ", end = "")
            print(self.RMSD(filter.X_cStack, x_true))
        if show:
            plt.show()
        
    def compare_one(self, filter, m:int, x_true=np.array([[]]), 
                    ym=None, Ys=np.array([[]]), title=None, plotXm=True, 
                    show=True, rmsd=True):
        """
        Plot one element (x_m) from X in 2D graph (value vs k)
        
        @param:
        
        filter(KF): another filter to compare with
    
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        ym (int, optional): i-th row elements in Ys is corresponding Observation of x_m. Defaults to None.
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Defaults to np.array([[]]).
        
        title (string, optional): Title of the plot. Defaults to None
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of the filters. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        plt.figure(figsize=(10,6))
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym != None):
            plt.plot(xAxis, Ys[ym,:][xAxis], 'y*', label="Observations")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m,:][xAxis], 'g', label = "without KF")
        plt.plot(xAxis, self.get_xk(m), 'r--', label = f"{self.filterType}. x{m}")
        plt.plot(xAxis, filter.get_xk(m), 'b--', label = f"{filter.filterType} x{m}")
        plotTitle1 = f"x{m}" if title is None else title
        plt.title(plotTitle1)
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
            print(f"{filter.filterType}:", end = "")
            print(self.RMSD(filter.X_cStack, x_true))
        if show:
            plt.show()
    
    def compare_two(self, filter, m1:int, m2:int, x_true=np.array([[]]), 
                    ym1=None, ym2=None, Ys=np.array([[]]), title1=None, title2=None, 
                    plotXm=True, show=True, rmsd=True):
        """
        Plot two element (x_m1 and x_m2) from X in 2D graph (value vs k)
        
        @param:
        
        filter(KF): another filter to compare with
        
        x_true (numpy.ndarray, optional): Column stacked true states. Defaults to np.array([[]]).
        
        ym1 (int, optional): ym1-th row elements in Ys is corresponding Observation of x_m1. Defaults to None.
        
        ym2 (int, optional): ym2-th row elements in Ys is corresponding Observation of x_m2. Defaults to None.
        
        Ys (numpy.ndarray, optional): ColumnStacked observations. Defaults to np.array([[]]).
        
        title1 & title2 (string, optional): Titles of the plots. Defaults to None.
        
        plotXm (boolean, optional): plot 'without KF' if True. Defaults to True.
        
        show (boolean, optional): execute plt.show() if True. Defaults to True.
        
        rmsd (boolean, optional): print RMSD of the filters. Defaults to True.
        """
        xAxis = np.arange(self.Xm_cStack.shape[1])
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m1,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym1 != None):
            plt.plot(xAxis, Ys[ym1,:], 'y*', label="Observations")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m1,:][xAxis], 'g', label = "without KF")
        plt.plot(xAxis, self.get_xk(m1), 'r--', label = f"{self.filterType} x{m1}")
        plt.plot(xAxis, filter.get_xk(m1), 'b--', label = f"{filter.filterType} x{m1}")
        plotTitle = f"x{m1}" if title1 is None else title1
        plt.title(plotTitle)
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        
        plt.subplot(1,2,2)
        if (x_true.shape[1] == len(xAxis)):
            plt.plot(xAxis, x_true[m2,:][xAxis], 'k', label="True")
        if (Ys.shape[1] == len(xAxis) and ym2 != None):
            plt.plot(xAxis, Ys[ym2,:], 'y*', label="Observations")
        if (plotXm):
            plt.plot(xAxis, self.Xm_cStack[m2,:][xAxis], 'g', label = "without KF")
        plt.plot(xAxis, self.get_xk(m2), 'r--', label = f"{self.filterType} x{m2}")
        plt.plot(xAxis, filter.get_xk(m2), 'b--', label = f"{filter.filterType} x{m2}")
        plotTitle = f"x{m2}" if title2 is None else title2
        plt.title(f"x{m2}")
        plt.xlabel("state")
        plt.ylabel("value")
        plt.legend(loc='best')
        
        if rmsd:
            print(f"{self.filterType}: ", end = "")
            print(self.RMSD(self.X_cStack, x_true))
            print(f"{filter.filterType}:", end = "")
            print(self.RMSD(filter.X_cStack, x_true))
        if show:
            plt.show()

class StochasticEnKF(KF):
    """Introduction to the principles and methods of data assimilation in the geosciences.pdf"""
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, Q, R, H, H_j, n=10):
        """
        Stochastic Ensemble Kalman Filter
        Use enForecast(), enForward(), and enAnalyze()
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator must return (m,j)) 
        
        H_j (function): H_j (Jacobian of H at X must return (m,j))
        
        n (int): Number of ensemble members
        """
        KF.__init__(self, dim_x, dim_y, X0, P, M, Q, R, H)
        self.filterType = "StochasticEnKF"
        self.n = n
        self.H_j = H_j
        self.sampling()
        
    def sampling(self):
        """Create n ensemble members with initial P"""
        self.enX = []
        for i in range(self.n):
            e = np.random.multivariate_normal([0]*self.dim_x, self.P).reshape((self.dim_x,1)) # background error
            xi = self.M(self.X) + e
            self.enX.append(xi)
        
    def enForecast(self):
        """Forecast EnKF"""
        for i in range(self.n): # forecast each enX[i]
            self.enX[i] = self.M(self.enX[i])
            
        self.X = np.mean(self.enX, axis=0) # forecast ensemble mean 
        
        p = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.n):
            X_diff = self.enX[i] - self.X
            p += X_diff @ X_diff.T

        self.P = p / (self.n-1) # forecast ensemble covariance
        
        self.Xm = self.M(self.Xm)
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
        
    def enForward(self):
        for i in range(self.n): # forward each enX[i]
            self.enX[i] = self.M(self.enX[i])
            
        self.X = np.mean(self.enX, axis=0) # forward ensemble mean 
        
        self.Xm = self.M(self.Xm)
        
        self.X_cStack = np.column_stack((self.X_cStack, self.X))
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
        
    def empiricalObsError(self, Y):
        """Create n perturbed observations and its Covariance"""
        perturbedY = []
        perturbs = []
        for i in range(self.n):
            perturb = np.random.multivariate_normal([0]*self.dim_y, self.R).reshape((self.dim_y,1))
            perturbs.append(perturb)
            perturbedY.append(Y + perturb) 
        
        # make bias = sum(perturb) = 0 tp avoid bias
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
    
    def enAnalyze(self, Y):
        """Analyze EnKF"""
        if (self.dim_y != Y.shape[0]):
            raise ValueError("Wrong dimension (Y)")
        
        perturbedY, Ru = self.empiricalObsError(Y) # empirical error covariance
        Hk = self.H_j(self.X)
        K = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + Ru) # Kalman Gain Matrix
        
        for i in range(self.n): # analyze xi
            self.enX[i] = ((K @ (perturbedY[i] - self.H(self.enX[i]))) + self.enX[i])
        
        self.X = np.mean(self.enX, axis=0) # analysis ensemble mean
        
        p = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.n):
            X_diff = self.enX[i] - self.X
            p += X_diff @ X_diff.T

        self.P = p / (self.n-1) # analysis ensemble covariance

        self.X_cStack = np.column_stack((self.X_cStack, self.X))

# TODO
class DeterministicEnKF(StochasticEnKF):
    """ensemble transform kalman filter """
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, Q, R, H, H_j, n=10):
        StochasticEnKF.__init__(self, dim_x, dim_y, X0, P, M, Q, R, H, H_j, n=10)
        self.filterType = "DeterministicEnKF"

# TODO
# local ensemble transform kalman filter

class EKF(KF):
    """Introduction to the principles and methods of data assimilation in the geosciences.pdf"""
    def __init__(self, dim_x:int, dim_y:int, X0, P, M, Q, R, H, M_j, H_j):
        """
        Extended Kalman Filter
        Use eForecast(), eForward(), and eAnalyze()
        
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
        self.X = self.M(self.X) 
        
        Mk = self.M_j(self.X)
        self.P = Mk @ self.P @ Mk.T + self.Q
        
        self.Xm = self.M(self.Xm)
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
        
    def eForward(self):
        self.X = self.M(self.X) 
        self.X_cStack = np.column_stack((self.X_cStack, self.X))
        
        self.Xm = self.M(self.Xm)
        self.Xm_cStack = np.column_stack((self.Xm_cStack, self.Xm))
    
    def eAnalyze(self, Y):
        """Analyze EKF"""
        Hk = self.H_j(self.X) 
        K = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + self.R) # Kalman Gain Matrix
        self.X = K @ (Y - self.H(self.X)) + self.X # analysis mean
        self.P = (np.eye(self.dim_x) - K @ Hk) @ self.P # analysis covariance
        
        self.X_cStack = np.column_stack((self.X_cStack, self.X))