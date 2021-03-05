import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numba

class QADF:
    r"""Quantile unit root test

    Based on the Koenker, R., Xiao, Z., 2004. methodology.

    Parameters
    ----------
    y       -  Nx1 matrix, data,

    model   -   nc = No constant, no trend.
                c  = Constant (Default in Koenker & Xiao (2004)
                ct = Constant and trend,

    pmax    -  Maximum number of lags for Dy

    ic      -  Information Criterion:
                "AIC     = Akaike
                "BIC"    = Schwarz
                "t-stat" = t-stat significance

    tau     -  quantile (0.1,...,1)

    Returns
    -------
    rho(quantile) - the estimated rho parameter for a given quantile
    rho (OLS)     - the estimated rho parameter for usual OLS estimation
    delta2        - see equation (10) at page 778
    tn(quantile)  - quantile unit root statistic (t-ratio for a given quantile)
    cv            - 1%, 5%, 10% critical values for the estimated delta2

    """

    def __init__(self, model='c', pmax=5, ic='AIC'):
        self.model = model
        self.pmax = pmax
        self.ic = ic
        self.results = None

    def fit(self, y, tau):
        self.y = np.array(y)
        qrADF = self.QRADF(y, tau, self.crit_QRadf,
                           self.bandwidth, self.model, self.pmax, self.ic)
        self.tau = qrADF['tau']
        self.rho_tau = qrADF['rho_tau']
        self.rho_ols = qrADF['rho_ols']
        self.delta2 = qrADF['delta2']
        self.QURadf = qrADF['QURadf']
        self.cv = qrADF['cv']
        self.p = qrADF['p']
        self.results = {
            'quantile': round(self.tau, 2),
            'Lags': self.p,
            'rho(quantile)': round(self.rho_tau, 3),
            'rho (OLS)': round(self.rho_ols, 3),
            'delta^2': round(self.delta2, 3),
            'ADF(quantile)': round(self.QURadf, 3),
            'Critical Values': round(self.cv, 3)
        }
        return self.results

    local = {
        'y': numba.float32[:],
        'model': numba.types.unicode_type,
        'pmax': numba.types.int32,
        'ic': numba.types.unicode_type,
        'tau': numba.types.float32,
        'adfuller': numba.types.pyfunc_type
    }

    @staticmethod
    @numba.jit(locals=local, forceobj=True, parallel=True)
    def QRADF(y, tau, crit_QRadf, bandwidth, model='c', pmax=5, ic='AIC'):
        r"""tau     -  quantile (0.1,...,1)
        """
        n = len(y)
        y1 = y[:-1]  # creating lags
        dy = np.diff(y)  # first difference
        x = y1
        resultsADF = adfuller(y, maxlag=pmax,
                              regression=model, autolag=ic)
        ADFt, p, cvADF = resultsADF[0], resultsADF[2], resultsADF[4]

        # Defining series in case of possible legs
        if p > 0:
            dyl = np.zeros((n, p))
            for j in range(1, p+1):
                dyl[j+1:, j-1] = dy[:-j]

            y = np.array(y[p+1:])
            y1 = np.array(y1[p:])
            dy = np.array(dy[p:])
            dyl = np.array(dyl[p+1:])
            x = np.array(np.c_[y1, dyl])
        # If the ic decides not to include lags
        elif p == 0:
            y = np.array(y[p+1:])
            y1 = np.array(y1[p:])
            dy = np.array(dy[p:])
            x = np.array(x[p:])

        # Running the quantile regression
        n = len(y)  # new
        qOut1 = sm.QuantReg(y, sm.add_constant(x)).fit(q=tau)

        # calculating rho
        rho_tau = qOut1.params[1]
        rho_ols = sm.OLS(y, sm.add_constant(x)).fit().params[1]

        # calculating delta2 using covariance
        ind = qOut1.resid < 0
        phi = tau - ind
        w = dy
        cov = np.cov(phi, w)[0, 1]
        delta2 = (cov / (np.std(w, ddof=1) * np.sqrt(tau * (1 - tau)))) ** 2

        # calculating critical values associate with our delta2
        crv = crit_QRadf(delta2, model)

        # Calculating quantile bandwidth
        h = bandwidth(tau, n, True)
        if tau <= 0.5 and h > tau:
            h = bandwidth(tau, n, False)
            if h > tau:
                h = tau/1.5

        if tau > 0.5 and h > 1-tau:
            h = bandwidth(tau, n, False)
            if h > (1 - tau):
                h = (1-tau)/1.5

        # Defining some inputs
        x1 = y1
        if p > 0:
            x1 = np.c_[y1, dyl]

        # Running the other 2 QuantRegs
        qOut2 = sm.QuantReg(y, sm.add_constant(x1)).fit(q=tau+h)
        qOut3 = sm.QuantReg(y, sm.add_constant(x1)).fit(q=tau-h)

        # Extracting betas
        rq1 = qOut2.params
        rq2 = qOut3.params

        # Setting inputs for the unit root test
        z = sm.add_constant(x1)
        mz = z.mean(axis=0)

        q1 = np.matmul(mz, rq1)
        q2 = np.matmul(mz, rq2)
        fz = 2 * h/(q1 - q2)
        if fz < 0:
            fz = 0.01

        xx = np.ones((len(x), 1))
        if p > 0:
            xx = sm.add_constant(dyl)

        # Constructing a NxN matrix
        PX = np.eye(len(xx)) - \
            xx.dot(np.linalg.inv(np.dot(xx.T, xx))).dot(xx.T)
        fzCrt = fz/np.sqrt(tau * (1 - tau))
        eqPX = np.sqrt(y1.T.dot(PX).dot(y1))

        # QADF statistic
        QURadf = fzCrt * eqPX * (rho_tau - 1)
        cv = crit_QRadf(delta2, model)

        return {
            'tau': tau,
            'rho_tau': rho_tau,
            'rho_ols': rho_ols,
            'delta2': delta2,
            'QURadf': QURadf,
            'cv': cv,
            'p': p
        }

    @staticmethod
    @numba.jit(forceobj=True, parallel=True)
    def bandwidth(tau, n, is_hs, alpha=0.05):
        x0 = norm.ppf(tau)  # inverse of cdf
        f0 = norm.pdf(x0)  # Probability density function

        if is_hs:
            a = n**(-1/3)
            b = norm.ppf(1 - alpha/2)**(2/3)
            c = ((1.5 * f0**2)/(2 * x0**2+1))**(1/3)

            h = a * b * c
        else:
            h = n**(-0.2) * ((4.5 * f0**4)/(2 * x0**2 + 1)**2)**0.2

        return h

    @staticmethod
    @numba.jit(forceobj=True, parallel=True)
    def crit_QRadf(r2, model):
        ncCV = [[-2.4611512, -1.783209, -1.4189957],
                [-2.494341, -1.8184897, -1.4589747],
                [-2.5152783, -1.8516957, -1.5071775],
                [-2.5509773, -1.895772, -1.5323511],
                [-2.5520784, -1.8949965, -1.541883],
                [-2.5490848, -1.8981677, -1.5625462],
                [-2.5547456, -1.934318, -1.5889045],
                [-2.5761273, -1.9387996, -1.602021],
                [-2.5511921, -1.9328373, -1.612821],
                [-2.5658, -1.9393, -1.6156]]

        cCV = [[-2.7844267, -2.115829, -1.7525193],
               [-2.9138762, -2.2790427, -1.9172046],
               [-3.0628184, -2.3994711, -2.057307],
               [-3.1376157, -2.5070473, -2.168052],
               [-3.191466, -2.5841611, -2.2520173],
               [-3.2437157, -2.639956, -2.316327],
               [-3.2951006, -2.7180169, -2.408564],
               [-3.3627161, -2.7536756, -2.4577709],
               [-3.3896556, -2.8074982, -2.5037759],
               [-3.4336, -2.8621, -2.5671]]

        ctCV = [[-2.9657928, -2.3081543, -1.9519926],
                [-3.1929596, -2.5482619, -2.1991651],
                [-3.3727717, -2.7283918, -2.3806008],
                [-3.4904849, -2.8669056, -2.5315918],
                [-3.6003166, -2.9853079, -2.6672416],
                [-3.6819803, -3.095476, -2.7815263],
                [-3.7551759, -3.178355, -2.8728146],
                [-3.8348596, -3.2674954, -2.973555],
                [-3.8800989, -3.3316415, -3.0364171],
                [-3.9638, -3.4126, -3.1279]]

        # Selecting the critical values set based on model type
        cvs = {'nc': ncCV, 'c': cCV, 'ct': ctCV}
        cv = cvs[model]

        delta2 = pd.Series(np.arange(0.1, 1.1, 0.1), name='delta2')
        crt = pd.DataFrame(cv, index=delta2, columns=['1%', '5%', '10%'])

        if r2 < 0.1:
            ct = crt.iloc[0, :]
        else:
            r210 = r2 * 10
            if (r210) >= 10:
                ct = crt.iloc[9, :]
            else:
                #  Main logic goes here
                r2a = int(np.floor(r210))
                r2b = int(np.ceil(r210))
                wa = r2b - r210
                ct = wa * crt.iloc[(r2a-1), :] + (1 - wa) * \
                    crt.iloc[(r2b-1), :]
        return ct

    def __repr__(self):
        if self.results != None:
            rmv_chars = {'}': '', '{': '', "'": ''}
            rmv_out = str(self.results).translate(str.maketrans(rmv_chars))
            out = rmv_out.replace('Values: ', 'Values:\n').replace(
                ',', '\n').replace('\n ', '\n')
            return out
        return object.__repr__(self)

    def summary(self):
        print(self.__repr__())
