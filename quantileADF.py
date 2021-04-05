import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import add_constant, OLS, QuantReg
from statsmodels.tsa.stattools import adfuller, lagmat
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
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

    # Fitting data for different quantiles
    def fitForQuantiles(self, y, quantiles):
        results = []
        for tau in quantiles:
            res = self.fit(y, tau)
            results.append(res)
        df = pd.DataFrame(results)

        # Adding the QKS statistic
        df['QKS'] = max(map(abs, df['tₙ(τ)']))
        df['name'] = y.name
        df.set_index('quantile', inplace=True)
        return df

    def fit(self, y, tau):
        self.y = np.array(y)
        qrADF = self.QRADF(y, tau, self.crit_QRadf,
                           self.bandwidth, self.model, self.pmax, self.ic)

        # Exposing variables
        self.tau = qrADF['τ']
        self.p = qrADF['p']
        self.alpha_tau = qrADF['α\u2080(τ)']
        self.rho_tau = qrADF['ρ\u2081(τ)']
        self.rho_ols = qrADF['ρ\u2081(OLS)']
        self.delta2 = qrADF['δ\u00B2']
        self.QURadf = qrADF['t\u2099(τ)']
        self.hl = qrADF['Half-lives']
        self.cvs = {
            'CV10%': qrADF['CV10%'],
            'CV5%': qrADF['CV5%'],
            'CV1%': qrADF['CV1%']
        }
        self.rawResults = qrADF
        self.results = {
            'quantile': round(self.tau, 2),
            'Lags': self.p,
            'α\u2080(τ)': round(self.alpha_tau, 3),
            'ρ\u2081(τ)': round(self.rho_tau, 3),
            'ρ\u2081(OLS)': round(self.rho_ols, 3),
            'δ\u00B2': round(self.delta2, 3),
            'Half-lives': self.hl,
            't\u2099(τ)': round(self.QURadf, 3),
            'CV10%': round(self.cvs['CV10%'], 4),
            'CV5%': round(self.cvs['CV5%'], 4),
            'CV1%': round(self.cvs['CV1%'], 4)
        }
        return self.results

    @staticmethod
    def QRADF(y, tau, crit_QRadf, bandwidth, model='c', pmax=5, ic='AIC'):
        r"""tau     -  quantile (0.1,...,1)
        """
        y = np.array(y)
        y1 = lagmat(y, maxlag=1)[1:]  # creating lags
        dy = np.diff(y)  # first difference
        x = y1
        
        resultsADF = adfuller(y, maxlag=pmax,
                              regression=model, autolag=ic)
        ADFt, p, cvADF = resultsADF[0], resultsADF[2], resultsADF[4]
        
        # Defining series in case of possible legs
        if p > 0:
            dyl = lagmat(dy, maxlag=p)[p:]
            x = np.c_[y1[p:], dyl]
        # If the ic decides not to include lags
        else:
            x = x[p:]
        
        # Removing tails   
        y = y[p+1:]
        y1 = y1[p:]
        dy = dy[p:]
        
        # Running the quantile regression
        n = len(y)
        qOut1 = QuantReg(y, add_constant(x)).fit(q=tau)

        # calculating alpha and rho
        alpha_tau = qOut1.params[0]
        rho_tau = qOut1.params[1]
        rho_ols = OLS(y, add_constant(x)).fit().params[1]

        # Estimating Half-lifes
        hl = np.log(0.5)/np.log(np.abs(rho_tau))
        hl = '∞' if hl < 0 else round(hl, 3)

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
        qOut2 = QuantReg(y, add_constant(x1)).fit(q=tau+h)
        qOut3 = QuantReg(y, add_constant(x1)).fit(q=tau-h)

        # Extracting betas
        rq1 = qOut2.params
        rq2 = qOut3.params

        # Setting inputs for the unit root test
        z = add_constant(x1)
        mz = z.mean(axis=0)

        q1 = np.matmul(mz, rq1)
        q2 = np.matmul(mz, rq2)
        fz = 2 * h/(q1 - q2)
        if fz < 0:
            fz = 0.01

        xx = np.ones((len(x), 1))
        if p > 0:
            xx = add_constant(dyl)

        # Constructing a NxN matrix
        PX = np.eye(len(xx)) - \
            xx.dot(np.linalg.inv(np.dot(xx.T, xx))).dot(xx.T)
        fzCrt = fz/np.sqrt(tau * (1 - tau))
        eqPX = np.sqrt((y1.T.dot(PX).dot(y1))[0][0])

        # QADF statistic
        QURadf = fzCrt * eqPX * (rho_tau - 1)
        cv = crit_QRadf(delta2, model)

        return {
            'τ': tau,
            'p': p,
            'α\u2080(τ)': alpha_tau,
            'ρ\u2081(τ)': rho_tau,
            'ρ\u2081(OLS)': rho_ols,
            'δ\u00B2': delta2,
            't\u2099(τ)': QURadf,
            'CV10%': cv['10%'],
            'CV5%': cv['5%'],
            'CV1%': cv['1%'],
            'Half-lives': hl
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
            out = rmv_out.replace(',', '\n').replace('\n ', '\n')
            return out
        return object.__repr__(self)

    def summary(self):
        print(self.__repr__())

# Function for the quantile autoregression (to be used separately from QADF)
def QAR(y, tau, pmax=12, regression='c', ic='AIC'):
    L1_y =  lagmat(y, maxlag=1, use_pandas=True)[1:]  # creating lags
    
    # Identifying optimal lags
    resultsADF = adfuller(y,pmax, regression, ic)
    lags = resultsADF[2]
    
    Ldy = lagmat(y.diff()[1:], maxlag=lags, use_pandas=True).add_prefix('Δ')
    X = pd.concat([L1_y, Ldy], axis=1)
    
    # endog and exog
    y = y[lags+1:]
    X = add_constant(X)[lags:]
    
    # Running the quantile regression (also OLS)
    qar = QuantReg(y, X).fit(q=tau)
    ols = OLS(y, X).fit()

    # calculating alpha and rho
    alpha_tau = qar.params[0]
    rho_tau = qar.params[1]
    alpha_ols = ols.params[0]
    rho_ols = ols.params[1]
    
    results =  {
        'quantile': tau,
        'Lags': lags,
        'α₀(τ)': alpha_tau,
        'ρ₁(τ)': rho_tau,
        'α₀(OLS)': alpha_ols,
        'ρ₁(OLS)': rho_ols,
        'model(QAR)': qar,
        'model(OLS)': ols
    }
    return results

# Function that creates a bootstrap following Koenker and Xiao's (2004) resampling procedure
def createBootstrap(y, lags, random_state=42):
    # Data
    dy = y.diff()[1:]
    np.random.seed(random_state)
    q = lags

    # 1) The q-order autoregression
    arModel = AutoReg(dy, lags=q, old_names=False, trend='n').fit()
    betas = np.array(arModel.params)
    resid = arModel.resid

    # 2) Bootstrap sample from the empirical distribution of the centred residuals
    cResid = resid - resid.mean()
    residStar = np.random.choice(cResid, len(cResid))

    # 3) where dy* = dy_j, j=1,..,q
    dyStar = list(dy[:q])
    for i in range(len(residStar)):
        dyStar_t = (betas * dyStar[-q:]).sum() + residStar[i]
        dyStar.append(dyStar_t)

    # 4) Bootstrap samples of levels y_1* = y_1
    yStar = [y[0]]
    for i in range(len(dyStar)):
        yStar_t = yStar[i] + dyStar[i]
        yStar.append(yStar_t)

    # Star series setup
    yStar = pd.Series(yStar, index=y.index[:len(
        y)], name=y.name+'Star'+str(random_state))
    return yStar

# Returns a dataframe of bootstraps
def bootstraps(y, lags, n_replications):
    boots = [createBootstrap(y, lags, random_state=i)
             for i in range(n_replications)]
    return pd.DataFrame(boots).T

# QAR class with a fit function that returns a statsmodels object
class QAR():
    """Quantile autoregressive model
    """
    defaultQuantiles = np.arange(0.1, 1, 0.1)
    def __init__(self, y, pmax=12, regression='c', ic='AIC'):
        """
        Parameters
        ----------
        y : array or dataframe
            endogenous/response variable.
        pmax : int
            Maximum lag which is included in test.
        regression : {"c","ct","ctt","nc"}
            Constant and trend order to include in regression.
        ic : {"AIC", "BIC", "t-stat", None}
            Information criteria to use when automatically determining the lag.
        """
        self.pmax = pmax
        self.regression = regression
        self.ic = ic
        self.name = y.name
        self.symbols = ['α₀', 'ρ₁']
        
        # Setup for endog and exog   
        L1_y =  lagmat(y, maxlag=1, use_pandas=True)[1:]  # creating lags

        # Identifying optimal lags
        resultsADF = adfuller(y, self.pmax, self.regression, self.ic)
        lags = resultsADF[2]

        Ldy = lagmat(y.diff()[1:], maxlag=lags, use_pandas=True).add_prefix('Δ')
        X = pd.concat([L1_y, Ldy], axis=1)

        # endog and exog
        self.y = y[lags+1:]
        self.X = add_constant(X)[lags:]
    
    def fit(self, q=0.5):
        # Running the quantile regression
        return QuantReg(self.y, self.X).fit(q=q)
    
    # Internal param generation function
    def _compareFit(self, q):
        # QAR
        qar = self.fit(q)
        rhoName = qar.params.index[1]

        #  alpha_tau 
        alpha_tau = qar.params[0]
        alpha_tauLowerCI = qar.conf_int().loc['const'][0]
        alpha_tauUpperCI = qar.conf_int().loc['const'][1]

        #  rho_tau 
        rho_tau = qar.params[1]
        rho_tauLowerCI = qar.conf_int().loc[rhoName][0]
        rho_tauUpperCI = qar.conf_int().loc[rhoName][1]

        #  OLS  
        ols = OLS(self.y, self.X).fit()
        rhoName = ols.params.index[1]

        #  alphaOLS  
        alphaOLS = ols.params[0]
        alphaOLSLowerCI = ols.conf_int().loc['const'][0]
        alphaOLSUpperCI = ols.conf_int().loc['const'][1]

        #  rhoOLS
        rhoOLS = ols.params[1]
        rhoOLSLowerCI = ols.conf_int().loc[rhoName][0]
        rhoOLSUpperCI = ols.conf_int().loc[rhoName][1]

        params = {
            'quantile': q,
            'α₀(τ)': alpha_tau,
            'α₀(τ):LB': alpha_tauLowerCI,
            'α₀(τ):UB': alpha_tauUpperCI,
            'α₀(OLS)': alphaOLS,
            'α₀(OLS):LB': alphaOLSLowerCI,
            'α₀(OLS):UB': alphaOLSUpperCI,
            'ρ₁(τ)': rho_tau,
            'ρ₁(τ):LB': rho_tauLowerCI,
            'ρ₁(τ):UB': rho_tauUpperCI,
            'ρ₁(OLS)': rhoOLS,
            'ρ₁(OLS):LB': rhoOLSLowerCI,
            'ρ₁(OLS):UB': rhoOLSUpperCI,
        }
        return params
    
    # Creates a single plot for a parameter
    def _paramPlot(self, param, index, quantiles, fig, nrows, ncolumns):
        # Fitting for many quantiles and plotting
        fits = pd.DataFrame([self._compareFit(q) for q in quantiles])
        x = fits['quantile']
        param_tau = param + '(τ)'
        param_tauLower = param_tau + ':LB'
        param_tauUpper = param_tau + ':UB'
        paramOLS = param + '(OLS)'
        paramOLSLower = paramOLS + ':LB'
        paramOLSUpper = paramOLS + ':UB'

        axs = fig.add_subplot(nrows, ncolumns, index)
        # α₀(τ) over quantiles
        axs.plot(x, fits[param_tau], color='black', label=param_tau)
        axs.plot(x, fits[param_tauLower], linestyle='dotted', color='black')
        axs.plot(x, fits[param_tauUpper], linestyle='dotted', color='black')

        # OLS constant α₀(OLS)
        axs.plot(x, fits[paramOLS], color='blue', label=paramOLS)
        axs.plot(x, fits[paramOLSLower], linestyle='dotted', color='blue')
        axs.plot(x, fits[paramOLSUpper], linestyle='dotted', color='blue')

        axs.set_ylabel(param, fontsize=15)
        axs.legend(loc='lower right')

    def summaryPlot(self, quantiles=defaultQuantiles, figsize=(8, 4), nrows=2, ncolumns=1):
        # For each symbol create a plot
        fig = plt.figure(figsize=figsize)
        for index, symbol in enumerate(self.symbols):
            self._paramPlot(symbol, (index+1), quantiles, fig, nrows, ncolumns)
            
        # common figure parameters
        fig.suptitle(self.name)
        plt.xlabel('Quantiles', fontsize=11)
        return fig

# Plotting data for multiple series on a 2xn plane with 'α₀' and 'ρ₁' as main rows
def comparisonPlot(data, figsize=(15, 8), quantiles=QAR.defaultQuantiles):
    n = len(data.columns)
    fig = plt.figure(figsize=figsize)
    for index, country in enumerate(data):
        y = data[country]
        model = QAR(y)
        for s_index, symbol in enumerate(model.symbols):
            uback = index + (n*s_index + 1)
            model._paramPlot(symbol, uback, quantiles, fig, nrows=2, ncolumns=n)
        fig.axes[index*2].set_title(model.name)
    return fig
    
    # Cleanup   
    fig.axes[0].get_shared_y_axes().join(*fig.axes[::2])
    fig.axes[1].get_shared_y_axes().join(*fig.axes[1::2])
    list(map(lambda axes: axes.set_xticklabels([]), fig.axes[::2]))
    list(map(lambda axes: axes.set_yticklabels([]), fig.axes[2::2]))
    list(map(lambda axes: axes.set_yticklabels([]), fig.axes[3::2]))
    return fig
