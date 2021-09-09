import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import add_constant, OLS, QuantReg
from statsmodels.tsa.stattools import adfuller, lagmat, add_trend
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numba


class QADF:
    r"""Quantile unit root test

    Based on the Koenker, R., Xiao, Z., 2004. methodology.

    Parameters
    ----------
    endog   -  Nx1 matrix. Preferably pandas series or a numpy array.
    model   -   c  = Constant (Default in Koenker & Xiao (2004)).
                ct = Constant and trend.
    pmax    -  Maximum number of lags for dy (serially correlated terms).
    ic      -  Information Criterion: Default: AIC
                "AIC"    = Akaike
                "BIC"    = Schwarz
                "t-stat" = t-stat significance
    exog    -  Additional exogenous variables to be included in the estimation. Default: None

    Returns
    -------
    τ (quantile)  - the quantile for which the regression was estimated for.
    Lags          - the effective number of lags determined through the information criterion.
    α₀(τ)         - the magnitude of the shock.
    ρ₁(τ)         - the estimated rho parameter for a given quantile. The speed of mean reversion.
    ρ₁(OLS)       - the estimated rho parameter for the OLS estimation.
    δ²            - A nuisance parameter used to estimate critical values. See equation (10) at page 778.
    Half-lives    - ln(0.5)/ln(abs(ρ₁(τ))).
    tₙ(τ)          - the quantile unit root statistic (t-ratio for a given quantile).
    cv            - 1%, 5%, 10% critical values for the estimated δ².

    """

    def __init__(self, endog, model='c', pmax=5, ic='AIC', exog=None):
        # setup
        self.convergence_failures = []
        self.endog_original = endog
        self.setup(endog, model, pmax, ic, exog)

    def setup(self, endog, model=None, pmax=None, ic=None, exog=None):
        # Erase previous results
        self.results = None

        # Custom instance changes
        if model != None:
            self.model = model
        if pmax != None:
            self.pmax = pmax
        if ic != None:
            self.ic = ic

        if type(endog) != pd.Series:
            self.endog = pd.Series(endog, name='y')
        else:
            self.endog = endog

        # Creating endog and exog
        y1 = pd.DataFrame(self.endog.shift(1)[1:]).add_suffix(
            '.L1')  # creating lags
        dy = self.endog.diff(1)[1:]  # first difference
        X = y1

        resultsADF = adfuller(self.endog, maxlag=self.pmax,
                              regression=model, autolag=self.ic)
        ADFt, p, cvADF = resultsADF[0], resultsADF[2], resultsADF[4]

        # Defining series in case of possible legs
        if p > 0:
            dyl = pd.DataFrame(lagmat(dy, maxlag=p), index=dy.index,
                               columns=['Diff.L' + str(i) for i in range(1, 1 + p)])[p:]
            X = pd.concat([y1[p:], dyl], axis=1)
        # If the ic decides not to include lags
        else:
            X = X[p:]

        # Removing tails
        self.lags = p
        self.extraExog = exog[p + 1:] if type(exog) != type(None) else None
        self.L1 = y1[p:]
        self.diff = dy[p:]
        self.diffLags = dyl if self.lags > 0 else None
        self.endog = self.endog[p + 1:]
        if self.model == 'c':
            self.exog = add_constant(X)
        elif self.model == 'ct':
            self.exog = add_trend(add_constant(X), 'ct')
        else:
            raise ValueError(
                'Model type is not recognized: ' + str(self.model))

        # Adding additional exog if present
        self.exog = pd.concat([self.exog, self.extraExog], axis=1)
        return self

    # Fitting data for different quantiles
    def fitForQuantiles(self, quantiles):
        r"""quantiles -  A list like array of the quantiles to perform the fitting.
            example: np.arange(0.1, 1, 0.1)
        """
        results = []
        for tau in quantiles:
            res = self.fit(tau)
            results.append(res.results)
            convergence_failure = False
            if res.regression.iterations >= 1000:
                convergence_failure = True
            self.convergence_failures.append({'name':self.endog_original.name, **res.results, 
                                             'convergence_failure':convergence_failure})
        df = pd.DataFrame(results)

        # Adding the QKS statistic
        df['QKS'] = max(map(abs, df['tₙ(τ)']))
        df['name'] = self.endog.name
        df.set_index('quantile', inplace=True)
        return df

    def fit(self, tau):
        r"""tau -  quantile (ranges: 0.1,...,1)
        """
        # Running the quantile regression
        n = len(self.endog)
        qOut1 = QuantReg(self.endog, self.exog).fit(q=tau)

        # calculating alpha and rho
        alpha_tau = qOut1.params[0]
        rho_tau = qOut1.params[1]
        rho_ols = OLS(self.endog, self.exog).fit().params[1]

        # Estimating Half-lifes
        hl = np.log(0.5) / np.log(np.abs(rho_tau))
        hl = '∞' if hl < 0 else round(hl, 3)

        # calculating delta2 using covariance
        ind = qOut1.resid < 0
        phi = tau - ind
        w = self.diff
        cov = np.cov(phi, w)[0, 1]
        delta2 = (cov / (np.std(w, ddof=1) * np.sqrt(tau * (1 - tau)))) ** 2

        # Calculating quantile bandwidth
        h = self.bandwidth(tau, n, True)
        if tau <= 0.5 and h > tau:
            h = self.bandwidth(tau, n, False)
            if h > tau:
                h = tau / 1.5

        if tau > 0.5 and h > 1 - tau:
            h = self.bandwidth(tau, n, False)
            if h > (1 - tau):
                h = (1 - tau) / 1.5

        # Defining some inputs
        if self.model == 'c':
            X = add_constant(self.L1)
        elif self.model == 'ct':
            X = add_trend(add_constant(self.L1), 'ct')
        else:
            raise ValueError(
                'Model type is not recognized: ' + str(self.model))

        X = pd.concat([X, self.extraExog], axis=1)

        # The common case
        if self.lags > 0:
            X = self.exog

        # Running the other 2 QuantRegs
        qOut2 = QuantReg(self.endog, X).fit(q=tau + h)
        qOut3 = QuantReg(self.endog, X).fit(q=tau - h)

        # Extracting betas
        rq1 = qOut2.params
        rq2 = qOut3.params

        # Setting inputs for the unit root test
        z = X
        mz = z.mean(axis=0)

        q1 = np.matmul(mz, rq1)
        q2 = np.matmul(mz, rq2)
        fz = 2 * h / (q1 - q2)
        if fz < 0:
            fz = 0.01

        xx = pd.Series(np.ones(len(X)), name='const')
        if self.lags > 0:
            if self.model == 'c':
                xx = add_constant(self.diffLags)
            elif self.model == 'ct':
                xx = add_trend(add_constant(self.diffLags), 'ct')
            else:
                raise ValueError(
                    'Model type is not recognized: ' + str(self.model))

        # Adding extra exog
        xx = pd.concat([xx, self.extraExog], axis=1)

        # aligning matrices
        xx = np.array(xx)
        y1 = np.array(self.L1)

        # Constructing a NxN matrix
        PX = np.eye(len(xx)) - \
            xx.dot(np.linalg.inv(np.dot(xx.T, xx))).dot(xx.T)
        fzCrt = fz / np.sqrt(tau * (1 - tau))
        eqPX = np.sqrt((y1.T.dot(PX).dot(y1))[0][0])

        # QADF statistic
        QURadf = fzCrt * eqPX * (rho_tau - 1)

        # calculating critical values associated with our delta2
        cv = self.crit_QRadf(delta2, self.model)

        # Exposing variables
        self.tau = tau
        self.alpha_tau = alpha_tau
        self.rho_tau = rho_tau
        self.rho_ols = rho_ols
        self.delta2 = delta2
        self.QURadf = QURadf
        self.hl = hl
        self.regression = qOut1
        self.cvs = {
            'CV10%': cv['10%'],
            'CV5%': cv['5%'],
            'CV1%': cv['1%']
        }
        self.results = {
            'quantile': round(self.tau, 2),
            'Lags': self.lags,
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
        return self

    @staticmethod
    @numba.jit(forceobj=True, parallel=True)
    def bandwidth(tau, n, is_hs, alpha=0.05):
        x0 = norm.ppf(tau)  # inverse of cdf
        f0 = norm.pdf(x0)  # Probability density function

        if is_hs:
            a = n**(-1 / 3)
            b = norm.ppf(1 - alpha / 2)**(2 / 3)
            c = ((1.5 * f0**2) / (2 * x0**2 + 1))**(1 / 3)

            h = a * b * c
        else:
            h = n**(-0.2) * ((4.5 * f0**4) / (2 * x0**2 + 1)**2)**0.2

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
                ct = wa * crt.iloc[(r2a - 1), :] + (1 - wa) * \
                    crt.iloc[(r2b - 1), :]
        return ct

    def __repr__(self):
        try:
            self.regression
        except AttributeError:
            return str(AutoReg)
        else:
            return object.__repr__(self.regression)

    def summary(self):
        if self.results != None:
            rmv_chars = {'}': '', '{': '', "'": ''}
            rmv_out = str(self.results).translate(str.maketrans(rmv_chars))
            out = rmv_out.replace(',', '\n').replace('\n ', '\n')
            print(out)
        else:
            return object.__repr__(self)

# Function that creates a bootstrap following Koenker and Xiao's (2004) resampling procedure


def createBootstrap(y, lags, random_state=42):
    # Data
    dy = y.diff()[1:]
    np.random.seed(random_state)
    q = lags if lags > 0 else 1

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
        y)], name=y.name + 'Star' + str(random_state))
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

    def __init__(self, y, exog=None, pmax=12, regression='c', ic='AIC'):
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
        L1_y = lagmat(y, maxlag=1, use_pandas=True)[1:]  # creating lags
        exog = exog[1:] if exog is not None else exog

        # Identifying optimal lags
        resultsADF = adfuller(y, self.pmax, self.regression, self.ic)
        lags = resultsADF[2]

        Ldy = lagmat(y.diff()[1:], maxlag=lags,
                     use_pandas=True).add_prefix('Δ')
        X = pd.concat([L1_y, Ldy, exog], axis=1)

        # endog and exog
        self.y = y[lags + 1:]
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
            self._paramPlot(symbol, (index + 1), quantiles,
                            fig, nrows, ncolumns)

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
            uback = index + (n * s_index + 1)
            model._paramPlot(symbol, uback, quantiles,
                             fig, nrows=2, ncolumns=n)
        fig.axes[index * 2].set_title(model.name)
    return fig

    # Cleanup
    fig.axes[0].get_shared_y_axes().join(*fig.axes[::2])
    fig.axes[1].get_shared_y_axes().join(*fig.axes[1::2])
    list(map(lambda axes: axes.set_xticklabels([]), fig.axes[::2]))
    list(map(lambda axes: axes.set_yticklabels([]), fig.axes[2::2]))
    list(map(lambda axes: axes.set_yticklabels([]), fig.axes[3::2]))
    return fig
