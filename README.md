# QADF
The module provides access to a quantile augmented Dickey-Fuller unit root procedure. The test performs unit root quantile autoregression inference following the Koenker and Xiao (2004) methodology.

Parts of the code/logic such as formulas for the bandwidth and critical values (Hansen, 1995) were written using the qradf function (by Saban Nazlioglu) in the Aptech GAUSS tspdlib library as a reference.
 
## Usage
```Python
>>> import pandas as pd
>>> from quantileADF import QADF

>>> y = pd.read_csv('examples.csv')['Y']

>>> qADF = QADF(y, model='c', pmax=5, ic='AIC')
>>> qADF.fit(tau=0.42)
>>> qADF.summary()
"""
quantile: 0.42
rho(quantile): 0.892
rho (OLS): 0.834
delta^2: 0.507
ADF(quantile): -1.881
Critical Values:
1%    -3.195
5%    -2.588
10%   -2.257
dtype: float64
"""
```


