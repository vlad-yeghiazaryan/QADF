# Main
import numpy as np
import pandas as pd

# Unit root
from quantileADF import bootstraps

# Tests and function
from statsmodels.distributions.empirical_distribution import ECDF

# Utilities
import time
import numba

def oneTailUpper(value, ecdfValue, significanceLevels):
    pValue = round(1 - ecdfValue, 3)
    starValue = str(round(value, 2)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t])
    return starValue

def oneTailLower(value, ecdfValue, significanceLevels):
    pValue = round(ecdfValue, 3)
    starValue = str(round(value, 2)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t])
    return starValue

def twoTail(value, ecdfValue, significanceLevels):
    pValue = round(min(ecdfValue, 1 - ecdfValue), 3)
    starValue = str(round(value, 2)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t/2])
    return starValue

def addStars(row, test, significanceLevels):
    boots = row[1]
    value = row[0]
    ecdfFunc = ECDF(boots, side='left')
    ecdfValue = ecdfFunc(value)
    starValue = test(value, ecdfValue, significanceLevels)
    return starValue

def customReport(CountryQADF, results, significanceLevels, dropColumns=None):
    values =  ['α₀(τ)', 'ρ₁(τ)', 'ρ₁(OLS)','tₙ(τ)']
    for value in values:
    	test = twoTail if value == 'α₀(τ)' else oneTailLower
    	CountryQADF[value] = pd.concat([CountryQADF,
        	results.groupby('quantile')[value].apply(lambda x:np.array(x))],
        	axis=1)[value].apply(addStars, axis=1, args=[test, significanceLevels])
    CountryQADF['QKS'] = addStars([CountryQADF['QKS'].mean(), results.groupby('name')['QKS'].mean()], oneTailUpper, significanceLevels)
    CountryQADF.loc[0.2:, 'QKS'] = ''
    CountryQADF['Half-lives'] = CountryQADF['Half-lives'].apply(lambda x: round(x/12) if not isinstance(x, str) else x)
    CountryQADF.rename(columns={'Half-lives':'Half-lives (years)'}, inplace=True)
    CountryQADF.drop(columns=dropColumns, inplace=True) if dropColumns != None else None

    # final cleanup
    report = CountryQADF.T.drop('name')
    report['name']  = CountryQADF['name'][0.1]
    report.set_index(['name',report.index], inplace=True)
    
    return report

def countryReport(y, model, quantiles, repetitions, significanceLevels, dropColumns=None):
        # Start timer
        t1 = time.time()
        # Run once to get final values and optimal lags     
        CountryQADF = model.fitForQuantiles(y, quantiles)
        # Generate bootstrap samples       
        boots = bootstraps(y, CountryQADF['Lags'][0.1], repetitions)
        # Get the bootstrap statistics    
        results = pd.concat([model.fitForQuantiles(boots[yStar], quantiles) for yStar in boots])
        # Customize the final output     
        report = customReport(CountryQADF, results, significanceLevels, dropColumns)
        # Print time spent
        print(f'{y.name} finished in: {round(time.time() - t1, 2)}s')
        return (report, results)

def funcTimer(func):
    def timedFunc(*args, **kwargs):
        t0 = time.time()
        print('Starting Execution:\n')
        results, report = func(*args, **kwargs)
        m, s = divmod(time.time() - t0, 60)
        print(f'\nTotal time spent executing: {round(m)}m, {round(s)}s')
        return (results, report)
    return timedFunc

@funcTimer
@numba.jit(forceobj=True, parallel=True)
def reportCountries(data, model, quantiles, repetitions, significanceLevels=[0.01,0.05,0.1], dropColumns=None):
    countriesReport = []
    resultsAll = []

    for country in data:
        y = data[country]
        countryInfo, results = countryReport(y, model, quantiles, repetitions, significanceLevels, dropColumns)
        resultsAll.append(results)
        countriesReport.append(countryInfo)
    report = pd.concat(countriesReport)
    report.index.names = ['Countries', 'Variable/Quantile']
    resultsAll = pd.concat(resultsAll)
    return (resultsAll, report)
