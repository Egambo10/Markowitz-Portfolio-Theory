import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    -Wealth Index
    -Previus Peaks
    -Percent Drawdowns
    '''
    wealth_index = 100*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/ previous_peaks
    return pd.DataFrame({
        'Wealth':wealth_index,
        'Peaks':previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    '''
    Load the Fama-French Dataset for returns of the Top and Bottom Deciles by MarketCap
    '''
    returns = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                     header=0, index_col=0, parse_dates= True,na_values = -99.99
                     )
    columns = ['Lo 10', 'Hi 10']
    returns = returns[columns]
    returns.columns= ['SmallCap','LargeCap']
    returns= returns/100
    returns.index= pd.to_datetime(returns.index, format= '%Y%m').to_period('M')
    return returns

def get_ffme_returns2(tickers):
    '''
    Load the Fama-French Dataset for returns of the Top and Bottom Deciles by MarketCap
    '''
    returns = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                     header=0, index_col=0, parse_dates= True,na_values = -99.99
                     )
    columns = tickers
    returns = returns[columns]
    returns= returns/100
    returns.index= pd.to_datetime(returns.index, format= '%Y%m').to_period('M')
    return returns

def get_hfi_returns():
    '''
    Load and format EDHEC Hedge Fund Index Returns
    '''
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', 
                     header=0, index_col=0, parse_dates= True,na_values = -99.99
                     )
    
    hfi= hfi/100
    hfi.index= hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    '''
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    '''
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header=0, index_col=0, parse_dates= True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    '''
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    '''
    ind = pd.read_csv('data/ind30_m_size.csv', header=0, index_col=0, parse_dates= True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    '''
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    '''
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header=0, index_col=0, parse_dates= True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind
def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

                         
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    '''
    Weights -> Returns
    '''
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    '''
    Weights -> Vol
    '''
    
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)

from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    '''
    target_ret -> W
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type' : 'eq',
        'args': (er,),
        'fun': lambda weights, er : target_return - portfolio_return(weights, er) 
    }
    weights_sum_to_1= {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
        
    results = minimize(portfolio_vol, init_guess, 
                       args=(cov,), method='SLSQP',
                      options={'disp':False},
                      constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x
    

def optimal_weights(n_points, er, cov):
    '''
    -> list of weights to run the optimizer on to minimize the vol
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate,er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
   
    weights_sum_to_1= {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Returns the engative of the sharpe ratio, given wieghts
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess, 
                       args=(riskfree_rate, er, cov,), method='SLSQP',
                      options={'disp':False},
                      constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def gmv(cov):
    '''
    Returns the weight of the Global Minimum Vol portfolio
    given the covariance matrix
    '''
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, style='.-', show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display EW
        ax.plot([vol_ew], [r_ew], color = 'goldenrod', marker='o', markersize=10)
    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display EW
        ax.plot([vol_gmv], [r_gmv], color = 'midnightblue', marker='o', markersize=10)
    if show_cml:
        ax.set_xlim(left = 0)
        rf = riskfree_rate
        w_msr = msr(rf,er, cov)
        r_msr = portfolio_return(w_msr,er)
        vol_msr = portfolio_vol(w_msr,cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [rf, r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', markersize=12, linewidth=2)
        return ax

def semideviation(r):
    '''
    Returns the semideviation aka negative semideviaiton of r
    r must be a series of a DF
    '''
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def is_normal(r, level = 0.01):
    '''
    Applies the Jarque-Bera test to determine if a eries is normal or not
    Test is applied at the 1% level by default
    Return True if the hypothesis of normality is accepted, False otherwise
    '''
    statistis, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level=5):
    '''
    Returns the historic VaR at a specified level
    i.e. returns the number such that "level" oercent of the returns
    fall bellow that number, and the (100-level) percent are above
    '''
    
    if isinstance(r, pd.DataFrame):
        # if r is a DF (the frist time called), the apply the function to every column 
        # in the DF
        return r.aggregate(var_historic, level = level)
    # then when is called for the second time and go on it is not going to be a DF because of numpy
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')

        
from scipy.stats import norm


def var_gaussian(r,level=5, modified= False):
    '''
    Returns the Parametric Gaussian VaR of a Series or DF
    '''
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
               (z**2 - 1) * s / 6 +
               (z**3 - 3 * z) * (k - 3) / 24 -
               (2*z**3 - 5*z) * (s**2) / 36
          )
        
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    '''
    Computes the COnditional VaR of a Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    # then when is called for the second time and go on it is not going to be a DF because of numpy
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')
