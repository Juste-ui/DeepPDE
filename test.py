import numpy as np

from DeepPDE.tools.transform import transform



strike =100
t_min=0
t_max =4
volatility_max=0.3
volatility_min=0.1
correlation_min = 0.2
correlation_max = 0.8
riskfree_rate_min = 0.1
riskfree_rate_max = 0.3

s_max = strike * (1 + 3*volatility_max*t_max)
x_max = np.log(s_max)
x_min = 2*np.log(strike) - x_max
normalised_max = 1
normalised_min = -1
normalise =transform(0,t_max=t_max, strike_price=strike,volatility_min= volatility_min,
                     volatility_max= volatility_max,normalise_min=normalised_min,normalise_max=normalised_max,r_min=riskfree_rate_min,
                     r_max= riskfree_rate_max,rho_min= correlation_min,rho_max= correlation_max)
def transform_ab_to_cd(x, a, b, c, d): 
    """
    Perform a linear transformation of a scalar from the souce interval
    to the target interval.

    Keyword arguments:
    x -- scalar point(s) to transform
    a, b -- interval to transform from
    c, d -- interval to transform to 
    """
    return c + (x-a) * (d-c) / (b-a)


normalised_max = 1
normalised_min = -1
riskfree_rate_min = 0.1
riskfree_rate_max = 0.3


def normalise_riskfree_rate(riskfree_rate):
    """ Transform risk-free rate to its corresponding normalised variable. """
    return transform_ab_to_cd(riskfree_rate,
                              riskfree_rate_min, riskfree_rate_max, 
                              normalised_min, normalised_max)


