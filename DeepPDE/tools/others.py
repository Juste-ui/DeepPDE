import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
from sklearn.utils.extmath import randomized_svd

from DeepPDE.tools.Transform import Transform



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
normalise =Transform(0,t_max=t_max, strike_price=strike,volatility_min= volatility_min,
                     volatility_max= volatility_max,normalise_min=normalised_min,normalise_max=normalised_max,r_min=riskfree_rate_min,
                     r_max= riskfree_rate_max,rho_min= correlation_min,rho_max= correlation_max)


def localisation(t, s1, s2, localisation_parameter,strike_price,riskfree_rate):
    """ Return the value of the localisation used in the network. """
    return 1/localisation_parameter * np.log(1 +
                    np.exp(localisation_parameter * (
                        0.5*(s1+s2) - np.exp(-riskfree_rate*t)*strike_price))
                    )


def get_random_points_of_interest(nr_samples, 
                    t_min_interest, t_max_interest, s_min_interest,
                    s_max_interest,dimension_states,dimension_param,
                    parameter_min_interest_normalised=normalised_min,
                    parameter_max_interest_normalised=normalised_max):
    """ Get a number of random points within the defined domain of interest. """
    t_sample = np.random.uniform(t_min_interest, t_max_interest, 
                                 [nr_samples, 1])
    t_sample_normalised = normalise.normalise_time(t_sample)

    s_sample = np.random.uniform(s_min_interest, s_max_interest, [nr_samples, dimension_states])
    
    s1_sample = s_sample[:, 0:1]
    s2_sample = s_sample[:, 1:2]

    x_sample_normalised = normalise.normalise_logprice(np.log(s_sample))

    parameter_sample_normalised = np.random.uniform( normalised_min, normalised_max, [nr_samples, dimension_param])
    data_normalised = np.concatenate( (t_sample_normalised, x_sample_normalised, parameter_sample_normalised),
        axis=1 )

    riskfree_rate_sample =normalise.transform_to_riskfree_rate(
        parameter_sample_normalised[:, 0])
    volatility1_sample =  normalise.transform_to_volatility(
        parameter_sample_normalised[:, 1])
    volatility2_sample = normalise.transform_to_volatility(
        parameter_sample_normalised[:, 2])
    correlation_sample = normalise.transform_to_correlation(
        parameter_sample_normalised[:, 3])
    
    return data_normalised, t_sample.reshape(-1), s1_sample.reshape(-1), \
            s2_sample.reshape(-1), riskfree_rate_sample, volatility1_sample, \
            volatility2_sample, correlation_sample



def get_points_for_plot_at_fixed_time(t_fixed,
                s_min_interest, s_max_interest,
                riskfree_rate_fixed,
                volatility1_fixed,
                volatility2_fixed,
                correlation_fixed,
                n_plot):
    """ Get the spacial and normalised values for surface plots 
    at fixed time and parameter, varying both asset prices. 
    """
    s1_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    s2_plot = np.linspace(s_min_interest, s_max_interest, n_plot).reshape(-1,1)
    [s1_plot_mesh, s2_plot_mesh] = np.meshgrid(s1_plot, s2_plot, indexing='ij')

    x1_plot_mesh_normalised = normalise.normalise_logprice( np.log(s1_plot_mesh)).reshape(-1,1)

    x2_plot_mesh_normalised = normalise.normalise_logprice(np.log(s2_plot_mesh)).reshape(-1,1)

    t_mesh = t_fixed  * np.ones((n_plot**2, 1))
    t_mesh_normalised = normalise.normalise_time(t_mesh)

    parameter1_mesh_normalised = (normalise.normalise_riskfree_rate(riskfree_rate_fixed)   * np.ones((n_plot**2, 1)))
    parameter2_mesh_normalised = (normalise.normalise_volatility(volatility1_fixed)   * np.ones((n_plot**2, 1)))
    parameter3_mesh_normalised = (normalise.normalise_volatility(volatility2_fixed)   * np.ones((n_plot**2, 1)))
    parameter4_mesh_normalised = (normalise.normalise_correlation(correlation_fixed)   * np.ones((n_plot**2, 1)))

    x_plot_normalised = np.concatenate((t_mesh_normalised,
                                        x1_plot_mesh_normalised,
                                        x2_plot_mesh_normalised,
                                        parameter1_mesh_normalised, 
                                        parameter2_mesh_normalised,
                                        parameter3_mesh_normalised, 
                                        parameter4_mesh_normalised), axis=1)

    
    return s1_plot_mesh, s2_plot_mesh, x_plot_normalised






def decompose_covariance_matrix(t, volatility1, volatility2, correlation):
    """ Decompose covariance matrix as in Lemma 3.1 of Bayer et. al (2018). """
    sigma_det = (1-correlation**2) * volatility1**2 * volatility2**2
    sigma_sum = (volatility1**2 + volatility2**2 
                  - 2*correlation*volatility1*volatility2)

    ev1 = volatility1**2 - correlation*volatility1*volatility2
    ev2 = -(volatility2**2 - correlation*volatility1*volatility2)
    ev_norm = np.sqrt(ev1**2 + ev2**2)

    eigenvalue = volatility1**2 + volatility2**2 - 2*sigma_det/sigma_sum

    v_mat = np.array([ev1, ev2]) / ev_norm
    d = t * np.array([sigma_det/sigma_sum, eigenvalue])
    return d, v_mat


def decompose_covariance_matrix2(t,sigma):
    
     v_mat, d,_ = randomized_svd(sigma,  n_components=15, n_iter=5, random_state=None)
     d=t*d
     return d, v_mat


def one_dimensional_exact_solution(
        t, s, riskfree_rate, volatility, strike_price):
    """ Standard Black-Scholes formula """

    d1 = (1 / (volatility*np.sqrt(t))) * (
            np.log(s/strike_price) 
            + (riskfree_rate + volatility**2/2.) * t
        )
    d2 = d1 - volatility*np.sqrt(t)
    return (norm.cdf(d1) * s 
            - norm.cdf(d2) * strike_price * np.exp(-riskfree_rate*t))


def exact_solution(
    t, s1, s2, riskfree_rate, volatility1, volatility2, correlation):
    """ Compute the option price of a European basket call option. """
    if t == 0:
        return np.maximum(0.5*(s1+s2) - strike, 0)

    d, v = decompose_covariance_matrix(
        t, volatility1, volatility2, correlation)
    
    beta = [0.5 * s1 * np.exp(-0.5*t*volatility1**2),
            0.5 * s2 * np.exp(-0.5*t*volatility2**2)]
    integration_points, integration_weights = hermgauss(33)

    # Transform points and weights
    integration_points = np.sqrt(2*d[1]) * integration_points.reshape(-1, 1)
    integration_weights = integration_weights.reshape(1, -1) / np.sqrt(np.pi)

    h_z = (beta[0] * np.exp(v[0]*integration_points)
           + beta[1] * np.exp(v[1]*integration_points))

    evaluation_at_integration_points = one_dimensional_exact_solution(
        t=1, s=h_z * np.exp(0.5*d[0]), 
        strike_price=np.exp(-riskfree_rate * t) * strike, 
        volatility=np.sqrt(d[0]), riskfree_rate=0.
        )
    
    solution = np.matmul(integration_weights, evaluation_at_integration_points)
    
    return solution[0, 0]

def exact_solution2 (t, s,riskfree_rate, sigma, dimension_states):
    """ Compute the option price of a European basket call option. """
    if t == 0:
        return np.maximum(0.5*(sum(s)) - strike, 0)
    
    d, v = decompose_covariance_matrix2(t,sigma)
    beta=[]
    for i in range(dimension_states):
        beta.append(0.5 * s[i] * np.exp(-0.5*t*sigma[i,i]**2))

    integration_points, integration_weights = hermgauss(33)

     # Transform points and weights
    integration_points = np.sqrt(2*d[dimension_states-1]) * integration_points.reshape(-1, 1)
    integration_weights = integration_weights.reshape(1, -1) / np.sqrt(np.pi)
    
    h_z =np.dot(np.exp(v[:,dimension_states-1]*integration_points) , beta).reshape(-1,1)
    

    evaluation_at_integration_points = one_dimensional_exact_solution(t=1, s=h_z*np.exp(0.5*d[0]),
         strike_price=np.exp(-riskfree_rate * t) * strike, 
         volatility=np.sqrt(d[0]), riskfree_rate=0.)
         
    solution = np.matmul(integration_weights, evaluation_at_integration_points)
    
    return solution[0, 0]
    


    
def BS_Call(S,T,riskfree_rate,sigma,strike_price):
  d_1 = (np.log(S/strike_price) + (riskfree_rate+sigma**2/2)*T)/(sigma*np.sqrt(T))
  d_2 = d_1 - sigma * np.sqrt(T)
  return(S*stats.norm.cdf(d_1)-strike_price*np.exp(-riskfree_rate*T)*stats.norm.cdf(d_2))

def BS_Vega_Call(S,strike_price,T,riskfree_rate,sigma):
  d_1 = (np.log(S/strike_price) + (riskfree_rate+sigma**2/2)*T)/(sigma*np.sqrt(T))
  return S * np.sqrt(T)*stats.norm.pdf(d_1)

# To compute the implied_volatility, we use Newton-Raphson method
def Implied_Volatility(Price,S,strike_price,riskfree_rate,T):
  try:
    Implied_vol = 0.5
    Target = 10**(-4)
    Keep_computing = True
    while Keep_computing:
      aux = Implied_vol
      Implied_vol = Implied_vol - (BS_Call(S,T,riskfree_rate,Implied_vol,strike_price) - Price)/ BS_Vega_Call(S,strike_price,T,riskfree_rate,Implied_vol)
      if abs(BS_Call(S,T,riskfree_rate,Implied_vol,strike_price) - Price) < Target:
        Keep_computing = False
    return(Implied_vol)
  except:
    return(Implied_Volatility_bis(Price,S,strike_price,riskfree_rate,T))

def Implied_Volatility_bis(Price,S,riskfree_rate,strike_price,T):
  Implied_vol = 0.9
  for i in range(2,8):
    if BS_Call(S,T,riskfree_rate,Implied_vol,strike_price) - Price > 0:
      while BS_Call(S,T,riskfree_rate,Implied_vol,strike_price) - Price > 0:
        Implied_vol = Implied_vol - 10**(-i)
    else:
      while BS_Call(S,T,riskfree_rate,Implied_vol,strike_price) - Price < 0:
        Implied_vol = Implied_vol + 10**(-i)
  if Implied_vol < 0.05:
    print(Implied_vol,Price,BS_Call(S,T,riskfree_rate,Implied_vol,strike_price),S,strike_price,riskfree_rate,T)
  return(Implied_vol)


