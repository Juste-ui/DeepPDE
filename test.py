import numpy as np

from DeepPDE.tools.transform import transform



normalise =transform(0,t_max=5, strike_price=100,volatility_min= 0.2,
                     volatility_max= 0.5,normalise_min=-1,normalise_max=1,r_min=0.01,
                     r_max= 0.09,rho_min= 0.1,rho_max= 0.9)

mu =np.array([0.02,0.3,0.3,0.3,0.01,0.3,0.3,0.3,0.006]).reshape(3,3)

cov=normalise.transform_to_sigma(mu,dimension_states=3)
corr=normalise.transform_to_corr(mu, dimension_states=3)

print(corr)