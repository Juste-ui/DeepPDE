import numpy as np
from DeepPDE.Classes.HighwayLayer import  HighwayLayer
from DeepPDE.tools.transform import transform

normalise = transform(0,4,100,0,0.20,-1,1,0,0.1,-1,1 )

print(normalise.normalise_correlation(0.8))