from DeepPDE.tools.Transform import Transform
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

np.random.seed(42)

strike_price =100
t_min=0
t_max =4
volatility_max=0.3
volatility_min=0.1
correlation_min = 0.2
correlation_max = 0.8
riskfree_rate_min = 0.1
riskfree_rate_max = 0.3

s_max = strike_price * (1 + 3*volatility_max*t_max)
x_max = np.log(s_max)
x_min = 2*np.log(strike_price) - x_max
normalised_max = 1
normalised_min = -1

normalise = Transform(t_min=0, t_max=t_max, strike_price=strike_price,volatility_min=volatility_min,
                     volatility_max=volatility_max, normalise_min=-1, normalise_max=1,
                     r_min=riskfree_rate_min, r_max=riskfree_rate_max,rho_max=correlation_max, rho_min=correlation_min)




class HighwayLayer(keras.layers.Layer):
    """ Define one layer of the highway network. """

    def __init__(self, units=50, original_input=None):
        """ Construct the layer by creating all weights and biases in keras. """
        super(HighwayLayer, self).__init__()
        self.units = units
    
       
        
        # create all weights and biases
        self.Uz = self.add_weight("Uz", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Ug = self.add_weight("Ug", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Ur = self.add_weight("Ur", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Uh = self.add_weight("Uh", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)

        self.Wz = self.add_weight("Wz", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wg = self.add_weight("Wg", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wr = self.add_weight("Wr", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wh = self.add_weight("Wh", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)

        self.bz = self.add_weight("bz", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.bg = self.add_weight("bg", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.br = self.add_weight("br", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.bh = self.add_weight("bh", shape=(self.units,),
                                  initializer="random_normal", trainable=True)

    def call(self, input_combined):
        """ Returns the result of the layer calculation.
        
        Keyord arguments:
        input_combined -- Dictionary containing the original input of 
        the neural network as 'original_variable' and 
        the output of the previous layer as 'previous layer'.
        """

        previous_layer = input_combined['previous_layer']
        original_variable = input_combined['original_variable']

        # Evaluate one layer using the weights created by the constructor
        Z = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Uz)
            + tf.matmul(previous_layer, self.Wz)
            + self.bz)

        G = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Ug)
            + tf.matmul(previous_layer, self.Wg)
            + self.bg)

        R = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Ur)
            + tf.matmul(previous_layer, self.Wr)
            + self.br)

        SR = tf.multiply(previous_layer, R)

        H = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Uh)
            + tf.matmul(SR, self.Wh)
            + self.bh)

        one_minus_G = tf.ones_like(G) - G

        return tf.multiply(one_minus_G, H) + tf.multiply(Z, previous_layer)


def create_network(inputs,nr_nodes_per_layer : int,localisation_parameter : float, dimension_total: int):
    """ Creates the neural network by creating three highway layers and an 
    output layer. Returns the output of these layers as a tensorflow variable.

    Keyword arguments:
    inputs -- Tensorflow variable of the input layer
    """
    layer0 = keras.layers.Dense(nr_nodes_per_layer, activation="tanh")

    layer1 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)
    layer2 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)
    layer3 = HighwayLayer(units=nr_nodes_per_layer,
                          original_input=dimension_total)
    
    last_layer = keras.layers.Dense(1)

    outputs_layer0 = layer0(inputs)
    outputs_layer1 = layer1({'previous_layer': outputs_layer0, 
                             'original_variable': inputs})
    outputs_layer2 = layer2({'previous_layer': outputs_layer1, 
                             'original_variable': inputs})
    outputs_layer3 = layer3({'previous_layer': outputs_layer2, 
                             'original_variable': inputs})

    outputs_dnn = last_layer(outputs_layer3)
    
    inputs_t_normalised = inputs[:, 0:1]
    inputs_x1_normalised = inputs[:, 1:2]
    inputs_x2_normalised = inputs[:, 2:3]
    inputs_p1_normalised = inputs[:, 3:4]

   

    inputs_t = normalise.transform_to_time(inputs_t_normalised)
    inputs_x1 = normalise.transform_to_logprice(inputs_x1_normalised)
    inputs_x2 = normalise.transform_to_logprice(inputs_x2_normalised)
    inputs_s_mean = (tf.math.exp(inputs_x1) + tf.math.exp(inputs_x2))/2.
    riskfree_rate = normalise.transform_to_riskfree_rate(inputs_p1_normalised)

    localisation = tf.math.log(1+tf.math.exp(localisation_parameter * (
            inputs_s_mean -  normalise.strike_price * tf.exp( - riskfree_rate * inputs_t)
              )))/localisation_parameter

    return outputs_dnn + localisation

