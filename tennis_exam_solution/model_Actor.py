import tensorflow as tf
import numpy      as np

import tensorflow_probability as tfp

# -----------------------------------------------------

class lambda_function () :

    def __init__ ( self , settings ) : self.settings = settings
    def expand   ( self ,        x ) : return tf.expand_dims ( x , axis = -1 )

class MyNormalLayer ( tf.keras.layers.Layer ) :

    def __init__ ( self , settings ) :

        super ( MyNormalLayer , self ).__init__ ()

        self.settings = settings

    def build                  (   self                 , input_shape                          ) :
        ones     = tf.ones     ( ( self.settings.agents , input_shape [ -1 ] ) , dtype = float )
        self.std = tf.Variable (        ones            , name = 'std'         , dtype = float )

    def call ( self , inputs , actions , bool_actions ) :

        tfp_dist = tfp.distributions.Normal ( loc = inputs , scale = self.std )

        if bool_actions == 0 : actions = tfp_dist.sample ()

        log_prob = tfp_dist.log_prob ( actions )

        log_prob_sum = tf.math.reduce_sum ( log_prob , axis = 2 , keepdims = True )

        entropy = tfp_dist.entropy ()

        entropy_sum = tf.math.reduce_sum ( entropy , axis = 2 , keepdims = True )

        # print ( self.std          ) # != [[1,1],[1,1]]
        # print ( bool_actions < -1 ) # != True/False

        return actions , log_prob_sum , entropy_sum

# -----------------------------------------------------
# -----------------------------------------------------

class Network_Actor () :
    def __init__    ( self , settings , RNN = 0 ) :
        self.settings      = settings
        self.optimizer     = tf.keras.optimizers.Adam ( learning_rate = 3e-4 , epsilon = 1e-5 ) # no clipnorm = 5

        self.states_input_shape = [ settings.agents , settings.state_size ]
        print ( 'states input_shape:' , self.states_input_shape )

        self.actions_input_shape = [ settings.agents , settings.action_size ]
        print ( 'actions input_shape:' , self.actions_input_shape )

        self.object_lambda = lambda_function ( settings )

        self.ortog = tf.keras.initializers.Orthogonal ( gain = 1e-3 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        States_Input  = tf.keras.Input ( shape = self.states_input_shape  )
        Actions_Input = tf.keras.Input ( shape = self.actions_input_shape )
        Bools_Input   = tf.keras.Input ( shape = ( 1                  , ) )

        Input_Lambda_1  = None
        Input_Reduced   = None
        Input_Reduced_2 = None

        DenseDeep1 = None
        DenseDeep2 = None

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        if RNN == 1 :

           DenseDeep1 = tf.keras.layers.SimpleRNN ( settings.hidden                  , name = 'RNN_relu_1' , input_shape = self.states_input_shape ,
                                                    activation = self.settings.basic ,
                                                    return_sequences = True          , go_backwards = False ) ( States_Input )

           DenseDeep2 = tf.keras.layers.SimpleRNN ( settings.hidden                  , name = 'RNN_relu_2' ,
                                                    activation = self.settings.basic ,
                                                    return_sequences = True          , go_backwards = False ) ( DenseDeep1 )
        else :

           DenseDeep1 = tf.keras.layers.Dense ( self.settings.hidden , input_shape = self.states_input_shape ,
                                                name = 'DenseDeep1'  , activation  = self.settings.basic     ) ( States_Input )

           DenseDeep2 = tf.keras.layers.Dense ( self.settings.hidden , name = 'DenseDeep2' , activation = self.settings.basic ) ( DenseDeep1 )

        Out_Actions = tf.keras.layers.Dense ( self.settings.action_size ,
                                              activation = self.settings.final ,
                                              name = 'Out_Actions' , kernel_initializer = self.ortog ) ( DenseDeep2 )

        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------

        myLayer = MyNormalLayer ( self.settings )

        Out_Actions_2 , log_prob_sum , entropy_sum = myLayer ( Out_Actions , Actions_Input , Bools_Input )

        self.model = tf.keras.Model ( inputs  = [ States_Input , Actions_Input , Bools_Input ] ,
                                      outputs = [ Out_Actions_2 , log_prob_sum , entropy_sum ] )
        self.model.summary ()

    def setWeights ( self , weights ) : self.model.set_weights ( weights )
