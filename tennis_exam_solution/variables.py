import tensorflow as tf
import numpy      as np
import math

# -----------------------------------------------------

class Variables () :

    def __init__ ( self , state_size = 24 , action_size = 2 , hidden = 512 ,

                          max_try = 1000 * 5 , basic = tf.nn.relu , final = tf.nn.tanh ,

                          thread_cpus = 11 , vector = 1 , agents = 2 ,

                          env = None , brain_name = '' , beta = 0.01 , clip_size = 0.2 ,

                          small_constant = 0.01 , batch_size = 5 , verbosity = 100 , quality = 0.01 ) :

        self.action_size = action_size
        self.state_size  = state_size
        self.hidden      = hidden

        self.clip_size = clip_size
        self.beta      = beta

        self.basic = basic
        self.final = final

        self.thread_cpus = thread_cpus
        self.agents      = agents

        self.env         = env
        self.brain_name  = brain_name
        self.vector_size = vector

        self.quality        = quality
        self.verbosity      = verbosity
        self.small_constant = small_constant

        self.batch_size     = batch_size
        self.steps_to_train = max_try
        self.memory_deque   = self.steps_to_train * vector

        print ( 'action_size:'   , self.action_size )
        print ( 'state_size:'    , self.state_size  )
        print ( 'hidden:'        , self.hidden      )
        print ( 'clip_size:'     , self.clip_size   )
        print ( 'beta/entropyW:' , self.beta        )

        print ( 'basic activation Dense:' , self.basic )
        print ( 'final activation Actor:' , self.final )

        print ( 'thread_cpus:' , self.thread_cpus )
        print ( 'agents:'      , self.agents      )
        print ( 'brain_name:'  , self.brain_name  )

        print ( 'quality:'        , self.quality        )
        print ( 'verbosity:'      , self.verbosity      )
        print ( 'small_constant:' , self.small_constant )

        print ( 'steps_to_train:' , self.steps_to_train )
        print ( 'vector_size:'    , self.vector_size    )
        print ( 'memory_deque:'   , self.memory_deque   )
        print ( 'batch_size:'     , self.batch_size     )

        print ( 'batch_length:' , self.memory_deque // self.batch_size )
