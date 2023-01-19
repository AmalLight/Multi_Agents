import tensorflow as tf
import numpy      as np
import threading

from collections import deque
import random

class ReplayBuffer () :

    def __init__     ( self , settings ) :
        self.settings       = settings
        self.maxlen         = settings.memory_deque

        self.memory   = deque ( maxlen = self.maxlen )
        self.memory_i = 0

    def add                ( self , states , actions , rewards , old_probs , advantages , values ) :
        self.memory.append (      ( states , actions , rewards , old_probs , advantages , values ) )

    def len_memory     ( self ) : return     len ( self.memory )
    def reset_memory_i ( self ) :                  self.memory_i = 0
    def destroy        ( self ) :                  self.memory = deque ( maxlen = self.maxlen )
    def shuffle_simple ( self ) : random.shuffle ( self.memory )
    def shuffle        ( self ) :

        for_each_batch_lenght = ( self.len_memory () // self.settings.batch_size )

        memory_tmp = deque ( maxlen = self.maxlen )

        range_step_cpus = list ( range ( 0 , self.len_memory () , for_each_batch_lenght ) )

        for i in range ( len ( range_step_cpus ) ) :

            id_start = random.sample ( range_step_cpus , 1 ) [ 0 ]
            range_step_cpus.remove ( id_start )

            for add_i in range ( id_start , id_start + for_each_batch_lenght ) :
                memory_tmp.append ( self.memory [ add_i ] )

        self.memory = None
        self.memory = memory_tmp

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def sample ( self ) :

        batch_return = []
        for batch in range ( self.len_memory () // self.settings.batch_size ) :

            batch_return  += [ self.memory [ self.memory_i ] ]
            self.memory_i += 1

        # random.shuffle ( batch_return )

        states , actions , rewards , probs , advantages , _ = zip ( * batch_return )

        return ( tf.convert_to_tensor ( states     , dtype = float ) , tf.convert_to_tensor ( actions , dtype = float ) ,
                 tf.convert_to_tensor ( rewards    , dtype = float ) , tf.convert_to_tensor ( probs   , dtype = float ) ,
                 tf.convert_to_tensor ( advantages , dtype = float ) )
