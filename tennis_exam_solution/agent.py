import tensorflow             as tf
import tensorflow_probability as tfp
import numpy                  as np

import random , sys , threading , math , time , gym

from unityagents import UnityEnvironment
from collections import deque

from model_Critic import Network_Critic
from model_Actor  import Network_Actor
from replay       import ReplayBuffer

# ---------------------------------------------------------------------------

class Agent () :

    def __init__ ( self , settings ) :
        self.settings   = settings
        self.collection_states      = []
        self.collection_next_states = []
        self.collection_continuosV  = []
        self.collection_dones       = []
        self.collection_rewards     = []
        self.collection_values      = []
        self.collection_probs       = []

        self.collection_rewards_step = np.zeros ( 1 )

    def play_EP ( self , ie = 0 , env_info = None ) :

        self.collection_rewards_sum = np.zeros ( 1 )

        states = np.array ( env_info.vector_observations )

        for t in range ( self.settings.steps_to_train ) :

            states = np.array       ( states     )
            states = np.expand_dims ( states , 0 )

            null_actions = tf.zeros ( self.settings.agents , self.settings.action_size )

            values                            = self.network_critic.model (   states                          ) [ 0 ].numpy () # ( 2 , 1 )
            continuosV , old_log_prob_sum , _ = self.network_actor .model ( [ states , null_actions , [ 0 ] ] )

            continuosV       = continuosV       [ 0 ].numpy () # ( 2 , 2 )
            old_log_prob_sum = old_log_prob_sum [ 0 ].numpy () # ( 2 , 1 )

            # ------------------------------------------------------

            env_info = self.settings.env.step ( continuosV ) [ self.settings.brain_name ]

            next_states = np.array ( env_info.vector_observations ) # ( 2 , 24 )
            rewards     = np.array ( env_info.rewards             ) # ( 2 ,  1 )
            dones       = np.array ( env_info.local_done          ) # ( 2 ,  1 )

            # ------------------------------------------------------

            states = np.squeeze ( states , 0 )

            self.collection_continuosV. append ( continuosV       ) # append on tail
            self.collection_states.     append ( states           ) # append on tail
            self.collection_next_states.append ( next_states      ) # append on tail
            self.collection_dones.      append ( 1 - dones        ) # append on tail
            self.collection_values.     append ( values           ) # append on tail
            self.collection_probs.      append ( old_log_prob_sum ) # append on tail

            self.collection_rewards_step += max ( rewards )
            self.collection_rewards_sum  += max ( rewards )

            self.collection_rewards.append ( rewards ) # append on tail

            states = next_states

            print ( '\rEP:'   , ie + 1 , 't:' , t + 1 ,
                      'EP:'   , round ( np.mean ( self.collection_rewards_sum  ) , 3 ) ,
                      'Step:' , round ( np.mean ( self.collection_rewards_step ) , 3 ) ,
                      'MEM1:' ,   len (           self.collection_states       )       ,
                      'MEM2:' ,                   self.replay.len_memory      ()       , end='' )

            # ------------------------------------------------------

            full_memory = ( ( self.replay.len_memory () + (t+1) ) == self.settings.memory_deque )

            if ( ( t + 1 ) % self.settings.steps_to_train == 0 ) or full_memory :

               real_steps_to_train = np.array ( self.collection_states ).shape [ 0 ]

               self.states_elite      = np.zeros ( ( real_steps_to_train , self.settings.agents , self.settings.state_size  ) )
               self.next_states_elite = np.zeros ( ( real_steps_to_train , self.settings.agents , self.settings.state_size  ) )
               self.continuosV_elite  = np.zeros ( ( real_steps_to_train , self.settings.agents , self.settings.action_size ) )
               self.rewards_elite     = np.zeros ( ( real_steps_to_train , self.settings.agents                             ) )
               self.dones_elite       = np.ones  ( ( real_steps_to_train , self.settings.agents                             ) )
               self.values_elite      = np.zeros ( ( real_steps_to_train , self.settings.agents , 1                         ) )
               self.probs_elite       = np.zeros ( ( real_steps_to_train , self.settings.agents , 1                         ) )

               if full_memory : print ( '' )
               self.training_for_elite ( full_memory )
               if full_memory : print ( '' )

               self.collection_states      = []
               self.collection_next_states = []
               self.collection_continuosV  = []
               self.collection_dones       = []
               self.collection_rewards     = []
               self.collection_values      = []
               self.collection_probs       = []

               self.collection_rewards_step = np.zeros ( 1 )

               if full_memory : self.replay.destroy ()
            if    full_memory : break # any ( dones )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def advantage_rewards ( self , rewards , last_missing_prediction ) :

        # TD_sarsa_next = alpha * ( reward + (gamma * new_state_action * dones) - old_Qvalue )
        # Q [ state ] [ action ] = ( old_Qvalue + sarsa )

        advantage_rewards = np.zeros ( rewards.shape )

        for row in reversed ( range ( advantage_rewards.shape [ 0 ] ) ) :

            old_TD_state_action = self.values_elite [ row     ]
            new_TD_state_action = self.values_elite [ row + 1 ] \
                                  \
                                  if ( ( (row+1) % advantage_rewards.shape [ 0 ] ) > 0 ) \
                                  \
                                  else last_missing_prediction # recursion

            new_TD_state_action *= ( self.dones_elite [ row ] * 0.99 ) # (X*N)+(0.99**N*X)

            error   = rewards [ row ] + new_TD_state_action - old_TD_state_action
            advantage_rewards [ row ] = error

            # for row in range ( advantage_rewards.shape [ 0 ] ) : advantage_rewards [ row , : ] *= ( 0.95 ** row )

            advantage_rewards [ row ] += ( advantage_rewards [ row + 1 ] * self.dones_elite [ row ] * 0.94 ) \
                                         \
                                         if ( ( (row+1) % advantage_rewards.shape [ 0 ] ) > 0 ) else 0 # no sum before if for reversed and dones_elite

        # --------------------------------------------------------------

        advantage_rewards_mean = np.mean ( advantage_rewards , axis = 1 , keepdims = True )
        advantage_rewards_std  = np.std  ( advantage_rewards , axis = 1 , keepdims = True )

        advantage_rewards -= advantage_rewards_mean
        advantage_rewards /= advantage_rewards_std
        return advantage_rewards

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def future_rewards ( self , rewards , last_missing_prediction ) :

        self.dones_elite = np.expand_dims ( self.dones_elite , 2 )

        for row in  reversed ( range ( rewards.shape  [ 0 ] ) ) :
            rewards_next           = ( rewards  [ row + 1 ] ) \
                                     \
                                     if ( ( (row+1) % rewards.shape  [ 0 ] ) > 0 ) \
                                     \
                                     else ( last_missing_prediction )

            # ( X * N ) + ( 0.99**N * X )
            rewards_next    *= ( self.dones_elite [ row ] * 0.99 )
            rewards [ row ] +=        rewards_next

        return rewards

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training_for_elite ( self , to_train = False ) :

        for i , e in enumerate ( range ( self.settings.agents ) ) :

            self.states_elite      [ : , i ] = np.array ( self.collection_states      ) [ : , e ]
            self.next_states_elite [ : , i ] = np.array ( self.collection_next_states ) [ : , e ]
            self.continuosV_elite  [ : , i ] = np.array ( self.collection_continuosV  ) [ : , e ]
            self.rewards_elite     [ : , i ] = np.array ( self.collection_rewards     ) [ : , e ]
            self.dones_elite       [ : , i ] = np.array ( self.collection_dones       ) [ : , e ]
            self.values_elite      [ : , i ] = np.array ( self.collection_values      ) [ : , e ]
            self.probs_elite       [ : , i ] = np.array ( self.collection_probs       ) [ : , e ]

        # --------------------------------------------------------------

        last_next_states = np.array       ( self.next_states_elite [ -1 ] )
        last_next_states = np.expand_dims ( last_next_states       ,  0   )

        last_missing_prediction = self.network_critic.model (last_next_states ) [ 0 ].numpy ()

        self.rewards_elite = np.expand_dims ( self.rewards_elite , 2 )

        self.future_rewards_elite    = self.future_rewards    ( self.rewards_elite.copy () , last_missing_prediction )
        self.advantage_rewards_elite = self.advantage_rewards ( self.rewards_elite.copy () , last_missing_prediction )

        self.training ( self.future_rewards_elite , to_train )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def thread_batch_optimizer ( self , iter_step , total ) :

        stats , acts , futurs , probs , advs = self.replay.sample ()

        with tf.GradientTape ( persistent = False ) as tape_values :
             tape_values.reset ()

             value_predictions = self.network_critic.model ( stats )
             loss_values = tf.math.reduce_mean ( tf.math.pow ( futurs - value_predictions , 2 ) ) * 0.5

        with tf.GradientTape ( persistent = False ) as tape_actions :
             tape_actions.reset ()

             _ , new_log_prob_sum , entropy_sum = self.network_actor.model ( [ stats , acts , [ 1 ] ] )

             # old_log_prob_sum = tf.math.exp (                            probs    )
             # new_log_prob_sum = tf.math.exp ( new_log_prob_sum                    ) # log_prob must be log based on 2 or e
             # ratio            =             ( new_log_prob_sum / old_log_prob_sum ) # >= 0
             ratio              = tf.math.exp ( new_log_prob_sum -         probs    ) # >= 0
             # ratio_min = ratio

             ratio_min = tf.math.reduce_min   (
                         tf.convert_to_tensor ( [ ratio                                                  * advs , \
                         tf.clip_by_value     (   ratio , clip_value_min = 1 - self.settings.clip_size ,
                                                          clip_value_max = 1 + self.settings.clip_size ) * advs ] ) , axis = 0 )

             loss_actions = - tf.math.reduce_mean ( ratio_min + self.settings.beta * entropy_sum )

        quality_actions = tf.math.reduce_mean ( probs - new_log_prob_sum )

        if quality_actions <= 1.5 * self.settings.quality :
                                    self.network_actor.optimizer.minimize \
                                    \
                                               ( loss_actions , var_list = self.network_actor .model.trainable_variables , tape = tape_actions )
        self.network_critic.optimizer.minimize ( loss_values  , var_list = self.network_critic.model.trainable_variables , tape = tape_values  )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def training ( self , future_rewards , to_train = False ) :

        self.collection_states     = self.states_elite
        self.collection_values     = self.values_elite
        self.collection_probs      = self.probs_elite
        self.collection_continuosV = self.continuosV_elite

        # ------------------------------------------------------------------------------------------

        for i in range ( self.collection_states.shape [ 0 ] ) :

            self.replay.add ( self.collection_states       [ i ] ,
                              self.collection_continuosV   [ i ] ,
                                   future_rewards          [ i ] ,
                              self.collection_probs        [ i ] ,
                              self.advantage_rewards_elite [ i ] ,
                              self.collection_values       [ i ] ) # append on tail

        total = self.settings.verbosity * self.settings.batch_size

        if not to_train : return 0

        iter_step = 0 ; print ( '\r' , iter_step , '/' , total , end='' , sep='' )

        # ------------------------------------------------------------------------------------------

        for v in range ( self.settings.verbosity ) :

            self.replay.shuffle_simple ()

            self.replay.reset_memory_i ()

            threads = [ None ] * self.settings.thread_cpus

            for i in range ( self.settings.batch_size ) :
                iter_step += 1
                i_thread_cpus = ( i + 1 ) % self.settings.thread_cpus

                threads [ i_thread_cpus ] = threading.Thread ( target = self.thread_batch_optimizer , args = ( iter_step , total , ) )
                threads [ i_thread_cpus ].start ()

                if ( ( ( i + 1 ) % self.settings.thread_cpus ) == 0 ) or ( ( i + 1 ) == self.settings.batch_size ) :

                   for i_thread_cpus in range ( self.settings.thread_cpus ) :

                       if threads [ i_thread_cpus ] != None : threads [ i_thread_cpus ].join ()

                   threads = [ None ] * self.settings.thread_cpus

            print ( '\r' , 'beta:' , round ( float ( self.settings.beta      ) , 7 ) ,
                          ' MEM2:' ,                 self.replay.len_memory () ,
                          ' iter:' , iter_step , '/' , total                   , end = '' , sep = '' )

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    def play_round ( self , env_info = None ) :

        states = env_info.vector_observations
        scores = np.zeros ( self.settings.agents )

        while True :

              states = np.array       ( states     )
              states = np.expand_dims ( states , 0 )

              null_actions = tf.zeros ( self.settings.agents , self.settings.action_size )

              actions , _ , _ = self.network_actor.model ( [ states , null_actions , [ 0 ] ] )
              actions = actions [ 0 ].numpy ()

              env_info = self.settings.env.step ( actions )[ self.settings.brain_name ]

              next_states = env_info.vector_observations
              rewards     = env_info.rewards
              dones       = env_info.local_done
              scores     += env_info.rewards

              states = next_states

              if np.any  (  dones ) : break
        return   np.mean ( scores )

    def cross_entropy_loss ( self ) :

        self.replay         = ReplayBuffer   ( settings = self.settings           )
        self.network_critic = Network_Critic ( settings = self.settings , RNN = 0 )
        self.network_actor  = Network_Actor  ( settings = self.settings , RNN = 0 )
        self.start = -1

        ie , red_flag  = 0 , False
        best_result    =   - np.inf
        scores_window  =     deque ( maxlen = 100 )

        for i , w in enumerate ( self.network_critic.model.weights ) : print ( 'el i:' , i , 'shape critic:' , w.shape )
        for i , w in enumerate ( self.network_actor .model.weights ) : print ( 'el i:' , i , 'shape actor:'  , w.shape )

        while not red_flag :
              print ( '' )

              self.play_EP              ( ie , self.settings.env.reset ( train_mode = True ) [ self.settings.brain_name ] )
              np_mean = self.play_round (      self.settings.env.reset ( train_mode = True ) [ self.settings.brain_name ] )

              if ie > self.start : scores_window.append ( np_mean ) # np.mean ( self.collection_rewards_sum )
              if ie > self.start : print ( '\nscores_window mean:' , np.mean ( scores_window ) )

              if        ie > self.start and best_result < np.mean ( scores_window ) :
                     if ie > self.start   : best_result = np.mean ( scores_window )

                     self.network_critic.model.save_weights ( 'saved_critic_model_weights_by_tensorflow.h5' )
                     self.network_actor .model.save_weights ( 'saved_actor__model_weights_by_tensorflow.h5' )

                     print ( 'saved_models_weights_by_tensorflow best:' , best_result )
              else : print ( 'best_scores_window mean:'                 , best_result )

              self.settings.beta = max ( self.settings.beta * 0.995 , 0.0001 ) # block the entropy if it is reduced to 0

              ie = ie + 1
