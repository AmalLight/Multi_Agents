import tensorflow as tf

from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment ( file_name = "./Tennis_Linux/Tennis.x86_64" )

# get the default brain
brain_name = env.brain_names [ 0 ]
brain = env.brains [ brain_name ]

# reset the environment
env_info = env.reset ( train_mode = True )[ brain_name ]

# number of agents
num_agents = len ( env_info.agents )
print ( 'Number of agents:' , num_agents )

# size of each action
action_size = brain.vector_action_space_size
print ( 'Size of each action:' , action_size )

# examine the state space
states = env_info.vector_observations
state_size = states.shape [ 1 ]
print ( 'There are {} agents. Each observes a state with length: {}'.format ( states.shape [ 0 ] , state_size ) )
print ( 'The state for the first agent looks like:'                         , states       [ 0 ] )

# -----------------------------------------------------------------------------------

from variables import Variables

settings = Variables ( state_size = state_size , action_size = action_size , agents = num_agents ,

                       env = env , brain_name = brain_name )

# -----------------------------------------------------------------------------------

from model_Actor import Network_Actor

network_actor = Network_Actor ( settings = settings , RNN = 0 )
network_actor.model.load_weights ( 'saved_actor__model_weights_by_tensorflow.h5' )

def play_round ( env_info = None ) :

    states = env_info.vector_observations
    scores = np.zeros ( settings.agents )
    t = 0

    while True :

          states = np.array ( states )
          states = np.expand_dims ( states , 0 )

          null_actions = tf.zeros ( settings.agents , settings.action_size )

          actions , _ , _ = network_actor.model ( [ states , null_actions , [ 0 ] ] )
          actions = actions [ 0 ].numpy ()

          # print ( len ( actions ) , actions )
          # 2 [[ 0.72445965 -0.87108535] ; [ 0.36885524 -1.4574045 ]]

          env_info = settings.env.step ( actions )[ settings.brain_name ]

          next_states = env_info.vector_observations
          rewards     = env_info.rewards
          dones       = env_info.local_done
          scores     += env_info.rewards

          print ( '\rEP:'         , 1 , 't:' , t + 1 ,
                    'EP Point:'   , round ( np.mean ( scores  ) , 3 ) ,
                    'Step Point:' , round ( np.mean ( rewards ) , 3 ) , end='' )

          states = next_states ; t = t + 1

          if np.any  (  dones ) : break
    return   np.mean ( scores )

np_mean = play_round ( settings.env.reset ( train_mode = False ) [ settings.brain_name ] )

from collections import deque

scores_window = deque ( maxlen = 100 )
scores_window.append  ( np_mean      )

print ( '\nscores_window mean:' , np.mean ( scores_window ) )

env.close ()
