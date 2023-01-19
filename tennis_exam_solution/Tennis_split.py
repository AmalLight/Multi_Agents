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

from agent import Agent
future =          Agent ( settings )

future.cross_entropy_loss ()
env.close                 ()
