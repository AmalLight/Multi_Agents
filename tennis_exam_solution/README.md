# Notes

  1. is not possible to use "agent_example_multi_envs" from the Pong exam inside the deepL_RL folder, because unity has some limits.
  2. we use PPO because DDPG stores old actions that we can have at every new episode.


# Download the Unity Environment

  For this project, you will not need to install Unity
  this is because we have already built the environment for you,
  and you can download it from one of the links below.
  You need only select the environment that matches your operating system:

  - Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
  - Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip 
  - Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
  - Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

  Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

  (For AWS):
  If you'd like to train the agent on AWS (and have not enabled a virtual screen),
  - to enable: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md

  then please use this link to obtain the "headless" version of the environment.
  - headless: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip

  You will not be able to watch the agent without enabling a virtual screen,
  but you will be able to train the agent. (To watch the agent,
  you should follow the instructions to enable a virtual screen,
  and then download the environment for the Linux operating system above.)

# INSTALLING

  * step 1
    - 1, get: https://github.com/AmalLight/ReinforcementL_byUdacity/blob/main/deep-reinforcement-learning.rar
      or
    - 2, get: https://github.com/udacity/Value-based-methods.git

  * step 2 execute : cd 1 or 2
  * step 3 cd into python folder
  * step 4 execute : pip3 install .

    ```
    tensorflow
    Pillow
    matplotlib
    numpy
    jupyter
    pytest
    docopt
    pyyaml
    protobuf
    grpcio
    torch
    pandas
    scipy
    toml
    ```

  ps: I took the 1 from: https://github.com/udacity/deep-reinforcement-learning
