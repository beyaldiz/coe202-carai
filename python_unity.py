import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()

env = UnityEnvironment(file_name = 'Road1/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 3)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

for i in range(100):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("cur observations : ", decision_steps.obs[0][0,:])
    # Set the actions
    env.set_actions(behavior_name, np.array([[0,150,150]]))
    # Move the simulation forward
    env.step()

env.close()
