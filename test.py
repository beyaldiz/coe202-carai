import math
import time
from queue import Queue

import numpy as np
import torch
from model import AIModel

import pygad
from pygad import torchga
from pygad.torchga import TorchGA



from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name = 'Road1/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 15)
env.reset()

behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

######################################################################################################################

actions = [np.array([[0,150,150]]), np.array([[-1,150,150]]), np.array([[1,150,150]])]
start_x, start_z = 0, 0
ALPHA, BETA = 10.0, 100.0
INIT_DRIVE_STEPS = 32
ADD_DIST_STEPS = 30
CRASH_THRESHOLD = 3.0
FINISH_THRESHOLD = 5.0
TEMPORAL_WINDOW = 16

model = AIModel(TEMPORAL_WINDOW * 5, 1, 32)
obs_queue = Queue(maxsize=TEMPORAL_WINDOW)

def dist(x1, z1, x2, z2):
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)

def set_action(action):
    env.set_actions(behavior_name, actions[action])

def init_drive():
    global obs_queue
    for i in range(INIT_DRIVE_STEPS):
        _, _, _, _, _, _, s1, s2, s3, s4, s5 = get_obs()
        if i >= INIT_DRIVE_STEPS - TEMPORAL_WINDOW:
            obs_queue.put(torch.tensor([s1, s2, s3, s4, s5]))
        set_action(0)
        env.step()

def get_obs():
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    return decision_steps.obs[0][0,:]

def reset_env():
    global start_x, start_z, obs_queue
    obs_queue = Queue(maxsize=TEMPORAL_WINDOW)
    env.reset()
    start_x, _, start_z, _, _, _, _, _, _, _, _ = get_obs()
    init_drive()

def is_crashed():
    x, _, z, _, _, _, _, _, _, _, _ = get_obs()
    return dist(start_x, start_z, x, z) < CRASH_THRESHOLD

def drive():
    global model, obs_queue
    reset_env()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()    
    x, _, z, _, _, _, _, _, _, _, _ = get_obs()
    last_pos = (x, z)

    distance, steps, finished = 0.0, 0, 0.0
    while not is_crashed():
        x, _, z, xx, _, zz, s1, s2, s3, s4, s5 = get_obs()
        if dist(x, z, xx, zz) < 5.0:
            return
        if steps % ADD_DIST_STEPS == 0:
            distance += dist(last_pos[0], last_pos[1], x, z)
            last_pos = (x, z)
        obs_queue.get()
        obs_queue.put(torch.tensor([s1, s2, s3, s4, s5]))
        obs = torch.cat(list(obs_queue.queue)).clone()
        out = model(obs)
        action = int(torch.argmax(out))
        set_action(action)
        env.step()
        steps += 1
    
    if not is_crashed():
        finished = 1.0
    
    return distance, steps, finished

dis, _, _ = drive()
print(dis)