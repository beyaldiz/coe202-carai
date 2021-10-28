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

actions = [np.array([[0,150,150]]), np.array([[-0.3,150,150]]), np.array([[0.3,150,150]])]
start_x, start_z = 0, 0
ALPHA, BETA = 10.0, 100.0
INIT_DRIVE_STEPS = 32
ADD_DIST_STEPS = 30
CRASH_THRESHOLD = 3.0
FINISH_THRESHOLD = 5.0
TEMPORAL_WINDOW = 16
best_fitness = 0.0

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
    global env, channel
    env.reset()
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
    x, _, z, _, _, _, _, _, _, _, _ = get_obs()
    last_pos = (x, z)

    distance, steps, finished = 0.0, 0, 0.0
    while not is_crashed():
        x, _, z, xx, _, zz, s1, s2, s3, s4, s5 = get_obs()
        if dist(x, z, xx, zz) < 5.0:
            break
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

def fitness_fn(solution, sol_idx):
    global model
    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    distance, steps, finished = drive()
    print(f"Dist: {distance}, Steps: {steps}, Finished: {finished}")
    return distance + ALPHA * (distance / steps) + BETA * finished

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    fitness = ga_instance.best_solution()[1]
    global best_fitness
    if fitness > best_fitness:
        best_fitness = fitness
        solution, _, _ = ga_instance.best_solution()
        best_solution_weights = torchga.model_weights_as_dict(model=model, weights_vector=solution)
        model.load_state_dict(best_solution_weights)
        torch.save(model.state_dict(), 'model_weights.pth')


torch_ga = TorchGA(model=model, num_solutions=20)

num_generations = 100 # Number of generations.
num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "adaptive"#"random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_fn,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_probability=[0.2, 0.03],
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_solution_weights = torchga.model_weights_as_dict(model=model, weights_vector=solution)
model.load_state_dict(best_solution_weights)
torch.save(model.state_dict(), 'model_weights.pth')