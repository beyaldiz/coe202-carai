import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()

env = UnityEnvironment(file_name = 'Road1/Prototype 1', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 15)

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]
x,y,z,xx,yy,zz,fr,r,fl,l,f = cur_obs
px,py,pz,pxx,pyy,pzz,pfr,pr,pfl,pl,pf = cur_obs

# function for moving forward/backward
def move(n):
    if(n < 0):
        for i in range(-n):
            env.set_actions(behavior_name, np.array([[0,-100,-100]]))
            env.step()
        return
    for i in range(n):
        env.set_actions(behavior_name, np.array([[0,100,100]]))
        env.step()
        
# function for moving left/turning left
def turn_left(n):
    if(n < 0):
        for i in range(-n):
            env.set_actions(behavior_name, np.array([[-1,-100,100]]))
            env.step()
        return
    for i in range(n):
        env.set_actions(behavior_name, np.array([[-0.3,100,100]]))
        env.step()

# function for moving right/turning right
def turn_right(n):
    if(n < 0):
        for i in range(-n):
            env.set_actions(behavior_name, np.array([[1,100,-100]]))
            env.step()
        return
    for i in range(n):
        env.set_actions(behavior_name, np.array([[0.3,100,100]]))
        env.step()
def update():
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    return(cur_obs)



while(True):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    x,y,z,xx,yy,zz,fr,r,fl,l,f = cur_obs
    print("cur observations : ", decision_steps.obs[0][0,:])

    # case that the car is near the destination
    if(abs(x - xx) < 10 and abs(z -zz) < 10):
        move(1)
        continue

    # case that we can go straight or go long left or go long right
    while(fr > 10 and fl > 10 and f > 10 and r > 4 and l > 4):
        if(fr > 18 and fl > 18):
            move(1)
        elif(fr+r > fl+l):
            turn_right(1)
        else:
            turn_left(1)
        x,y,z,xx,yy,zz,fr,r,fl,l,f = update()


    # case that we stuck (super near the object) and need to move back
    if((fl < 5 and fr < 5 and f < 5)):
        for i in range(10):
            if(fr+r < fl+l):
                turn_right(-1)
            else:
                turn_left(-1)
            move(-2)

    # case that we not stuck that much, just turn left or turn right
    elif(fl < 7 or fr < 7 or f < 7):
        if(fl+l < fr+r):
            turn_right(-1)
        else:
            turn_left(-1)

    # case that on the sides of the car, there are objects that might the car might hit
    elif(r < 4 or l < 4):
        if(l < 3 or r < 3):
            for i in range(20):
                if(r < l):
                    turn_right(-1)
                else:
                    turn_left(-1)
                move(-2)
        elif(l < r):
            turn_right(-1)
        else:
            turn_left(-1)

    # other case (just move)
    else:
        move(1)

  
    env.step()
    
        

env.close()
