import gym
import time
from pyglet.window import key

env = gym.make('CarRacing-v0')


print("Nb actions : ", env.action_space.n)


Key = [False,False,False]
Loop = True

def key_press(k, mod):
    global Key, Loop
    if k == key.LEFT  : Key[0] = True
    if k == key.RIGHT : Key[1] = True
    if k == key.UP    : Key[2] = True
    if k == key.ESCAPE : Loop = False


def key_up(k,mod):
    global Key
    if k == key.LEFT  : Key[0] = False
    if k == key.RIGHT : Key[1] = False
    if k == key.UP    : Key[2] = False

# init game window
env.reset()
env.render()
env.viewer.window.on_key_press    = key_press
env.viewer.window.on_key_release  = key_up

def GetAction(env):
    if Key[0] : return 1
    if Key[1] : return 3
    if Key[2] : return 2
    return 0

# Main game loop

while Loop :
    done = False
    observation = env.reset()
    TotalScore = 0

    while not done and Loop:
        w = env.render()
        action = GetAction(env)
        observation, reward, done, info = env.step(action)
        TotalScore += reward
        time.sleep(0.02)

    env.reset()
    print("Score : " , TotalScore)

env.close()