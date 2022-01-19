import gym
import time
from pyglet.window import key

env = gym.make('CartPole-v0')

# appuyez sur la barre espace pour aller à droite
# par  défaut le chariot va vers la gauche
# ESC : quitter
# lorsque la tige est trop penchée, le jeu recommence automatiquement

Key = False
Loop = True

def key_press(k, mod):
    global Key, Loop
    if k == key.SPACE  : Key = True
    if k == key.ESCAPE : Loop = False

def key_up(k,mod):
    global Key
    if k == key.SPACE  : Key = False


# init game window
env.reset()
env.render()
env.viewer.window.on_key_press    = key_press
env.viewer.window.on_key_release  = key_up

def GetAction(env):
    if Key : return 1
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
        time.sleep(0.3)

    env.reset()
    print("Score final : " , TotalScore)

env.close()