"""
IA et Jeux - 03 Algo REINFORCE
Jan. 2022

Objectif :
  - Faire en sorte que la tige reste le plus vertical possible pendant 200 itérations. 

Etat :
  - Un vecteur de 4 valeurs indiquant : 
    - la position horizontale du chariot
    - l'angle de la tige par rapport à la verticale
    - les dérivées de ces deux grandeurs

Actions : 
  - Déplacer le chariot sur la gauche (0) ou sur la droite (1)
  - Remarquez qu'il n'est pas possible de laisser le chariot immobile !

Récompense : 
  - +1 à chaque itération

Arret :
  - Lorsque la tige fait par rapport à la verticale un angle supérieur à 12°
  - Lorsque le chariot quitte la zone de jeu.
  - Après 200 itérations

"""
import gym
import time
from pyglet.window import key
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Environnement du jeu
env = gym.make('CartPole-v0')

# Modèle
class Net(nn.Module):
    """
    Réseau de neurones en forme standard pour décrire la politique (𝜋0) avec
    deux couches Linear de 20 neurones sur la couche intermédiaire.

    L'entrée du réseau correspondra aux 4 valeurs associées à l'état du jeu
    et les sorties du réseau donnent 2 valeurs qui seront transformées en pro-
    babilités en utilisant la fonction Softmax.
    Pour déterminer quelle action va être sélectionnée suivant la politique (𝜋0),
    nous utiliserons la classe Categorical.

    Une fois la politique mise en place, on peut l'utiliser pour effectuer des si-
    mulations. Lancez le programme, à ce niveau, le contrôle va être très mauvais
    car les poids des réseaux ont été initialisés au hasard. Si cette étape fonction-
    ne, la moitié du chemin a été effectué.
    """

    def __init__(self):
        """
        Constructeur naturel.
        """
        super(Net, self).__init__()
        self.couche1 = nn.Linear(4, 20)
        self.couche2 = nn.Linear(20, 2)
        self.couche_output = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward.
        """
        # Couche input
        output = self.couche1(x)
        output = F.relu(output)

        output = self.couche2(output) # Couche intermediaire
        output = self.couche_output(output) # Output layer

        return output

    def LossV1(self, R, LogProbas):
        """
        V1 Fonction de cout.
        """
        loss_res = -(torch.FloatTensor(R) * torch.FloatTensor(LogProbas)).sum()
        loss_res.requires_grad_(True)
        return loss_res
    
    def Loss(self, values):
        """
        V2 Fonction de cout.
        """
        loss_res = values.sum()
        return loss_res
    
def GetAction(x: int) -> int:
    """
    IA basique

    Parameters
    ----------
        x (int) Position du charriot
    
    Returns
    -------
        1 (droite) si charriot à gauche sinon
        0 (gauche) si charriot à droite.
    """
    return 1 if x < 0 else 0

def pi(probas_scores):
    """
    Tire aleatoirement la prochaine action par rapport aux probabilites
    donnees.
    """
    m = torch.distributions.categorical.Categorical(probas_scores)

    # Tirage aleatoire avec probas de l'action (0 ou 1)
    index_action = m.sample()

    # Calcul de la log proba
    log_prob = m.log_prob(index_action)
    return index_action.item(), log_prob

# Parametres des simulations
RENDER: bool = True
LOG_STEP: int = 100
N_SIMULATIONS: int = int(1e4)
results: dict = {'scores': []}

# Parametres de politique
LEARNING_RATE: float = 0.01
POLICY = Net()
OPTIMIZER = torch.optim.Adam(POLICY.parameters(), lr=LEARNING_RATE)

# Main game loop
for n in range(N_SIMULATIONS):
    # Gestion d'une simulation
    done = False
    state = env.reset()
    TotalReward = []
    LogProbas = torch.FloatTensor([])
    LossTheta = 0

    env.reset()

    while not done:
        # Affiche la simulation
        if RENDER and n % LOG_STEP == 0:
          w = env.render()

        # Récupère la position du chariot
        #x = state[0]

        # Decide de l'action à effectuer
        #action = GetAction(x)
        tensor_state = torch.reshape(torch.FloatTensor(state), (1, -1))
        action_probas = POLICY(tensor_state)
        action, log_prob = pi(action_probas)
        
        # Mise à jour de l'environnement
        state, reward, done, _ = env.step(action)

        # Sauvegarde reward & log proba
        TotalReward.append(reward)
        LogProbas = torch.cat((LogProbas, log_prob), 0)

        # Frequence de rafraichissement de l'animation
        if RENDER and n % LOG_STEP == 0:
            time.sleep(0.005)
    
    # Calcul de la loss
    LossTheta = torch.sum(-torch.FloatTensor(TotalReward)) * torch.sum(LogProbas)

    # Descente du gradient
    OPTIMIZER.zero_grad()
    LossTheta.backward()
    OPTIMIZER.step()

    # Log du score
    if n % LOG_STEP == 0:
        score_reward = sum(TotalReward)
        loss_value = round(LossTheta.item(), 1)
        print(f"[{n}/{N_SIMULATIONS}] - Score({score_reward}) - Loss({loss_value})")

    results['scores'].append(sum(TotalReward))

# Affichage des resultats
SPACING: int = 50
print()
print("-" * SPACING)
print(f"| -> Paramètres".ljust(SPACING-1) + '|')
print(f'| Nombre de simulations : {N_SIMULATIONS}'.ljust(SPACING-1) + '|')
print(f"| Learning rate         : {LEARNING_RATE}".ljust(SPACING-1) + '|')
print("|".ljust(SPACING-1) + '|')
print(f"| -> Resultats".ljust(SPACING-1) + '|')
print(f"| Reward moyen          : {sum(results['scores']) / len(results['scores'])}".ljust(SPACING-1) + '|')
print("-" * SPACING)

env.close()

