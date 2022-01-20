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
        x_1 = self.couche1(x)
        x_1 = F.relu(x_1)

        # Couche intermediaire
        x_2 = self.couche2(x_1)

        # Output layer
        output = self.couche_output(x_2)

        return output

    def Loss(self, R, LogProbas):
        """
        Fonction de cout.
        """
        loss_policy = list()
        
        for log_prob, reward in zip(LogProbas, R):
            loss_policy.append(-log_prob * reward)

        loss_policy = torch.cat(loss_policy).sum()

        return loss_policy
    
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
RENDER: bool = False
LOG_STEP: int = 20
N_SIMULATIONS: int = int(1e3)
results: dict = {
    'scores': []
}

# Parametres de politique
LEARNING_RATE: float = 0.015
policy = Net()
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# Main game loop
for n in range(N_SIMULATIONS):
    # Gestion d'une simulation
    done = False
    state = env.reset()
    TotalReward = list()
    LogProbas = list()
    env.reset()

    while not done:
        # Affiche la simulation
        if RENDER:
          w = env.render()

        # Récupère la position du chariot
        x = state[0]

        # TODO Decide de l'action à effectuer
        #action = GetAction(x)
        tensor_state = torch.reshape(torch.FloatTensor(state), (1, -1))
        action_probas = policy.forward(tensor_state)
        action, log_prob = pi(action_probas)
        
        # Mise à jour de l'environnement
        state, reward, done, _ = env.step(action)

        # TODO Sauvegarde reward & log proba
        TotalReward.append(reward)
        LogProbas.append(log_prob)

        # Frequence de rafraichissement de l'animation
        if RENDER:
            time.sleep(0.01)
    
    # TODO Calcul de la loss
    LossTheta = policy.Loss(TotalReward, LogProbas)

    # TODO Descente du gradient
    optimizer.zero_grad()
    LossTheta.backward()
    optimizer.step()

    # Log du score
    if n % LOG_STEP == 0:
        print(f"[{n+1}/{N_SIMULATIONS}] - Score({sum(TotalReward)}) - Loss({round(LossTheta.item(), 1)})")
    results['scores'].append(sum(TotalReward))

# Affichage des resultats
SPACING: int = 50
print("-" * SPACING)
print(f'| Nombre de simulations : {N_SIMULATIONS}'.ljust(SPACING-1) + '|')
print(f"| Learning rate         : {LEARNING_RATE}".ljust(SPACING-1) + '|')
print("|".ljust(SPACING-1) + '|')
print(f"| >> Resultats".ljust(SPACING-1) + '|')
print(f"| Reward moyen          : {sum(results['scores']) / len(results['scores'])}".ljust(SPACING-1) + '|')
print("-" * SPACING)

env.close()

