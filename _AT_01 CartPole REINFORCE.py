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

# Environnement du jeu
env = gym.make('CartPole-v0')

def politique():
    """TODO
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
    pass

def compute_loss():
    #
    """TODO
    Calcul la loss.
    """
    pass

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

# Parametres des simulations
N_SIMULATIONS: int = 10
LEARNING_RATE: float = 0.01
#optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
results: dict = {
    'scores': []
}

# Main game loop
for n in range(N_SIMULATIONS):
    # Gestion d'une simulation
    done = False
    state = env.reset()
    TotalScore = 0
    env.reset()

    while not done :
        # Affiche la simulation
        w = env.render()

        # Récupère la position du chariot
        x = state[0]

        # TODO Decide de l'action à effectuer
        action = GetAction(x)
        # action, log_prob = politique(x)
        
        # Mise à jour de l'environnement
        state, reward, done, _ = env.step(action)

        # TODO Sauvegarde reward & log proba
        TotalScore += reward

        # Frequence de rafraichissement de l'animation
        time.sleep(0.01)
    
    # TODO Calcul de la loss
    ...

    # TODO Descente du gradient
    #optimizer.zero_grad()
    #Loss.backward()
    #optimizer.step()

    # Log du score
    print(f"[{n+1}/{N_SIMULATIONS}] - Score final : {TotalScore}")
    results['scores'].append(TotalScore)

# Affichage des resultats
print()
print("-" * 30)
print(f'Nombre de simulations : {N_SIMULATIONS}')
print(f"Taux d'apprentissage  : {LEARNING_RATE}")
print()
print(f"Score moyen           : {sum(results['scores']) / len(results['scores'])}")
print("-" * 30)

env.close()

