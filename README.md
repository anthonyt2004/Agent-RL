# Agent Collecteur de Fromage (RL)

Ce projet implémente et compare deux algorithmes d'apprentissage par renforcement (Reinforcement Learning) pour entraîner un agent (une souris) à collecter du fromage tout en évitant du poison dans une grille 2D.

## Algorithmes implémentés
* **Q-Learning (Value-based) :** Utilise une table Q pour apprendre la valeur de chaque action dans chaque état.
* **REINFORCE (Policy Gradient) :** Utilise une approche basée sur le gradient de politique pour optimiser directement les probabilités d'action de l'agent.

## Fonctionnalités
* **Environnement dynamique :** Grille générée aléatoirement avec du poison et différentes quantités de fromage.
* **Visualisation Pygame :** Une interface graphique permet de voir l'agent s'entraîner puis exécuter sa politique en temps réel.
* **Récompenses :** L'agent reçoit des bonus pour le fromage, des malus pour le poison, et un petit bonus de proximité pour l'encourager à explorer.

## Utilisation
Pensez à créer un environnement virtuel (via ```python -m venv venv```) et à l'activer avant d'installer les dépendances avec la commande ```pip install -r requirements.txt```.

Pour lancer le projet, il vous suffit d'exécuter le fichier principal [main.py](main.py) :

```bash
python main.py
```

### Changer l'algorithme

Par défaut, le script utilise l'un des deux algorithmes (Q-Learning ou Policy Gradient). Si vous souhaitez tester l'autre, ouvrez le fichier main.py et décommentez la ligne correspondant à l'algorithme voulu.

#### Pour REINFORCE (Policy Gradient) :
```visualize(learn_reinforce, policy)```

#### Pour Q-Learning :
```visualize(learn_q_table, greedy_policy)```

---
### Personnalisation

Il est possible de modifier les paramètres de la grille (taille, probabilité de poison) ainsi que les hyperparamètres de l'algorithme (taux d'apprentissage, nombre d'itérations, etc.) en modifiant directement les valeurs dans le fichier [config.py](config.py).
