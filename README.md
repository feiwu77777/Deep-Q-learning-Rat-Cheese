# Deep Q-Learning for the rat&cheese game

This project is an exercice from the [MVA](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/) course, more specifically the [course](https://www.labri.fr/perso/vlepetit/deep_learning_mva.php) about Reinforcement Learning (lecture 4 and the mini-project) taught by [Vincent Lepetit](https://www.labri.fr/perso/vlepetit/index.php).

1. Game Rules
2. Deep Q-Learning approach
3. Results

## Game Rules
A 13x13 grid is initialized with edible cheese (red) and poisonous cheese (blue). A rat (white) that appear on that grid has to collect all the edible cheese by avoiding as much as possible the poisonous ones (he can still eat some). There is a time limit after which the game is over, the rat can move by one cell (up, down , left, right) at each step. The score of a game initialized at 0 is udpated with the following rule: collecting an edible cheese grants 0.5 point and collecting a poisonous one takes off 1 point.
![Screenshot from 2020-04-08 16-18-33](https://user-images.githubusercontent.com/34350063/78794965-b1115a00-79b4-11ea-8b01-c1cdd2c21d96.png){:height="300px" width="300px"}
## Deep Q-Learning approach
- The rat (agent) will learn what is the best action to take in a given state according to the value of a Q-value function. The Q-value function    takes as inputs a state S and an action A and output a value representing the expected culmunative reward throughout the game for taking the specific action A when the state was S. Given a state S, the rat have to choose the action A that maximize this Q-value function.
- The Q-value function will be modeled by a neural network, it will take as input a state and output 4 values, one for each of the available actions.
![Screenshot from 2020-04-08 16-16-23](https://user-images.githubusercontent.com/34350063/78794756-6d1e5500-79b4-11ea-95a5-bed351dd091a.png)
