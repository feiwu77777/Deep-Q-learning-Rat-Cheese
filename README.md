# Deep Q-Learning for the rat&cheese game

This project is an exercice from the [MVA](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/) course, more specifically the [course](https://www.labri.fr/perso/vlepetit/deep_learning_mva.php) about Reinforcement Learning (lecture 4 and the mini-project) taught by [Vincent Lepetit](https://www.labri.fr/perso/vlepetit/index.php).

***Summary***
1. [Game Rules](https://github.com/feiwu77777/Deep-Q-learning-Rat-Cheese#1-game-rules)
2. [Deep Q-Learning Approach](https://github.com/feiwu77777/Deep-Q-learning-Rat-Cheese#2-deep-q-learning-approach)
3. [Results](https://github.com/feiwu77777/Deep-Q-learning-Rat-Cheese#3-results)

## 1. Game Rules
- A 13x13 grid is initialized with edible cheese (red) and poisonous cheese (blue). 
- A rat (white) has to collect all the edible cheese by avoiding as much as possible the poisonous ones (he can still eat some). 

 <img src="https://user-images.githubusercontent.com/34350063/78794965-b1115a00-79b4-11ea-8b01-c1cdd2c21d96.png" width="250" height="250">

- The rat can only see as far as 2 cells, he needs to explore the environment to find all the cheese.

 <img src="https://user-images.githubusercontent.com/34350063/78923920-3c641b80-7a99-11ea-920d-1aedf4eb4354.png" width="150" height="150">

- There is a time limit after which the game is over, the rat can move by one cell (up, down , left, right) at each step. 
- The score of a game (initially 0) is udpated with the following rule: collecting an edible cheese grants 0.5 point and collecting a poisonous one takes off 1 point.


## 2. Deep Q-Learning Approach
- The rat (agent) will learn what is the best action to take in a given state according to the value of a Q-value function. 
- The Q-value function takes as inputs a state **S** and an action **A** and output a value representing the expected culmunative reward throughout the game for taking the specific action **A** when the state was **S**. Given a state **S**, the rat have to choose the action **A** that maximize this Q-value function.
- The Q-value function will be modeled by a neural network, it will take as input a state and output 4 values, one for each of the available actions.

 <img src="https://user-images.githubusercontent.com/34350063/78794756-6d1e5500-79b4-11ea-95a5-bed351dd091a.png" width="500" height="300">

## 3. Results
- The Q-value neural network is then trained for several episodes and the time limit is set to 200 steps so the rat has to collect all edible cheese within 200 steps by avoiding as much as possible the poisonous ones.
- Here are some winning games achieved by an agent that choose actions according to this Q-value function (additional games are available in mp4 format inside the video/test folder).

<p float="center">
  <img src="https://user-images.githubusercontent.com/34350063/78797480-09962680-79b8-11ea-87b9-0148a3e3d75a.gif" width="300" />
  <img src="https://user-images.githubusercontent.com/34350063/78798046-d2744500-79b8-11ea-8789-de8b7c632472.gif" width="300" /> 
  <img src="https://user-images.githubusercontent.com/34350063/78798275-16674a00-79b9-11ea-8721-ff28f711fd83.gif" width="300" />
</p>
