# Python Implementation of Population Based Training (PBT)
## Introduction:
[PBT](https://arxiv.org/abs/2003.06212) is a Model-Based training algorithm that uses an algorith that is similar to alpha-zero and its implementation of monte carlo tree search to train neural network agents, this method consist of 3 phases per iteration which are:
### 1 - Collecting Data :
* Agents play against each other or against themselves other to collect shared training data and results.
### 2 - Training:
* Each agent train using the shared data with different hyperparameteres than other agents.
### 3 - Evaluation:
* Each agent play against all other agents a number of matches per opponents alternating between side (Black or White) ([Round Robin](https://en.wikipedia.org/wiki/Round-robin_tournament)).
* A copy of Top 20% of the agents will replace lowest 20%, including network parameters, however each of the hyperparameters has a chance to be multiplied by 1.2 or 0.8(devided by 1.2 in this implementation) allowing us to get rid of the least suffecient training hyperparameters while have the ability to explore better hyperparameters candidates.
* In this implementation differs from the Paper by keeping track of the strongest agent so far by matching the strongest agents each iteration against the current strongest agent so far.

## Requirements
* Python 3.9.7 environment.
* git.

## Steps
* Download or pull this github repo
    >git pull ***
* Install python libraries using requirements.txt file
    >pip install -r requirements.txt



