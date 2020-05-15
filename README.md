## Collaboration and competition

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.
The action space is continuous [-1.0, +1.0] and consists of 2 values for horizontal and jumping moves. <br/>
The state space is also continous with 24 vectors.<br />

This repository contains an implementation of deep reinforcement learning based on Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm based on <link>

### Contents :
* Report.md
*  __agents__ : contains the agents definition and related utilities
    ** model.py : Class definition of Actor and Critic networks
    ** maddpg.py : Implements the MADDPG algorithm based on the paper cited above
    ** utils.py  : contains the required utility function and classes like noise, memory buffer, etc.
*  __final_weights__ : contains the final stored weights of the agent
* Multi Agent Deep Deterministic Policy Gradient.ipynb : Primary notebook from where agent is trained and executed. Contains the last executed instance as well.


### Requirements
To run the codes, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
    ```bash
    conda create --name ddpg python=3.6
    activate ddpg
	```
	* __Windows__: 
	```bash
	conda create --name ddpg python=3.6 
	activate ddpg
	```
* Follow the installation of the AI-RL repository here(https://github.com/udacity/deep-reinforcement-learning) with certain modifications to the requirements. In the root folder of the repo, find requirements.txt in the python folder, change the following (some are given below as well)
    * tensorflow==1.7.1 to tensorflow==1.15
    * torch==0.4.0 to torch

* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install PyTorch
    ```bash
    pip install pytorch
    ```


* Install jupyter notebook or jupyter lab
```bash
	pip install jupyter notebook jupyterlab
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
* Start jupyter notebook from the root of this python codes
```bash
jupyter lab
```
* If necessary, inside the ipynb files, change the path to the unity environment appropriately

