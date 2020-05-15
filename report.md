


# Report : Collaboration and Competition : Solving the tennis environment

### Index
1. Problem statement
2. Algorithm
3. Experiment and results
4. Next Steps


### 1. Problem Statement

##### Environment and Agent:
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.


__Both state and action space__ are real valued, which makes this environment challenging to solve with traditional methos

##### Goal: 
The agents must hit the ball so that the opponent cannot hit a valid return. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5. Develop a reinforcment learning based agent to control the racket and solve the environment

### 2. Algorithm

#### Actor Critic Approach
This approaches uses two function approximator namely an Actor and a Critic.
* Actor learns a stochastic policy by using a Monte-Carlo approach. It decides the probability of action(s) to take.  
* Critic learns the value function. It is used to tell whether the actions taken by the Actor are good or not.
Proess is as follows

1. Observe state $s$ from environment 
2. Use Actor to get action distribution $\pi(a|s;\theta_\pi)$. Stochasticially select one action stochastically and feed back to the environment.  
3. Observe next state $s'$ and reward $r$.  
4. Use the tuple $(s, a, r, s')$ for the TD estimate $y=r + \gamma V(s'; \theta_v)$
5. Use critic loss as $L=(y - V(s;\theta_v)^2$ for training the Critic network to minimize it 
6. Calculate the advantage $A(s,a) = r + \gamma V(s'; \theta_v) - V(s; \theta_v)$ and use it to traing the Actor


#### MADDPG

**Multi Agent Deep Deterministic Policy Gradient (DDPG)<a href='https://arxiv.org/abs/1706.02275'>Ryan Lowe et al. </a>** is an off policy algorithm that uses the above explained Actor-Critic. It adopts a framework of centralized training and decentralized execution. For more details, please refer to the link.
Basic algorithm looks like this 
- Initialize both local and target actor-critic network 
- Copy local weights into the target networks
- Initialize a replay buffer
- Repeat the next steps until the environment is solved
    - Initialize a noise generator to disturb the output of the local actor network
    - Receive initial observations state s1 of each agent
    - Do until one episode is done
    - Generate the actions of the agents from the local actor network and disturb it with the noise generator
    - Execute the actions and observe the rewards and the new states
    - To promote collaboration, ensure that the rewards of the agents are the sum of the rewards received by all agents
    - At each update frequency, update the local networks as per the pseudocode described below
    - Perform a soft network update to the target network

The MADDPG pseudocode:

<img src='img/maddpg.png' height='200'/>

Bootstrapping is also implemented as the rewards are scarce. Without bootstrapping, the convergence was very slow.

Hyperparameters choses
* Network size :- Both actor and critic network are three layered. Deeper network was slow to converge and a tradeoff between noise and depth of the network is important
* Learning rate :- All networks were set to 0.0003, much lower than normal rate of 0.001. This was needed so as to avoid the agent getting stuck in a local minima as the exploration via noise reduces with each epoch
* Repreated learning : many were useful, but certainly more than 4 is needed for convergene within hours
* Buffer size and batch size : These two were the most important during training. A very large buffer allows retaining not so useful memory from which agent will not learn appropriately. Similarly, as very large batch size will also lead to slower learning as the rewards are scarce.
* Bootrstap : this is set to 4, 

 
### 5. Experiment and Results

Many variations of hyperparameters are explored. All variation solves the environment and attains a score of 0.5+
1. Baseline (Update loops : 16, buffer : 50000, batch size : 128) ~3120 seconds
    1700s
2. Buffer : 1e6, batch : 128
    2900s  10000+ episodes
    <img src='img/score_128.png' height='100'><br />
3. Gradient cliping = 3, update loop = 12, Update every 8th iteration
    ~1600s, 2319 episodes
    <img src='img/score_2.png' height='100'><br />
4. Update loop = 8, update every 4th iteration
   4300+s   slowest, but under 3000 episodes
   <img src='img/score_3.png' height='100'><br />

Slowest is 2nd configuration, fastest is 3rd

### 6. Next Steps
There are many things that can be tried
- A different algorithm like PPO
- Different hyperparameters like making tau 1e-5 but much larger batch size, like 1024 or more
- Other options for noise
- Parameter noise, instead of action noise
- Prioritied buffer