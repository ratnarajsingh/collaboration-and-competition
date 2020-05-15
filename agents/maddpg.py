import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from agents.utils import soft_update, hard_update, SimpleNoise, ReplayBuffer
from agents.model import Actor, Critic


class MADDPG():
    """
    Class definition of MADDPG agent. Interacts with and learns from the environment
    Comprises of a pair of Actor-Critic network ad implements centralized training and decentralized exeution (learn function)
    """
    
    def __init__(self, state_size, action_size, num_agents, device, seed = 23520,
                GRADIENT_CLIP = 1,
                ACTIVATION = F.relu,
                BOOTSTRAP_SIZE = 5,
                GAMMA = 0.99, 
                TAU = 1e-3, 
                LR_CRITIC = 5e-4,
                LR_ACTOR = 5e-4, 
                UPDATE_EVERY = 1,
                TRANSFER_EVERY = 2,
                UPDATE_LOOP = 10,
                ADD_NOISE_EVERY = 5,
                WEIGHT_DECAY = 0,
                MEMORY_SIZE = 5e4,
                BATCH_SIZE = 64):
        """Initialize an Agent object.
        
        Params
        ======
            state_size  : dimension of each state
            action_size : dimension of each action
            num_agents  : number of running agents
            device: cpu or cuda:0 if available
            -----These are hyperparameters----
            BOOTSTRAP_SIZE      : How far ahead to bootstrap
            GAMMA               : Discount factor
            TAU                 : Parameter for performing soft updates of target parameters
            LR_CRITIC, LR_ACTOR : Learning rate of the networks
            UPDATE_EVERY        : How often to update the networks
            TRANSFER_EVERY      : How often to transfer the weights from local to target
            UPDATE_LOOP         : Number of iterations for network update
            ADD_NOISE_EVERY     : How often to add noise to favor exploration
            WEIGHT_DECAY        : L2 weight decay for critic optimizer
            GRADIENT_CLIP       : Limit of gradient to be clipped, to avoid exploding gradient issue
        """
        
        # Actor networks
        self.actor_local  = Actor(state_size,action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        hard_update(self.actor_local,self.actor_target)
        
        #critic networks
        self.critic_local  = Critic(state_size*2,action_size).to(device)
        self.critic_target = Critic(state_size*2,action_size).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        hard_update(self.critic_local,self.critic_target)
        
        self.device = device
        self.num_agents = num_agents
        
        # Noise : using simple noise instead of OUNoise
        self.noise = [SimpleNoise(action_size, scale=1) for i in range(num_agents)] 
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, device, int(MEMORY_SIZE), BATCH_SIZE, seed)
        
        # Initialize time steps (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.n_step = 0
        
        #keeping hyperparameters within the instance
        self.BOOTSTRAP_SIZE = BOOTSTRAP_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TRANSFER_EVERY = TRANSFER_EVERY
        self.UPDATE_LOOP = UPDATE_LOOP
        self.ADD_NOISE_EVERY = ADD_NOISE_EVERY
        self.GRADIENT_CLIP = GRADIENT_CLIP
        
        # initialize these variables to store the information of the n-previous timestep that are necessary to apply the bootstrap_size
        self.rewards = deque(maxlen=BOOTSTRAP_SIZE)
        self.states = deque(maxlen=BOOTSTRAP_SIZE)
        self.actions = deque(maxlen=BOOTSTRAP_SIZE)
        self.gammas = np.array([[GAMMA ** i for j in range(num_agents)] for i in range(BOOTSTRAP_SIZE)])
        
        self.loss_function = torch.nn.SmoothL1Loss()
    
    def reset(self):
        if self.noise:
            for n in self.noise:
                n.reset()
        
    def set_noise(self, noise):
        self.noise = noise
        
    def save(self, filename):
        torch.save(self.actor_local.state_dict(),"{}_actor_local.pth".format(filename))
        torch.save(self.actor_target.state_dict(),"{}_actor_target.pth".format(filename))
        torch.save(self.critic_local.state_dict(),"{}_critic_local.pth".format(filename))
        torch.save(self.critic_target.state_dict(),"{}_critic_target.pth".format(filename))
     
    def load(self, path):
        self.actor_local.load_state_dict(torch.load(path +"_actor_local.pth"))
        self.actor_target.load_state_dict(torch.load(path+"_actor_target.pth"))
        self.critic_local.load_state_dict(torch.load(path+"_critic_local.pth"))
        self.critic_target.load_state_dict(torch.load(path+"_critic_target.pth"))
    
    def act(self, states, noise = 0.0):
        """
        Returns actions of each actor for given states.
        
        Params
        ======
            state    : current states
            add_noise: Introduce some noise in agent's action or not. During training, this is necessary to promote the exploration but should not be used during validation
        """
        actions = None
        
        self.n_step = (self.n_step + 1) % self.ADD_NOISE_EVERY
        
        with torch.no_grad():
            self.actor_local.eval()
            states = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
            actions = self.actor_local(states).squeeze().cpu().data.numpy()
            self.actor_local.train()
            if self.n_step == 0:
                for i in range(len(actions)):
                    actions[i] += noise * self.noise[i].sample()
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """
        Take a step for the current episode
        1. Save the experience
        2. Bootstrap the rewards
        3. If update conditions are statisfied, perform learning on required number of loops
        """
        
        # Save experience in replay memory
        self.rewards.append(rewards)
        self.states.append(states)
        self.actions.append(actions)
            
        if len(self.rewards) == self.BOOTSTRAP_SIZE:
            # get the bootstrapped sum of rewards per agents
            reward = np.sum(self.rewards * self.gammas, axis = -2)
            self.memory.add(self.states[0], self.actions[0], reward, next_states, dones)
            
        if np.any(dones):
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            
        # Learn every UPDATE_EVERY timesteps
        self.u_step = (self.u_step + 1) % self.UPDATE_EVERY
        
        t_step=0
        if len(self.memory) > self.memory.batch_size and self.u_step == 0:
            for _ in range(self.UPDATE_LOOP):
                self.learn()
                # transfer the weights as specified
                t_step=(t_step + 1) % self.TRANSFER_EVERY
                if t_step == 0:
                    soft_update(self.actor_local, self.actor_target, self.TAU)
                    soft_update(self.critic_local, self.critic_target, self.TAU)
    
    def transform_states(self, states):
        """
        Transforms states to full states so that both agents can see each others state via the critic network
        """
        batch_size = states.shape[0]
        state_size = states.shape[-1]
        num_agents = states.shape[-2]
        transformed_states = torch.zeros((batch_size, num_agents, state_size * num_agents)).to(self.device)
        for i in range(num_agents):
            start = 0
            for j in range(num_agents):
                transformed_states[:,i,start:start + state_size] += states[:,j]
                start += state_size
        return transformed_states
    
    def learn(self):
        """
        Update the network parameters using the experiences. The algorithm is described in detail in readme

        Params
        ======
            experiences : List of (s, a, r, s', done) tuples
        """
        # sample the memory to disrupt the internal correlation
        states, actions, rewards, next_states, dones = self.memory.sample()
        full_states = self.transform_states(states)

        # The critic should estimate the value of the states to be equal to rewards plus
        # the estimation of the next_states value according to the critic_target and actor_target
        with torch.no_grad():
            self.actor_target.eval()
            self.critic_target.eval()
            # obtain next actions as given by the target network and get transformed states for the critic
            next_actions = self.actor_target(next_states)
            next_full_states = self.transform_states(next_states)
            # calculate Q value using transformed next states and next actions, basically predict what the next value is from target's perspective
            q_next = self.critic_target(next_full_states, next_actions).squeeze(-1)
            # calculate the target's value
            targeted_value = rewards + (self.GAMMA**self.BOOTSTRAP_SIZE)*q_next*(1 - dones)
         
        current_value = self.critic_local(full_states, actions).squeeze(-1)
        loss = self.loss_function(current_value, targeted_value)
        # During the optimization, the critic tells how much the value is off from the action value and adjusts the network towards. Basically, the critic takes the actions predicted by the actor, and tells how good or bad they are by calculating its Q-value 
        
        # calculate the loss of the critic network and backpropagate
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.GRADIENT_CLIP)
        self.critic_optim.step()

        # optimize the actor by having the critic evaluating the value of the actor's decision
        self.actor_optim.zero_grad()
        actions_pred = self.actor_local(states)
        mean = self.critic_local(full_states, actions_pred).mean()
        (-mean).backward()
        self.actor_optim.step()    
