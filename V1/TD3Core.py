import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        sd      = state_dim
        #self.l1 = nn.Linear(sd, sd*1.5)
        self.l1 = nn.Linear(int(sd), int(sd*1.5))
        self.l2 = nn.Linear(int(sd*1.5), int(sd*2))
        self.l3 = nn.Linear(sd*2, sd*4)
        self.l4 = nn.Linear(sd*4, sd*4)
        self.l5 = nn.Linear(sd*4, sd*4)
        self.l6 = nn.Linear(sd*4, sd*4)
        self.l7 = nn.Linear(sd*4, sd*2)
        self.l8 = nn.Linear(sd*2, sd*1)
        self.l9 = nn.Linear(sd*1, sd*1)
        self.lA = nn.Linear(sd*1, action_dim)
        
        self.max_action = max_action
        
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)   
        torch.nn.init.xavier_uniform_(self.l3.weight)  
        torch.nn.init.xavier_uniform_(self.l4.weight)  
        torch.nn.init.xavier_uniform_(self.l5.weight)  
        torch.nn.init.xavier_uniform_(self.l6.weight)
        torch.nn.init.xavier_uniform_(self.l7.weight) 
        torch.nn.init.xavier_uniform_(self.l8.weight) 
        torch.nn.init.xavier_uniform_(self.l9.weight)   
        torch.nn.init.xavier_uniform_(self.lA.weight)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.tanh(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.tanh(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.tanh(self.l6(x))
        x = F.tanh(self.l7(x))
        x = F.tanh(self.l8(x))
        x = F.tanh(self.l9(x))
        x = (self.lA(x))
        return x
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        cd      = state_dim + action_dim
        self.l1 = nn.Linear(cd, int(cd*1.5))
        self.l2 = nn.Linear(int(cd*1.5), cd*2)
        self.l3 = nn.Linear(cd*2, cd*5)
        self.l4 = nn.Linear(cd*5, cd*5)
        self.l5 = nn.Linear(cd*5, cd*5)
        self.l6 = nn.Linear(cd*5, cd*5)
        self.l7 = nn.Linear(cd*5, cd*5)
        self.l8 = nn.Linear(cd*5, cd*3)
        self.l9 = nn.Linear(cd*3, cd)
        self.lA = nn.Linear(cd, 1)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)   
        torch.nn.init.xavier_uniform_(self.l3.weight)  
        torch.nn.init.xavier_uniform_(self.l4.weight)  
        torch.nn.init.xavier_uniform_(self.l5.weight)
        torch.nn.init.xavier_uniform_(self.l6.weight)
        torch.nn.init.xavier_uniform_(self.l7.weight)
        torch.nn.init.xavier_uniform_(self.l8.weight)  
        torch.nn.init.xavier_uniform_(self.l9.weight)  
        torch.nn.init.xavier_uniform_(self.lA.weight)  
        
    def forward(self, state, action):
        state_action = torch.cat([state, action/99], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.tanh(self.l2(q))
        q = F.relu(self.l3(q))
        q = F.tanh(self.l4(q))
        q = F.relu(self.l5(q))
        q = F.tanh(self.l6(q))
        q = F.relu(self.l7(q))
        q = F.relu(self.l8(q))
        q = F.relu(self.l9(q))
        q = self.lA(q)
        return q
    
class TD3:
    def __init__(self, alr, c1lr, c2lr, state_dim, action_dim, max_action, gpu_num):
        
        self.device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=c1lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=c2lr)
        
        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action_).to(self.device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(self.device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory, name, mode = 1):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        if not mode==1: self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
        
      
        
