import numpy as np
from gym import spaces
import torch

'''
Class definnitions for replay buffer used in off-policy training
'''

class SequenceReplayBuffer:
    def __init__(self, buffer_size, observation_space,
                 action_space, hidden_state_size, sequence_length=1,
                 burn_in_length=0, n_envs=1,
                 alpha=0.6, beta=0.4, 
                 beta_increment=0.0001, max_priority=1.0):
        '''
        A replay buffer for R2D2 algorithm that when sampled, produces sequences of time steps.
        Any index can be samples from, and burn_in_length steps before the index and sequence_length
          steps after the index will be passed together
        Note that there is one torch tensor per variable (observations, actions, rewards, dones, rnn_hxs)
          and it will continuously be overwritten. Each tensor has length burn_in_length+buffer_size+sequence_length.
          Think of it as having buffer_size, plus a chunk behind and ahead to handle burn in and sequence.
          
        self.pos keeps track of the next index to be written to. When it reaches the end (burn_in_length+buffer_size)
          it loops back to the start (burn_in_length).
          When it loops back, the burn_in_length chunk is copied from the end of the buffer.
          As it covers the the first sequence_length worth of steps in the buffer, these get copied to the end
            sequence_length chunk of the buffer
        
        buffer_size: number of steps to hold in buffer
        sequence_length: number of steps in sequence
        burn_in_length: number of steps before idx to be passed with sequence
        '''
        self.buffer_size = buffer_size
        total_buffer_size = buffer_size + sequence_length + burn_in_length
        self.n_envs = n_envs
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = max_priority

        self.priorities = np.zeros(total_buffer_size)
        
        action_shape = get_action_dim(action_space)
        self.observations = np.zeros((total_buffer_size, n_envs, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((total_buffer_size, n_envs, action_shape), dtype=action_space.dtype)
        self.rewards = np.zeros((total_buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((total_buffer_size, n_envs), dtype=np.float32)
        self.hidden_states = np.zeros((total_buffer_size, n_envs, hidden_state_size), dtype=np.float32)

        self.pos = burn_in_length
        self.full = False
        
    def add(self, obs, next_obs, action, reward, done, hidden_state):
        '''
        Add to the buffer
        '''
        self.observations[self.pos] = np.array(obs).copy()
        self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.hidden_states[self.pos] = np.array(hidden_state).copy()
        
        
        bil = self.burn_in_length
        bs = self.buffer_size
        sl = self.sequence_length
        
        #Make copies to extra end portion
        #Note that this makes it so that for a sequence_length period of time
        #while we are filling up the end of the buffer, end steps cannot be used
        if self.pos < bil + sl:
            self.observations[self.pos+bs] = self.observations[self.pos].copy()
            self.actions[self.pos+bs] = self.actions[self.pos].copy()
            self.rewards[self.pos+bs] = self.rewards[self.pos].copy()
            self.dones[self.pos+bs] = self.dones[self.pos].copy()
            self.hidden_states[self.pos+bs] = self.hidden_states[self.pos].copy()
        
        self.pos += 1
        if self.pos == self.buffer_size + self.burn_in_length:
            self.pos = self.burn_in_length
            self.full = True
            
            #Make copies to the burn_in portion
            self.observations[:bil] = self.observations[bs:bs+bil].copy()
            self.actions[:bil] = self.actions[bs:bs+bil].copy()
            self.rewards[:bil] = self.rewards[bs:bs+bil].copy()
            self.dones[:bil] = self.dones[bs:bs+bil].copy()
            self.hidden_states[:bil] = self.hidden_states[bs:bs+bil].copy()
        
        self.priorities[self.pos] = self.max_priority
        
        
    def sample(self, batch_size):
        '''
        Generate a sample of data to be trained with from the buffer
        '''
        probs = self.priorities ** self.alpha
        probs /= probs.sum()

        # Calculate the valid indices for sampling sequences
        valid_idxs = self.get_valid_idxs()

        # Normalize the probabilities of the valid indices
        valid_probs = probs[valid_idxs]
        valid_probs /= valid_probs.sum()

        # Sample the indices using the normalized probabilities
        idxs = np.random.choice(valid_idxs, batch_size, p=valid_probs)
        start_idxs = idxs - self.burn_in_length
        
        window_idxs = np.arange(-self.burn_in_length, self.sequence_length)
        window_length = len(window_idxs)
        seq_idxs = idxs[:, np.newaxis] + window_idxs
        
        #Randomly sample env_ids
        env_idxs = np.random.randint(0, high=self.n_envs, size=(batch_size,))
        seq_env_idxs = np.full((batch_size, window_length), env_idxs[:, np.newaxis])
        
        weights = (probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment)
        
        obs = torch.Tensor(self.observations[seq_idxs, seq_env_idxs])
        next_obs = torch.Tensor(self.observations[seq_idxs+1, seq_env_idxs])
        actions = torch.Tensor(self.actions[seq_idxs, seq_env_idxs])
        rewards = torch.Tensor(self.rewards[seq_idxs, seq_env_idxs])
        dones = torch.Tensor(self.dones[seq_idxs, seq_env_idxs])
        next_dones = torch.Tensor(self.dones[seq_idxs+1, seq_env_idxs])
        
        hidden_states = torch.Tensor(self.hidden_states[start_idxs, env_idxs]).unsqueeze(0)
        next_hidden_states = torch.Tensor(self.hidden_states[start_idxs+1, env_idxs]).unsqueeze(0)
        
        sample = {
            'observations': obs,
            'next_observations': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_dones': next_dones,
            'hidden_states': hidden_states,
            'next_hidden_states': next_hidden_states
        }
        
        return sample
    
    
    def get_valid_idxs(self):
        '''
        Get array of valid indexes that can be sampled. Valid indexes are those with enough time steps
        of data earlier to match burn_in_lenth and with enough time steps of data later to match
        sequence_length.
        
        Note: there is a slight bug here to be fixed in the future - there is a period where self.pos
          loops back that the sequence_length post buffer is stale until overwritten fully 
        '''
        start = self.burn_in_length
        end = self.burn_in_length + self.buffer_size
        
        if self.full:
            #Have enough terms ahead to be usable
            valid1 = np.arange(start, self.pos - self.sequence_length)
            #Have enough terms behind to be usable
            valid2 = np.arange(self.pos + self.burn_in_length + 1, end)
            valid_idxs = np.concatenate([valid1, valid2])
        else:
            #First burn_in_length steps are not valid because they haven't been copied
            valid_idxs = np.arange(start + self.burn_in_length, self.pos - self.sequence_length)
        return valid_idxs

    def update_priorities(self, indices, priorities):
        '''
        indices: shape [N,] for N batches
        priorities: shape [N, seq_len+burn_in]
        
        Note: currently assuming that we calculate priorities for seq_len + burn_in, but it should just be seq_len
        '''
        for idx, priority in zip(indices, priorities):
            self.priorities[idx+self.burn_in_length:idx+self.sequence_length+self.burn_in_length] = priority
            self.max_priority = max(self.max_priority, max(priority))

    def __len__(self):
        return len(self.buffer)
    
    
    
    
def get_action_dim(action_space):
    """
    Get the dimension of the action space.
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")