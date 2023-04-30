import torch
from torch import nn


'''
Class definitions for R2D2 network
'''
    
class ResettingGRU(nn.Module):
    '''
    Modification to GRU that can take dones on the forward call to tell when
    state in the middle of a batched sequence forward call should be reset
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.gru = nn.GRU(**kwargs)
        
        self.hidden_size = self.gru.hidden_size
        self.batch_first = self.gru.batch_first
        
    def get_rnn_hxs(self, num_batches=1):
        '''
        Get torch.zeros hidden states to start off with
        '''
        if num_batches == 1:
            return torch.zeros(1, self.hidden_size)
        else:
            return torch.zeros(1, num_batches, self.hidden_size)
        
    
    def forward(self, x, hidden_state, dones=None):
        if dones == None:
            return self.gru(x, hidden_state)
        
        #Unfortunately need to split up the batch here
        if self.batch_first == False:
            raise NotImplementedError        
        
        def single_gru_row(x_i, rnn_hx, done):
            breakpoints = (done == 1).argwhere()
            cur_idx = 0
            output_row = torch.zeros(x_i.shape[0], self.hidden_size)
            
            for breakpoint in breakpoints:
                if breakpoint == 0:
                    rnn_hx = self.get_rnn_hxs()
                    continue
                out, out_hx = self.gru(x_i[cur_idx:breakpoint], rnn_hx)
                output_row[cur_idx:breakpoint] = out
                rnn_hx = self.get_rnn_hxs()
                cur_idx = breakpoint
                
            if cur_idx < x_i.shape[0]:
                out, out_hx = self.gru(x_i[cur_idx:], rnn_hx)
                output_row[cur_idx:] = out
            return output_row, out_hx
        
        if hidden_state.dim() == 2:
            output, output_hx = single_gru_row(x, hidden_state, dones)
            
        else:
            num_batches = hidden_state.shape[1]
            
            # Output will have shape [N, L, hidden_size]
            full_out = torch.zeros((x.shape[0], x.shape[1], self.hidden_size))
            # hidden_state output has shape [1, N, hidden_size]
            full_hx_out = torch.zeros((1, x.shape[0], self.hidden_size))
            
            batchable_rows = (dones == 0).all(dim=1)
            individual_rows = (~batchable_rows).argwhere().reshape(-1)
            
            # First batch all computations that have no dones in them
            out, out_hx = self.gru(x[batchable_rows], hidden_state[:, batchable_rows, :])
            full_out[batchable_rows] = out
            full_hx_out[:, batchable_rows, :] = out_hx
            
            for i in individual_rows:
                d = dones[i]
                x_i = x[i]
                rnn_hx = hidden_state[:, i, :]
                
                output_row, output_hx_row = single_gru_row(x_i, rnn_hx, d)
                full_out[i] = output_row
                full_hx_out[:, i, :] = output_hx_row
            
        
        return full_out, full_hx_out
            
            


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope*t + start_e, end_e)


class RNNQNetwork(nn.Module):
    '''RNN Network using ResettingGRU that outputs Q-values'''
    
    def __init__(self, env, hidden_size):
        super(RNNQNetwork, self).__init__()
        self.hidden_size = hidden_size
        state_size = env.observation_space.shape[0]
        
        self.relu = nn.ReLU()
        self.gru = ResettingGRU(input_size=state_size, hidden_size=hidden_size,
                               batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        

        self.fc0 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, env.action_space.n)
        
    def forward(self, state, hidden_state, dones=None):
        '''
        Forward pass of GRU network. 
            hidden_state should have size [1, hidden_size] for unbatched or [1, N, hidden_size] for batched
            state should have size [L, input_size] for unbatched or [N, L, input_size] for batched
        return (unbatched)
            q_values [L, 1], gru_out [L, hidden_size]
        return (batched)
            q_values [N, L, 1], gru_out[N, L, hidden_size]
        '''        
        gru_out, (next_hidden_state) = self.gru(state, hidden_state, dones)
        
        x = self.relu(gru_out)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        q_values = x
                        
        return q_values, gru_out
    
    def get_rnn_hxs(self, num_batches=1):
        '''
        Get torch.zeros hidden states to start off with
        '''
        if num_batches == 1:
            return torch.zeros(1, self.hidden_size)
        else:
            return torch.zeros(1, num_batches, self.hidden_size)