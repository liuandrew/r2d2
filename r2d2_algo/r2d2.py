
import torch
import torch.optim as optim
import gym
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

from model import RNNQNetwork, linear_schedule
from storage import SequenceReplayBuffer
from args import get_args

if __name__ == '__main__':
    args = get_args()
    env_id = args.env_id
    learning_rate = args.learning_rate
    buffer_size = args.buffer_size
    total_timesteps = args.total_timesteps
    learning_starts = args.learning_starts
    train_frequency = args.train_frequency
    gamma = args.gamma
    tau = args.tau
    target_network_frequency = args.target_network_frequency

    start_e = args.start_e
    end_e = args.end_e
    exploration_fraction = args.exploration_fraction

    burn_in_length = args.burn_in_length
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    
    seed = args.seed
    torch_deterministic = args.torch_deterministic
    track = args.track
    exp_name = args.exp_name
    cuda = args.cuda
    
    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    if track:
        import wandb
        
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')


    env = gym.make(env_id)

    hidden_size = 64
    q_network = RNNQNetwork(env, hidden_size).to(device)
    target_network = RNNQNetwork(env, hidden_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    rb = SequenceReplayBuffer(buffer_size, env.observation_space, env.action_space,
                            hidden_size, sequence_length, burn_in_length)
    obs = env.reset()

    lengths = []
    returns = []
    
    global_update_steps = 0
    cur_episode_t = 0
    cur_episode_r = 0
    start_time = time.time()
    
    rnn_hxs = q_network.get_rnn_hxs()
    for global_step in range(total_timesteps):
        
        # Collect environment data
        epsilon = linear_schedule(start_e, end_e, 
                                exploration_fraction*total_timesteps,
                                global_step)
        
        obs_tensor = torch.Tensor(obs).to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        q_values, next_rnn_hxs = q_network(obs_tensor, rnn_hxs)
        if random.random() < epsilon:
            action = np.array([[env.action_space.sample()]])
        else:
            action = np.array([[q_values.argmax()]])
        
        next_obs, reward, done, info = env.step(action.item())
        cur_episode_t += 1
        cur_episode_r += reward
                    
        if done:            
            next_obs = env.reset()
            next_rnn_hxs = q_network.get_rnn_hxs()
            lengths.append(cur_episode_t)
            returns.append(cur_episode_r)
            
            writer.add_scalar('charts/episodic_return', cur_episode_r, global_step)
            writer.add_scalar('charts/episodic_length', cur_episode_t, global_step)
            writer.add_scalar('charts/epsilon', epsilon, global_step)
            
            cur_episode_t = 0
            cur_episode_r = 0

        rb.add(obs, next_obs, action, reward, done, rnn_hxs.detach())
        
        obs = next_obs
        rnn_hxs = next_rnn_hxs
        
        #Training
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                # states, actions, rewards, next_states, dones, _, _, hidden_states, next_hidden_states = rb.sample(batch_size//sequence_length)
                sample = rb.sample(batch_size//sequence_length)
                states = sample['observations']
                next_states = sample['next_observations']
                hidden_states = sample['hidden_states']
                next_hidden_states = sample['next_hidden_states']
                actions = sample['actions']
                rewards = sample['rewards']
                dones = sample['dones']
                next_dones = sample['next_dones']
                
                with torch.no_grad():
                    target_q, _ = target_network(next_states, next_hidden_states, next_dones)
                    target_max, _ = target_q.max(dim=2)
                    td_target = rewards + gamma * target_max * (1 - dones)
                old_q, _ = q_network(states, hidden_states, dones)
                old_val = old_q.gather(2, actions.long()).squeeze()

                loss = F.mse_loss(td_target[:, burn_in_length:], old_val[:, burn_in_length:])
                
                if global_update_steps % 10 == 0:
                    writer.add_scalar('losses/td_loss', loss, global_step)
                    writer.add_scalar('losses/q_values', old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    print('SPS:', int(sps))
                    writer.add_scalar('charts/SPS', sps, global_step)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                global_update_steps += 1
                
                #checkpoint
                #if args.checkpoint_interval > 0 and global_update_steps % args.checkpoint_interval == 0:
                #   checkpoint_path = f'saved_checkpoints/{args.save_name}'
                #   ...
                
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1 - tau) * target_network_param.data
                    )
             


        if global_step % 2000 == 0:
            print(f'Mean episode length {np.mean(lengths)}, mean return {np.mean(returns)}')
            
            returns = []
            lengths = []
            
    if args.save_model:
        save_name = f'{args.exp_name}__{args.seed}'

        # Note: add manual save name here later
        # if args.save_name is not None:
        #   save_name = args.save_name
        
        #Save just the q_network which can be used to generate actions
        save_path = f'saved_models/{save_name}.pt'
        torch.save(q_network.state_dict(), save_path)
        
        #Code to save entire training history which can be reinitialized later
        # torch.save({
        #     'q_network': q_network,
        #     'target_network': target_network,
        #     'buffer': rb,
        #     'last_obs': obs,
        #     'last_rnn_hxs': rnn_hxs,
        #     'env': env,
        #     'global_step': global_step,
        #     'global_update_steps': global_update_steps
        # }, save_path)