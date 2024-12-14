import datetime
import pathlib

import time
import numpy as np
import torch

from .abstract_game import AbstractGame

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time # Only for added randomness
import pgx
import jax
import jax.numpy as jnp
import cairosvg 

import os
os.environ["JAX_PLATFORMS"] = "cpu"


class MuZeroConfig:
    def __init__(self):

        self.seed = int(time.time())  # Seed for numpy, torch and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (17, 9, 9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(82))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 48  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Maximum number of moves if game is not finished before
        self.num_simulations = 32  # Number of future moves self-simulated
        self.discount = 0.95  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 4  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 64  # Number of channels in reward head
        self.reduced_channels_value = 64  # Number of channels in value head
        self.reduced_channels_policy = 64  # Number of channels in policy head
        self.resnet_fc_reward_layers = [128]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [128]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [128]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 64
        self.fc_representation_layers = [128,32]  # Hidden layers in the representation network
        self.fc_dynamics_layers = [128,32]  # Hidden layers in the dynamics network
        self.fc_reward_layers = [64,32]  # Hidden layers in the reward network
        self.fc_value_layers = [64,32]  # Hidden layers in the value network
        self.fc_policy_layers = [64,32]  # Hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 16000 # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 0.95  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 32  # Number of game moves to keep for every batch element
        self.td_steps = 40  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):

    def __init__(self, seed=None):
        self.env = MiniGo()

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        return self.env.to_play()
    
    def legal_actions(self):
        return self.env.legal_actions()
    
    def reset(self):
        return self.env.reset()
        
    def render(self):
        self.env.render()
        # input("Press enter to take a step:")
        return;
    
    def human_to_action(self):
        legal_actions = set(self.env.legal_actions())
        while True:
            print("Available Actions:") 
            for a in legal_actions:
                print(a, end=' ')
            print()
            chosen = input("Choose an action:")
            try:
                chosen = int(chosen)
                if chosen in legal_actions:
                    break
            except: 
                continue
        return chosen
    
    def expert_agent(self):
        return self.env.expert_action()

    def action_to_string(self, action_number):
        return f"Playing Action:{action_number}"


# MiniGo wrapper on PGX
class MiniGo:

    ENV_ID = "go_9x9"

    def __init__(self):
        self.env_id = self.ENV_ID
        self.reset_count = 0
        self.reset()

    def to_play(self):
        return int(self.state.current_player)

    def reset(self):
        self.reset_count += 1
        self.env = pgx.make(self.env_id)
        init = jax.jit(self.env.init)
        self.env_step = jax.jit(self.env.step)
        randomness = round(time.time()) + self.reset_count + int( random.random() * 1e6 )
        key = jax.random.PRNGKey(randomness)
        self.state = init(key)
        observations = self.state.observation.__array__()
        transformed_observations = np.transpose(observations, (2, 0, 1))
        return transformed_observations

    def step(self, action):
        step_player = self.to_play()
        vectorized_action = jnp.array(action)
        new_state = self.env_step(self.state, vectorized_action)
        observations = new_state.observation.__array__()
        transformed_observations = np.transpose(observations, (2, 0, 1))
        reward = int(new_state.rewards[step_player])
        done = bool(new_state.terminated) or bool(new_state.truncated)
        self.state = new_state 
        return transformed_observations, reward, done

    def legal_actions(self):
        action_mask = self.state.legal_action_mask
        legal = np.where(action_mask.__array__() == True)[0].tolist()
        return legal

    def render(self):
        source_name = "temp_state.svg"
        target_name = str(int(time.time())) + ".png"
        self.state.save_svg(source_name)
        # Convert to PNG
        target_location = f"./viz/minigo/{target_name}"
        cairosvg.svg2png(url=source_name, write_to=target_location)
        # Display the state
        # img = mpimg.imread("temp_state.png")
        # plt.imshow(img)
        # plt.axis('off') 
        # plt.show()
        return;

    def expert_action(self):
        ''' Only returns a random legal action, not implemented because it does not affect training! '''
        legal_actions = self.legal_actions()
        if len(legal_actions) == 0:
            return 0 
        return random.choice(legal_actions)
