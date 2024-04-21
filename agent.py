import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from neural import MarioNet  # Import the MarioNet class, assumed to be defined elsewhere.


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        """
        Initialize the Mario agent with the necessary dimensions and settings.

        Parameters:
        - state_dim (tuple): Dimensions of the state representation.
        - action_dim (int): Number of possible actions the agent can take.
        - save_dir (Path): Directory to save checkpoints.
        - checkpoint (Path, optional): Path to a pre-existing checkpoint to load.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA if available.
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10000))  # Use a memory-efficient replay buffer.
        self.batch_size = 32  # Batch size for training from experiences.

        # Exploration settings for the agent.
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9  # Discount factor for future rewards.

        # Learning and synchronization settings.
        self.curr_step = 0
        self.burnin = 1e4  # Minimum number of steps before starting training.
        self.learn_every = 3  # Number of steps between updates to Q_online.
        self.sync_every = 1e4  # Steps between synchronizations of Q_target and Q_online.
        self.save_every = 5e5  # Steps between saving MarioNet checkpoints.
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)  # Load checkpoint if provided.

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=0.00025)  # Optimizer for the network.
        self.loss_fn = torch.nn.SmoothL1Loss()  # Loss function for training.

    def act(self, state):
        """
        Choose an action based on the state using an epsilon-greedy policy.

        Parameters:
        - state (LazyFrame): Current state of the environment.

        Returns:
        - action_idx (int): Chosen action index.
        """
        if np.random.rand() < self.exploration_rate:  # Exploration with a chance of epsilon.
            action_idx = np.random.randint(self.action_dim)
        else:  # Exploitation: choose best action based on network's output.
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the transition in the replay buffer.

        Parameters:
        - state, next_state (LazyFrame): Current and next state of the environment.
        - action (int): Action taken at the current state.
        - reward (float): Reward received after taking the action.
        - done (bool): Whether the episode has ended.
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state, next_state = map(
            lambda x: torch.tensor(first_if_tuple(x).__array__(), dtype=torch.float32),
            [state, next_state])
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({
            "state": state, "next_state": next_state, "action": action,
            "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from the replay buffer.

        Returns:
        - tuple of (state, next_state, action, reward, done) tensors.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        return (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))

    def td_estimate(self, state, action):
        """
        Calculate the TD estimate for the Q function.

        Parameters:
        - state (tensor): State at which to evaluate.
        - action (tensor): Actions taken.

        Returns:
        - Tensor representing the estimated Q values for the provided actions.
        """
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Calculate the TD target for updates.

        Parameters:
        - reward (tensor): Rewards received.
        - next_state (tensor): Subsequent state after action.
        - done (tensor): Whether the episode has ended.

        Returns:
        - Tensor of target Q values.
        """
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Perform a single update step on the Q_online model using the loss between TD estimates and TD targets.

        Parameters:
        - td_estimate (tensor): Estimated Q values.
        - td_target (tensor): Target Q values.

        Returns:
        - float: The loss value computed.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        Synchronize the target network with the online network.
        """
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """
        Conduct a learning step if enough experiences have been gathered.

        Returns:
        - tuple (float, float): Mean estimate of Q and the loss, or None if no learning was performed.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None  # Not enough data to learn yet.

        if self.curr_step % self.learn_every != 0:
            return None, None  # Not a learning step.

        state, next_state, action, reward, done = self.recall()  # Sample from buffer.

        td_est = self.td_estimate(state, action)  # Get TD estimate.
        td_tgt = self.td_target(reward, next_state, done)  # Get TD target.

        loss = self.update_Q_online(td_est, td_tgt)  # Update the network.
        return td_est.mean().item(), loss

    def save(self):
        """
        Save the current model and exploration rate to a checkpoint file.
        """
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        """
        Load model and exploration rate from a checkpoint file.

        Parameters:
        - load_path (Path): Path to the checkpoint file.
        """
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
