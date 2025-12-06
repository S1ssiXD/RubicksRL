import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from cube import Cube, Move, moves
from cube_nn import cube_to_tensor_one_hot

# Hyperparameters for PPO
DISCOUNT_FACTOR = 0.9
GAE_LAMBDA = 0.95

AC_TRAINING_STEPS = 5
TARGET_KL = 0.01
ENTROPY_COEF = 0.01

# Constants of the environment
N_STATE = 54
N_STATE_TORCH = N_STATE*6  # one-hot encoding of the cube state
N_ACTION = 18


class ACModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(N_STATE_TORCH, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTION)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(N_STATE_TORCH, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        x = self.actor(obs)
        # logit = F.log_softmax(x, dim=1)
        logit = x

        y = self.critic(obs)
        value = y.squeeze(1)
        return logit, value


class RLAgent():
    def __init__(self, ac_model: nn.Module, max_rollout_length: int) -> None:
        self.ac_model = ac_model
        self._eval_mode = True
        self._max_rollout_length = max_rollout_length

        self._history_idx = 0
        self._cube_state_history = np.zeros(
            (N_STATE_TORCH, max_rollout_length), dtype=np.uint8)
        self._action_history = np.zeros(max_rollout_length, dtype=np.uint8)
        self._value_history = np.zeros(max_rollout_length, dtype=np.float32)

    def get_move(self, cube: Cube) -> Move:

        # One-hot encode the cube state
        cube_tensor = cube_to_tensor_one_hot(cube).flatten().unsqueeze(0)
        # cube_tensor = cube_to_tensor_direct(cube).flatten().unsqueeze(0)

        # Save the current state to history
        self._cube_state_history[:, self._history_idx] = cube_tensor.numpy()

        logit, value = self.ac_model(cube_tensor)
        self._value_history[self._history_idx] = value.item()

        dist = Categorical(logits=logit)

        if self._eval_mode:
            # Deterministic: pick the most probable action
            action = int(torch.argmax(dist.probs).item())
        else:
            # Stochastic: sample from distribution
            action = int(dist.sample().item())
        self._action_history[self._history_idx] = action

        self._history_idx += 1

        move = moves[action]
        return move

    def set_to_eval_mode(self) -> None:
        self.ac_model.eval()
        self._eval_mode = True

    def set_to_train_mode(self) -> None:
        self.ac_model.train()
        self._eval_mode = False

    def reset(self) -> None:
        self._history_idx = 0
        self._cube_state_history = np.zeros(
            (N_STATE_TORCH, self._max_rollout_length), dtype=np.uint8)
        self._action_history = np.zeros(
            self._max_rollout_length, dtype=np.uint8)
        self._value_history = np.zeros(
            self._max_rollout_length, dtype=np.float32)

    def get_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cube_state_history[:, :self._history_idx], self._action_history[:self._history_idx], self._value_history[:self._history_idx]


def compute_rewards(cube_state: np.ndarray, solved: bool) -> np.ndarray:
    rewards = np.zeros(cube_state.shape[1], dtype=np.float32)
    # Small negative reward for each step to encourage shorter solutions
    rewards = rewards - 0.1

    # Reward only if the cube is solved at the end of the rollout
    if solved:
        rewards[-1] = 1.0
    return rewards


def compute_advantage_gae(values, rewards, gae_lambda, discount):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    n_steps = len(rewards)

    delta = -values[:-1] + rewards[:-1] + discount * values[1:]
    advantages[-1] = delta[-1]
    for i in range(n_steps-1):
        advantages[n_steps-2-i] = (gae_lambda*discount) * \
            advantages[n_steps-1-i] + delta[n_steps-2-i]

    return advantages


def compute_discounted_return(rewards, discount):
    returns = np.zeros_like(rewards, dtype=np.float32)

    returns[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        returns[i] = returns[i+1] * discount + rewards[i]

    return returns


def update_parameters_ppo(optimizer, acmodel, exps):
    def _compute_policy_loss_ppo(obs, old_logp, actions, advantages):
        policy_loss, approx_kl = 0, 0

        def g(e, A):
            return torch.where(A >= 0, (1+e)*A, (1-e)*A)

        logits, value = acmodel(obs)
        dist = Categorical(logits=logits)

        policy_loss = - torch.mean(torch.min(torch.exp(dist.log_prob(
            actions) - old_logp)*advantages, g(ENTROPY_COEF, advantages)))

        logr = old_logp - dist.log_prob(actions)
        k3 = (logr.exp() - 1) - logr
        approx_kl = k3.mean()

        return policy_loss, approx_kl

    def _compute_value_loss(obs, returns):
        _, values = acmodel(obs)
        loss = nn.MSELoss()
        value_loss = loss(values, returns)

        return value_loss

    obs = torch.tensor(exps['states'].T, dtype=torch.float32)
    actions = torch.tensor(exps['actions'], dtype=torch.float32)

    logits, value = acmodel(obs)
    dist = Categorical(logits=logits)
    old_logp = dist.log_prob(actions).detach()

    advantages = torch.tensor(exps['advantages'], dtype=torch.float32)
    returns = torch.tensor(exps['returns'], dtype=torch.float32)

    policy_loss, _ = _compute_policy_loss_ppo(
        obs, old_logp, actions, advantages)
    value_loss = _compute_value_loss(obs, returns)

    for i in range(AC_TRAINING_STEPS):
        optimizer.zero_grad()
        loss_pi, approx_kl = _compute_policy_loss_ppo(
            obs, old_logp, actions, advantages)
        loss_v = _compute_value_loss(obs, returns)

        loss = loss_v + loss_pi
        if approx_kl > 1.5 * TARGET_KL:
            break

        loss.backward(retain_graph=True)
        optimizer.step()

    update_policy_loss = policy_loss.item()
    update_value_loss = value_loss.item()

    return {
        'policy_loss': update_policy_loss,
        'value_loss': update_value_loss
    }


def collect_experiences(agent: RLAgent, n_cubes: int, n_rollouts_per_cube: int, n_scrambling_moves: int, max_rollout_length: int) -> dict:
    """ 
    Collect experiences by trying to solve randomly scrambled cubes.

    Returns:
        A dictionary containing the following keys:
        - 'states': A list of states encountered during the episode.
        - 'actions': A list of actions taken during the episode.
        - 'rewards': A list of rewards received during the episode.
        - 'advantages': A list of advantages calculated during the episode.
        - 'returns': A list of returns calculated during the episode.
    """

    # Initialize lists to store the experiences
    states = []
    actions = []
    rewards = []
    advantages = []
    returns = []

    solved = 0

    for _ in range(n_cubes):
        # Create a new cube and scramble it
        scrambled_cube = Cube()
        while scrambled_cube.is_solved():  # Ensure the cube is not already solved
            scrambled_cube.scramble(n_scrambling_moves)

        for _ in range(n_rollouts_per_cube):
            cube = scrambled_cube.copy()

            # Try to solve the cube
            iter_count = 0

            agent.reset()
            while not cube.is_solved() and iter_count < max_rollout_length - 1:
                iter_count += 1

                move = agent.get_move(cube)
                cube.move(move)

            agent.get_move(cube)  # Register the terminal state

            states_, actions_, values_ = agent.get_history()

            if cube.is_solved():
                solved += 1

            # Calculate the rewards
            rewards_ = compute_rewards(states_, cube.is_solved())
            returns_ = compute_discounted_return(rewards_, DISCOUNT_FACTOR)
            advantages_ = compute_advantage_gae(
                values_, rewards_, GAE_LAMBDA, DISCOUNT_FACTOR)

            # Append the experiences to the lists
            states.append(states_)
            actions.append(actions_)
            rewards.append(rewards_)
            advantages.append(advantages_)
            returns.append(returns_)

    return {
        'states': np.hstack(states),
        'actions': np.hstack(actions),
        'rewards': np.hstack(rewards),
        'advantages': np.hstack(advantages),
        'returns': np.hstack(returns),
        'solved': solved,
    }
