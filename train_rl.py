"""
Training script for Rubik's Cube using Stable-Baselines3 with DQN and HER.
"""
import numpy as np
from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Type
import gymnasium as gym

from cube_env import RubiksCubeGoalEnv


def make_env(
    max_steps: int = 100,
    initial_scramble_moves_range: Tuple[int, int] = (1, 20),
    scramble_moves_range: Tuple[int, int] = (1, 20),
):
    """Create and wrap the environment."""
    def _init():
        env = RubiksCubeGoalEnv(
            max_steps=max_steps,
            initial_scramble_moves_range=initial_scramble_moves_range,
            scramble_moves_range=scramble_moves_range,
        )
        env = Monitor(env)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning that progressively increases scramble difficulty.

    The curriculum starts with easy scrambles (e.g., 1 move) and increases the difficulty
    once the agent achieves a target success rate.
    """

    def __init__(
        self,
        eval_env: DummyVecEnv,
        start_scramble: int = 1,
        max_scramble: int = 20,
        success_threshold: float = 0.8,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        """
        Args:
            eval_env: Evaluation environment
            start_scramble: Initial scramble difficulty (number of moves)
            max_scramble: Maximum scramble difficulty
            success_threshold: Success rate to reach before increasing difficulty (0-1)
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes to evaluate
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_scramble = start_scramble
        self.max_scramble = max_scramble
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        self.evaluation_history = []

    def _calculate_max_steps(self, scramble_moves: int) -> int:
        """Calculate maximum steps based on scramble difficulty.

        Formula: For 1 move -> 3 steps, for 2 moves -> 5 steps
        Pattern: max_steps = 2 * scramble_moves + 1
        """
        return 2 * scramble_moves + 1

    def _update_env_difficulty(self, new_scramble: int):
        """Update scramble difficulty and max steps for all environments."""
        new_max_steps = self._calculate_max_steps(new_scramble)

        for env_idx in range(self.training_env.num_envs):
            env = self.training_env.envs[env_idx]
            # Access the actual RubiksCubeGoalEnv (unwrap Monitor)
            if hasattr(env, 'env'):
                actual_env = env.env
            else:
                actual_env = env
            actual_env.scramble_moves_range = (new_scramble, new_scramble)
            actual_env.max_steps = new_max_steps

        # Update eval env as well
        for env_idx in range(self.eval_env.num_envs):
            env = self.eval_env.envs[env_idx]
            if hasattr(env, 'env'):
                actual_env = env.env
            else:
                actual_env = env
            actual_env.scramble_moves_range = (new_scramble, new_scramble)
            actual_env.max_steps = new_max_steps

    def _evaluate(self) -> float:
        """Evaluate the agent and return success rate."""
        # Set model to eval mode to disable dropout
        self.model.policy.set_training_mode(False)

        successes = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, info = self.eval_env.step(action)
                # For vectorized envs, dones is an array
                done = dones[0] if isinstance(dones, np.ndarray) else dones

            # Check if solved (info is a list for vectorized env)
            if isinstance(info, list):
                if info[0].get('is_solved', False):
                    successes += 1
            else:
                if info.get('is_solved', False):
                    successes += 1

        # Set model back to training mode
        self.model.policy.set_training_mode(True)

        return successes / self.n_eval_episodes

    def _on_step(self) -> bool:
        """Check if we should evaluate and potentially increase difficulty."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps

            # Evaluate current performance
            success_rate = self._evaluate()
            self.evaluation_history.append({
                'timestep': self.num_timesteps,
                'scramble_difficulty': self.current_scramble,
                'success_rate': success_rate,
            })

            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Curriculum Evaluation @ {self.num_timesteps:,} steps")
                print(f"Current difficulty: {self.current_scramble} moves")
                print(f"Success rate: {success_rate*100:.1f}%")
                print(f"{'='*60}\n")

            # Check if we should increase difficulty
            if success_rate >= self.success_threshold and self.current_scramble < self.max_scramble:
                self.current_scramble += 1
                self._update_env_difficulty(self.current_scramble)
                new_max_steps = self._calculate_max_steps(
                    self.current_scramble)

                if self.verbose > 0:
                    print(
                        f"\n🎓 CURRICULUM ADVANCE! Difficulty: {self.current_scramble} moves, Max steps: {new_max_steps}\n")

            # Log to tensorboard if available
            if self.logger is not None:
                self.logger.record(
                    "curriculum/scramble_difficulty", self.current_scramble)
                self.logger.record("curriculum/success_rate", success_rate)

        return True


class CubeAttentionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with attention mechanism for processing cube states.

    This extracts features from the dict observation space containing
    'observation', 'achieved_goal', and 'desired_goal'.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, embed_dim: int = 64):
        # Features dim is the output size of this extractor
        super().__init__(observation_space, features_dim)

        self.embed_dim = embed_dim
        self.num_colors = 6

        # Position embedding for 108 stickers (54 from current + 54 from goal)
        self.position_embedding = nn.Parameter(torch.randn(108, embed_dim))

        # Linear projection from one-hot encoding to embed_dim
        # One-hot: 6 dimensions per sticker
        self.color_projection = nn.Linear(self.num_colors, embed_dim)

        # Transformer encoder for attention between current and goal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Project pooled transformer output to feature space
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to extract features.

        Args:
            observations: Dict with 'observation', 'desired_goal'
        Returns:
            features: (batch, features_dim) tensor
        """
        batch_size = observations['observation'].shape[0]

        # Concatenate current and goal cubes along sequence dimension
        # Current: stickers 0-53, Goal: stickers 54-107
        combined_states = torch.cat([
            observations['observation'],
            observations['desired_goal']
        ], dim=1)  # (batch, 108)

        # Convert to one-hot encoding
        # combined_states: (batch, 108) with values 0-5
        one_hot = torch.nn.functional.one_hot(
            combined_states.long(),
            num_classes=self.num_colors
        ).float()  # (batch, 108, 6)

        # Project one-hot to embedding dimension
        embedded = self.color_projection(one_hot)  # (batch, 108, embed_dim)

        # Add positional encoding (different for each position)
        embedded = embedded + self.position_embedding.unsqueeze(0)

        # Apply transformer - attention across all stickers from both cubes
        transformed = self.transformer(embedded)  # (batch, 108, embed_dim)

        # Global average pooling
        pooled = transformed.mean(dim=1)  # (batch, embed_dim)

        # Project to feature space
        features = self.feature_projection(pooled)

        return features


def train_dqn_her(
    total_timesteps: int = 100_000,
    initial_scramble_moves_range: Tuple[int, int] = (1, 20),
    scramble_moves_range: Tuple[int, int] = (1, 20),
    max_steps: int = 100,
    learning_rate: float = 1e-2,
    buffer_size: int = 100_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    save_path: str = "./models/rubiks_dqn_her",
    log_path: str = "./logs/rubiks_dqn_her",
    use_her: bool = True,
    use_curriculum: bool = False,
    curriculum_start_scramble: int = 1,
    curriculum_max_scramble: int = 20,
    curriculum_success_threshold: float = 0.8,
    curriculum_eval_freq: int = 5_000,
    curriculum_eval_episodes: int = 20,
):
    """
    Train DQN with HER on Rubik's Cube environment.

    Args:
        total_timesteps: Total number of timesteps to train
        initial_scramble_moves_range: Range of scramble moves for initial cube states
        scramble_moves_range: Range of scramble moves for goal states
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        gamma: Discount factor
        exploration_fraction: Fraction of training for epsilon decay
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        target_update_interval: Steps between target network updates
        save_path: Path to save model checkpoints
        log_path: Path to save logs
        use_her: Whether to use HER
        use_curriculum: Whether to use curriculum learning
        curriculum_start_scramble: Starting scramble difficulty for curriculum
        curriculum_max_scramble: Maximum scramble difficulty for curriculum
        curriculum_success_threshold: Success rate threshold to advance curriculum (0-1)
        curriculum_eval_freq: Evaluate curriculum progress every N steps
        curriculum_eval_episodes: Number of episodes for curriculum evaluation
    """
    print("="*60)
    print("Training Rubik's Cube Solver with DQN" +
          (" + HER" if use_her else "") +
          (" + Curriculum Learning" if use_curriculum else ""))
    print("="*60)
    print(f"Total timesteps: {total_timesteps:,}")
    if use_curriculum:
        # Calculate initial max steps based on starting scramble
        curriculum_initial_max_steps = 2 * curriculum_start_scramble + 1
        print(
            f"Curriculum: {curriculum_start_scramble} → {curriculum_max_scramble} moves")
        print(f"Success threshold: {curriculum_success_threshold*100:.0f}%")
        print(
            f"Initial max steps: {curriculum_initial_max_steps} (adapts with difficulty)")
    else:
        print(f"Scramble moves: {scramble_moves_range}")
        print(f"Max steps per episode: {max_steps}")
    print(f"Using HER: {use_her}")
    print("="*60)

    # Create environment (with curriculum or fixed difficulty)
    if use_curriculum:
        # Start with easiest difficulty and appropriate max steps
        curriculum_initial_max_steps = 2 * curriculum_start_scramble + 1
        env = DummyVecEnv([make_env(
            max_steps=curriculum_initial_max_steps,
            initial_scramble_moves_range=initial_scramble_moves_range,
            scramble_moves_range=(curriculum_start_scramble,
                                  curriculum_start_scramble),
        )])
        eval_env = DummyVecEnv([make_env(
            max_steps=curriculum_initial_max_steps,
            initial_scramble_moves_range=initial_scramble_moves_range,
            scramble_moves_range=(curriculum_start_scramble,
                                  curriculum_start_scramble),
        )])
    else:
        # Use fixed difficulty range
        env = DummyVecEnv([make_env(
            max_steps=max_steps,
            initial_scramble_moves_range=initial_scramble_moves_range,
            scramble_moves_range=scramble_moves_range,
        )])
        eval_env = DummyVecEnv([make_env(
            max_steps=max_steps,
            initial_scramble_moves_range=initial_scramble_moves_range,
            scramble_moves_range=scramble_moves_range,
        )])

    # Configure replay buffer (with or without HER)
    if use_her:
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=4,  # Number of virtual transitions per real transition
            goal_selection_strategy='future',  # Use future strategy for HER
        )
    else:
        replay_buffer_class = None
        replay_buffer_kwargs = None

    # Create policy kwargs with custom attention-based feature extractor
    policy_kwargs = dict(
        features_extractor_class=CubeAttentionExtractor,
        features_extractor_kwargs=dict(),
        # net_arch=[64],  # Q-network architecture after feature extraction
    )

    # Create DQN model
    model = DQN(
        policy="MultiInputPolicy",  # Use MultiInputPolicy for dict observations
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_path,
        device="cuda",  # Use GPU for faster training
    )
    # model = DQN(
    #     'MultiInputPolicy',
    #     env,
    #     learning_rate=1e-3,
    #     buffer_size=10000,
    #     learning_starts=100,
    #     batch_size=32,
    #     gamma=0.99,
    #     exploration_fraction=0.5,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.1,
    #     verbose=0,
    #     device='cuda',
    # )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path,
        name_prefix="rubiks_dqn_her" if use_her else "rubiks_dqn",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Prepare callback list
    callbacks = [checkpoint_callback, eval_callback]

    # Add curriculum callback if enabled
    if use_curriculum:
        curriculum_callback = CurriculumCallback(
            eval_env=eval_env,
            start_scramble=curriculum_start_scramble,
            max_scramble=curriculum_max_scramble,
            success_threshold=curriculum_success_threshold,
            eval_freq=curriculum_eval_freq,
            n_eval_episodes=curriculum_eval_episodes,
            verbose=1,
        )
        callbacks.append(curriculum_callback)

    # Train the model
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True,
    )

    # Save final model
    final_save_path = f"{save_path}/final_model"
    model.save(final_save_path)
    print(f"\nTraining completed! Final model saved to {final_save_path}")

    return model, env


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    initial_scramble_moves_range: Tuple[int, int] = (1, 20),
    scramble_moves_range: Tuple[int, int] = (1, 20),
    max_steps: int = 100,
    render: bool = False,
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        initial_scramble_moves_range: Range of scramble moves for initial state
        scramble_moves_range: Range of scramble moves for goal state
        max_steps: Maximum steps per episode
        render: Whether to render episodes
    """
    print(f"\nEvaluating model: {model_path}")
    print(
        f"Episodes: {n_episodes}, Initial scramble: {initial_scramble_moves_range}, Goal scramble: {scramble_moves_range}")

    # Create environment first (needed for HER models)
    env = RubiksCubeGoalEnv(
        max_steps=max_steps,
        initial_scramble_moves_range=initial_scramble_moves_range,
        scramble_moves_range=scramble_moves_range,
    )

    # Load model (pass env for HER compatibility)
    model = DQN.load(model_path, env=env, device="cuda")

    # Evaluate
    successes = 0
    total_steps = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if render:
                env.render()

        if info['is_solved']:
            successes += 1
        total_steps.append(steps)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Success rate: {successes / (episode + 1) * 100:.1f}%, "
                  f"Avg steps: {np.mean(total_steps):.1f}")

    success_rate = successes / n_episodes * 100
    avg_steps = np.mean(total_steps)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Min steps: {np.min(total_steps)}")
    print(f"Max steps: {np.max(total_steps)}")
    print("="*60)

    env.close()

    return success_rate, avg_steps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or evaluate Rubik's Cube RL agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--initial-scramble-min", type=int, default=1,
                        help="Minimum initial scramble moves")
    parser.add_argument("--initial-scramble-max", type=int, default=20,
                        help="Maximum initial scramble moves")
    parser.add_argument("--scramble-min", type=int, default=1,
                        help="Minimum goal scramble moves")
    parser.add_argument("--scramble-max", type=int, default=20,
                        help="Maximum goal scramble moves")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--use-her", action="store_true", default=True,
                        help="Use HER (Hindsight Experience Replay)")
    parser.add_argument("--use-curriculum", action="store_true",
                        help="Use curriculum learning")
    parser.add_argument("--curriculum-start", type=int, default=1,
                        help="Starting scramble difficulty for curriculum")
    parser.add_argument("--curriculum-max", type=int, default=20,
                        help="Maximum scramble difficulty for curriculum")
    parser.add_argument("--curriculum-threshold", type=float, default=0.8,
                        help="Success rate threshold to advance curriculum (0-1)")
    parser.add_argument("--curriculum-eval-freq", type=int, default=5000,
                        help="Evaluate curriculum every N steps")
    parser.add_argument("--curriculum-eval-episodes", type=int, default=20,
                        help="Number of episodes for curriculum evaluation")
    parser.add_argument("--model-path", type=str, default="./models/rubiks_dqn_her/final_model",
                        help="Path to model (for eval mode)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")

    args = parser.parse_args()

    if args.mode == "train":
        train_dqn_her(
            total_timesteps=args.timesteps,
            initial_scramble_moves_range=(
                args.initial_scramble_min, args.initial_scramble_max),
            scramble_moves_range=(args.scramble_min, args.scramble_max),
            max_steps=args.max_steps,
            use_her=args.use_her,
            # use_her=False,
            use_curriculum=args.use_curriculum,
            curriculum_start_scramble=args.curriculum_start,
            curriculum_max_scramble=args.curriculum_max,
            curriculum_success_threshold=args.curriculum_threshold,
            curriculum_eval_freq=args.curriculum_eval_freq,
            curriculum_eval_episodes=args.curriculum_eval_episodes,
        )
    elif args.mode == "eval":
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            initial_scramble_moves_range=(
                args.initial_scramble_min, args.initial_scramble_max),
            scramble_moves_range=(args.scramble_min, args.scramble_max),
            max_steps=args.max_steps,
            render=args.render,
        )
