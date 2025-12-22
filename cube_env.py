"""
Gym environment for Rubik's Cube with goal-based learning support for HER.
Compatible with Stable-Baselines3.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from cube import Cube, moves, moves_idx, Move


class RubiksCubeGoalEnv(gym.Env):
    """
    Gym environment for Rubik's Cube with goal-based learning.

    This environment supports HER (Hindsight Experience Replay) by exposing
    'achieved_goal' and 'desired_goal' in the observation space.

    The agent learns to transition from any cube state (achieved_goal) to any
    target state (desired_goal).
    """

    def __init__(
        self,
        max_steps: int = 100,
        initial_scramble_moves_range: Tuple[int, int] = (1, 20),
        scramble_moves_range: Tuple[int, int] = (1, 20)
    ):
        super().__init__()

        self.max_steps = max_steps
        self.initial_scramble_moves_range = initial_scramble_moves_range
        self.scramble_moves_range = scramble_moves_range

        # Initialize cubes
        self.current_cube = Cube()
        self.goal_cube = self.current_cube.copy()

        # Action space: 18 possible moves (6 faces * 3 specifiers)
        self.action_space = spaces.Discrete(len(moves))

        # Observation space: dict with observation, achieved_goal, desired_goal
        # Each cube state is represented as 54 integers (6 faces * 9 stickers, flattened)
        obs_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.uint8)

        self.observation_space = spaces.Dict({
            'observation': obs_space,
            'achieved_goal': obs_space,
            'desired_goal': obs_space,
        })

        self.current_step = 0

    @staticmethod
    def _get_cube_state(cube: Cube) -> np.ndarray:
        """Convert cube to flat array representation."""
        return cube._cube.flatten().astype(np.uint8)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation dict for HER."""
        current_state = self._get_cube_state(self.current_cube)
        goal_state = self._get_cube_state(self.goal_cube)

        return {
            'observation': current_state.copy(),
            'achieved_goal': current_state.copy(),
            'desired_goal': goal_state.copy(),
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            'is_solved': self.current_cube == self.goal_cube,
            'current_step': self.current_step,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Compute the reward for HER.

        This method must handle both single samples and batches.
        """
        # Handle batched inputs
        if achieved_goal.ndim == 2:
            # Batch of goals: compare each achieved with desired
            is_equal = np.all(achieved_goal == desired_goal, axis=1)
            return np.where(is_equal, 10.0, -1.0).astype(np.float32)
        else:
            # Single goal
            is_equal = np.array_equal(achieved_goal, desired_goal)
            return 10.0 if is_equal else -1.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        # Reset step counter
        self.current_step = 0

        # Generate initial cube state
        n_initial_scrambles = int(self.np_random.integers(
            self.initial_scramble_moves_range[0],
            self.initial_scramble_moves_range[1] + 1
        ))
        self.current_cube = Cube(n_scramble_moves=n_initial_scrambles)
        n_scrambles = int(self.np_random.integers(
            self.scramble_moves_range[0],
            self.scramble_moves_range[1] + 1
        ))
        self.goal_cube = self.current_cube.copy()
        self.goal_cube.scramble(n_scrambles)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action index (0-17 for 18 possible moves)

        Returns:
            observation: Dict with observation, achieved_goal, desired_goal
            reward: Scalar reward
            terminated: Whether the episode is done (goal reached)
            truncated: Whether the episode is truncated (max steps)
            info: Additional information
        """
        # Execute move
        move = moves[action]
        self.current_cube.move(move)

        self.current_step += 1

        # Get observation
        observation = self._get_obs()
        info = self._get_info()

        # Compute reward
        reward = self.compute_reward(
            observation['achieved_goal'],
            observation['desired_goal'],
            info
        )

        # Check termination
        terminated = (self.current_cube == self.goal_cube)
        truncated = (self.current_step >= self.max_steps)

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        self.current_cube.plot_3d()
