# RL MiniProject Pacman Group42

This project simulates a Pac-Man game using Q-Learning and visualizes the results in a Pygame application. Pac-Man navigates through a grid, collects rewards, and tries to avoid ghosts. The training process allows both Pac-Man and ghosts to learn through Q-Learning, which helps them optimize their actions in the environment. The project consists of two Python scripts:

1. **training_statistics_write_json.py**: Performs Q-Learning, generates game states, and saves them as JSON files..
2. **pygame_read_json.py**: Visualizes the generated game states using Pygame.

---

## Requirements

- Python 3.7+
  - https://www.python.org/downloads/ 

---

## Installation

1. **Repository klonen**:
   ```bash
   git https://github.com/Adrianus252/RL_MiniProject_Pacman_Group42.git
   cd RL_MiniProject_Pacman_Group42
   ```

2. **Libraries Installation with Pip**:
    ```bash
      pip install ipython
      pip install plotly
      pip install pygame
      pip install numpy
      pip install matplotlib
      pip install seaborn
    ```

## Usage

### Training the Agents
1. Run the training script:
   ```bash
   python training_statistics_write_json.py
   ```
2. Training results:
  - JSON files for each episode stored in the game_states/ directory.
  - Cumulative reward and win statistics are plotted at the end of the training.

### Visualizing the Game
  1. Ensure game state JSON files are available in the game_states/ directory.
  2. Run the visualization script:
     ```bash
     python pygame_read_json.py
  3. Use the Pygame interface to watch Pac-Man and ghosts navigate the environment.

---
## File Structure
```bash
RL_MiniProject_Pacman_Group42/
├─ code 
│ ├─ training_statistics_write_json.py # Generates game environment and trains agents using Q-Learning
│ ├─ pygame_read_json.py               # Visualizes the game states stored as JSON files
├─ sprites/                            # Directory for graphical assets (Pac-Man, Ghosts, Rewards, etc.)
├─ game_states/                        # Directory for saved game state JSON files
└─ README.md                           # Project documentation
```

## Environment Setup
The game environment is represented as an 8x8 grid where:

- Pac-Man and the ghosts move around trying to achieve different goals.
- Rewards are placed randomly on the grid.
- Walls and obstacles are also present to add complexity.

### Key Parameters:
- Grid Size: 8x8 grid
- Rewards: Small rewards (value 5), medium rewards (value 10), and a big reward (value 100).
- Ghosts: Ghosts move through the grid and try to catch Pac-Man.
- Walls: Random walls are placed on the grid to obstruct movement.

### Game Logic:
- Pac-Man can move up, down, left, or right on the grid.
- Pac-Man can collect rewards and must avoid the ghosts.
- Ghosts move in random directions, and if they catch Pac-Man, they win the round.
- Game End: The game ends when Pac-Man collects all rewards, is caught by a ghost, or achieves the big reward goal.

### Actions:
- 0: Up
- 1: Down
- 2: Left
- 3: Right

### Rewards:
- Small reward: +5
- Medium reward: +10
- Big reward: +100
- Move penalty: -1
- Ghost catch penalty: -100

## Q-Learning Details
Both Pac-Man and the ghosts use Q-learning to learn the optimal policy for action selection. The algorithm parameters are:

- Learning Rate (α): 0.3
- Discount Factor (γ): 0.9
- Exploration Rate (ε): Starts at 1.0 and decays over time.
- Exploration Decay: 0.995 per episode.
- Minimum Exploration Rate: 0.01

### Q-Learning Update Rule

The core of the Q-Learning algorithm is the **Q-value update rule**. When an agent performs an action in a given state and receives a reward, the Q-value for the state-action pair is updated using the following formula:

Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α [Rₜ₊₁ + γ * maxₐQ(sₜ₊₁, a') - Q(sₜ, aₜ)]

Where:
- Q(sₜ, aₜ) is the current Q-value for the state sₜ and action aₜ.
- Rₜ₊₁ is the reward received after performing the action.
- γ is the **discount factor**, which determines the importance of future rewards.
- α is the **learning rate**, which indicates how much the Q-value should be updated.
- maxₐ' Q(sₜ₊₁, a') is the maximum Q-value for the next state sₜ₊₁ across all possible actions \a', representing the best future value.

### Training Process:
- Both Pac-Man and the ghosts explore the grid and update their Q-values after each action.
- Pac-Man updates its Q-table based on the reward received after moving to the next position.
- Ghosts update their Q-table based on whether they catch Pac-Man or just move.
- The exploration rate (ε) decreases with each episode to encourage more exploitation of learned strategies.

## Results
After the training loop, the average cumulative rewards for Pac-Man and the ghosts are calculated and displayed. The results include the following:

- **Average Rewards Collected:** Displays the average rewards collected by Pac-Man.
- **Win Conditions:** Tracks the number of times Pac-Man wins by collecting all small, medium, or big rewards, and the number of times ghosts win by catching Pac-Man.
- **Episode Summary:** For each episode, the rewards, wins, and positions of Pac-Man and the ghosts are displayed.

## Visualizations
Several plots are generated to visualize the training results:

1. **Training Performance Plot:** Cumulative rewards for Pac-Man and the ghosts over the training episodes.
2. **Win Statistics:** A bar chart showing the number of wins by Pac-Man and ghosts.
3. **Heatmaps:** Heatmaps of state visit counts for both Pac-Man and the ghosts, highlighting areas visited frequently.