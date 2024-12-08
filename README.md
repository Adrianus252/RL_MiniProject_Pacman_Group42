# Pac-Man Q-Learning Visualization Project

This project simulates a Pac-Man game using Q-Learning and visualizes the results in a Pygame application. Pac-Man navigates through a grid, collects rewards, and tries to avoid ghosts. The project consists of two Python scripts:

1. **Training-Skript**: Performs Q-Learning, generates game states, and saves them as JSON files..
2. **Visualisierungsskript**: Visualizes the generated game states using Pygame.

---

## Requirements

- Python 3.7+
- Abhängigkeiten: 
  - `pygame`
  - `numpy`
  - `matplotlib`

---

## Installation

1. **Repository klonen**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

---
## Usage

  **Training the Agents**
1. Run the training script:
   ```bash
   python pacgh_mg_save_json.py
   ```
2. Training results:
  - JSON files for each episode stored in the game_states/ directory.
  - Cumulative reward and win statistics are plotted at the end of the training.

---
## Visualizing the Game
  1. Ensure game state JSON files are available in the game_states/ directory.
  2. Run the visualization script:
     ```bash
     python pacgh_mg_pygame_sprite.py
  3. Use the Pygame interface to watch Pac-Man and ghosts navigate the environment.

---
## File Structure
```bash
Pac-Man-Qlearning/
│
├── training_script.py         # Generates game environment and trains agents using Q-Learning
├── visualization_script.py    # Visualizes the game states stored as JSON files
├── sprites/                   # Directory for graphical assets (Pac-Man, Ghosts, Rewards, etc.)
├── game_states/               # Directory for saved game state JSON files
└── README.md                  # Project documentation
```

---

## How It Works
**Training**
- Environment Setup:
  - The environment is an 8x8 grid with walls, rewards, and ghosts placed randomly.
  - Rewards:
    - Small Rewards: +5 points
    - Medium Rewards: +10 points
    - Big Reward: +100 points
  - Ghosts: -100 points if they catch Pac-Man.
- Actions:
  - Pac-Man and ghosts move in one of four directions: Up, Down, Left, Right.
- Q-Learning:
  - Agents learn the optimal strategy based on rewards and penalties over 1,000 episodes.

**Visualization**
- The pacgh_mg_pygame_sprite.py uses the saved game states from training to render the gameplay in Pygame.
- Game elements (Pac-Man, ghosts, walls, rewards) are displayed on a grid, and the progress is shown step-by-step.
