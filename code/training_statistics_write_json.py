import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
from copy import deepcopy
import seaborn as sns

# Define environment parameters
GRID_SIZE = 8
NUM_SMALL_REWARDS = 3
NUM_MEDIUM_REWARDS = 2
NUM_GHOSTS = 1
NUM_WALLS = 4

# Rewards
SMALL_REWARD_VALUE = 5
MEDIUM_REWARD_VALUE = 10
BIG_REWARD_VALUE = 100
PACMAN_MOVE_PENALTY = -1
PACMAN_GHOST_PENALTY = -100

GHOST_MOVE_PENALTY = -1
GHOST_CATCH_REWARD = 100

# Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
ACTIONS = [0, 1, 2, 3]

# Q-Learning Parameters
ALPHA = 0.3  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995  # Exploration decay
MIN_EPSILON = 0.01

EPISODES_COUNT = 1000


RANDOM_PACMAN_START = False  # Set to True for random position every episode, False for a fixed start

# Initialize Pac-Man's fixed start position if RANDOM_PACMAN_START is False
fixed_start_position = None


# Helper functions
def create_environment():
    """Create a grid environment with goals, ghosts, and walls."""
    env = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Place walls
    walls = random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], NUM_WALLS)
    for wall in walls:
        env[wall] = -2

    # Check available positions after placing walls
    available_positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if (i, j) not in walls]
    print(f"Available positions after walls: {len(available_positions)}")

    # Place big reward (main goal)
    big_reward = random.choice(available_positions)
    env[big_reward] = 100

    # Remove used position for big reward
    available_positions = [pos for pos in available_positions if pos != big_reward]
    print(f"Available positions after big reward: {len(available_positions)}")

    # Place medium rewards
    medium_rewards = random.sample(available_positions, min(NUM_MEDIUM_REWARDS, len(available_positions)))
    for pos in medium_rewards:
        env[pos] = 10

    # Remove used positions for medium rewards
    available_positions = [pos for pos in available_positions if pos not in medium_rewards]
    print(f"Available positions after medium rewards: {len(available_positions)}")

    # Place small rewards
    small_rewards = random.sample(available_positions, min(NUM_SMALL_REWARDS, len(available_positions)))
    for pos in small_rewards:
        env[pos] = 5

    # Remove used positions for small rewards
    available_positions = [pos for pos in available_positions if pos not in small_rewards]
    print(f"Available positions after small rewards: {len(available_positions)}")

    # Place ghosts
    ghosts = random.sample(available_positions, NUM_GHOSTS)
    for ghost in ghosts:
        env[ghost] = -1

    print(f"Environment created with:\n  Walls: {walls}\n  Big Reward: {big_reward}\n"
          f"  Medium Rewards: {medium_rewards}\n  Small Rewards: {small_rewards}\n  Ghosts: {ghosts}")
    return env, big_reward, medium_rewards, small_rewards, ghosts, walls

def is_valid_position(pos, walls):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in walls

def get_next_position(pos, action, walls):
    x, y = pos
    if action == 0:  # Up
        next_pos = (x - 1, y)
    elif action == 1:  # Down
        next_pos = (x + 1, y)
    elif action == 2:  # Left
        next_pos = (x, y - 1)
    elif action == 3:  # Right
        next_pos = (x, y + 1)
    return next_pos if is_valid_position(next_pos, walls) else pos

def get_pacman_reward(position, big_reward, medium_rewards, small_rewards, ghosts):
    """Get Pac-Man's reward for the current position."""
    if position == big_reward:
        return BIG_REWARD_VALUE
    elif position in medium_rewards:
        medium_rewards.remove(position)  # Remove collected medium reward
        return MEDIUM_REWARD_VALUE
    elif position in small_rewards:
        small_rewards.remove(position)  # Remove collected small reward
        return SMALL_REWARD_VALUE
    elif position in ghosts:
        return PACMAN_GHOST_PENALTY
    return PACMAN_MOVE_PENALTY

def plot_training_performance(pacman_rewards, ghost_rewards):
    """Plot cumulative rewards for Pac-Man and Ghosts side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Pac-Man's rewards
    axes[0].plot(pacman_rewards, label="Pac-Man Rewards", color='blue')
    axes[0].set_title("Pac-Man Cumulative Rewards")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].grid()
    axes[0].legend()

    # Plot Ghosts' rewards
    axes[1].plot(ghost_rewards, label="Ghost Rewards", color='red')
    axes[1].set_title("Ghosts Cumulative Rewards")
    axes[1].set_xlabel("Episodes")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_win_statistics(pacman_wins, ghost_wins):
    """Plot win statistics for Pac-Man and Ghosts."""
    labels = ["Pac-Man Wins", "Ghost Wins"]
    counts = [pacman_wins, ghost_wins]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=["blue", "red"])
    plt.title("Win Statistics")
    plt.ylabel("Number of Wins")
    plt.show()

# Create a directory to store game state files
os.makedirs("game_states", exist_ok=True)

def save_game_state(filename, state):
    """Save the game state to a JSON file."""
    with open(filename, "w") as file:
        json.dump(state, file, indent=4)
def plot_pacman_win_conditions(pacman_win_conditions):
    labels = list(pacman_win_conditions.keys())
    counts = list(pacman_win_conditions.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=["green", "blue", "orange"])
    plt.title("Pac-Man Win Conditions")
    plt.ylabel("Number of Wins")
    plt.xlabel("Win Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_annotated_heatmap(visit_counts, title, BIG_highlight_position, MEDIUM_highlight_position, SMALL_highlight_position, cmap="Blues"):
    """Plots a heatmap and annotates a specific field."""
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(visit_counts, cmap=cmap, annot=True, cbar=True, square=True, annot_kws={"size": 5})


    # Annotate the highlighted cell
    ax.text(BIG_highlight_position[1] + 0.5, BIG_highlight_position[0] + 0.5, 'L', 
            ha='center', va='center', color="red", fontsize=16)
    
    for x in MEDIUM_highlight_position:
      ax.text(x[1] + 0.5, x[0] + 0.5, 'M', 
            ha='center', va='center', color="red", fontsize=16)
    for x in SMALL_highlight_position:
      ax.text(x[1] + 0.5, x[0] + 0.5, 'S', 
            ha='center', va='center', color="red", fontsize=16)

    plt.title(title)
    plt.xlabel("X (Grid Column)")
    plt.ylabel("Y (Grid Row)")
    plt.show()

# Initialize environment
original_env, original_big_reward, original_medium_rewards, original_small_rewards, original_ghosts, original_walls = create_environment()

# Initialize Q-tables
pacman_Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
ghosts_Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Training loop
os.makedirs("game_states", exist_ok=True)
pacman_rewards_per_episode = []
ghost_rewards_per_episode = []
pacman_wins = 0
ghost_wins = 0
total_small_rewards_collected = []
total_medium_rewards_collected = []

# Initialize Variables for Headmaps
pacman_visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))
ghost_visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))

pacman_win_conditions = {"big_reward": 0, "all_medium_rewards": 0, "all_small_rewards": 0}

for episode in range(EPISODES_COUNT):
    # Reset environment for each episode
    env = original_env.copy()
    big_reward = original_big_reward
    medium_rewards = deepcopy(original_medium_rewards)
    small_rewards = deepcopy(original_small_rewards)
    ghosts = deepcopy(original_ghosts)
    walls = deepcopy(original_walls)

    initial_small_rewards = len(small_rewards)
    initial_medium_rewards = len(medium_rewards)


    # Generate Pac-Man's position
    if RANDOM_PACMAN_START:
        position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        while (
            position in walls or 
            position == big_reward or 
            position in ghosts or 
            position in small_rewards or 
            position in medium_rewards
        ):
            position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    else:
        if episode == 0 or fixed_start_position is None:
            fixed_start_position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            while (
                fixed_start_position in walls or 
                fixed_start_position == big_reward or 
                fixed_start_position in ghosts or 
                fixed_start_position in small_rewards or 
                fixed_start_position in medium_rewards
            ):
                fixed_start_position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        position = fixed_start_position

    # position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    # while (
    #     position in walls or 
    #     position == big_reward or 
    #     position in ghosts or 
    #     position in small_rewards or 
    #     position in medium_rewards
    # ):
    # #while position in walls or position == big_reward or position in ghosts:
    #     position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    print(f"\n--- Episode {episode + 1} ---")

    pacman_cumulative_reward = 0
    ghost_cumulative_reward = 0
    done = False
    step = 0

    game_states = []  # Store game states for this episode

    # Save the initial game state (Step 0)
    initial_state = {
        "step": 0,  # Step 0 indicates the initial state before any moves
        "pacman_position": position,
        "ghost_positions": ghosts,
        "walls": walls,
        "big_reward": big_reward,
        "medium_rewards": deepcopy(medium_rewards),
        "small_rewards": deepcopy(small_rewards),
        "grid": env.tolist(),
        "pacman_cumulative_reward": 0,
        "ghost_cumulative_reward": 0,
    }
    game_states.append(initial_state)  # Add the initial state to the game states

    # Print inital step 0
    # print(f"\n Initial Episode State {episode}, Step {step}:")
    # print(f"Pac-Man Wins: {pacman_wins}")
    # print(f"Ghost Wins: {ghost_wins}")
    # print(f"  Pac-Man Position: {position}")
    # print(f"  Ghost Positions: {ghosts}")
    # print(f"  Small Reward Positions: {small_rewards}")
    # print(f"  Medium Reward Positions: {medium_rewards}")
    # print(f"  Big Reward Position: {big_reward}")


    while not done:
        step += 1

        # Deepcopy rewards for current state
        current_medium_rewards = deepcopy(medium_rewards)
        current_small_rewards = deepcopy(small_rewards)

        # Pac-Man chooses an action
        if random.uniform(0, 1) < EPSILON:
            pacman_action = random.choice(ACTIONS)  # Explore
        else:
            x, y = position
            pacman_action = np.argmax(pacman_Q_table[x, y])  # Exploit

        # Ghosts choose their actions
        ghost_actions = []
        for ghost in ghosts:
            if random.uniform(0, 1) < EPSILON:
                ghost_action = random.choice(ACTIONS)  # Explore
            else:
                x, y = ghost
                ghost_action = np.argmax(ghosts_Q_table[x, y])  # Exploit
            ghost_actions.append(ghost_action)

        # Update Pac-Man's position
        next_position = get_next_position(position, pacman_action, walls)
        pacman_reward = get_pacman_reward(next_position, big_reward, medium_rewards, small_rewards, ghosts)
        pacman_cumulative_reward += pacman_reward

        # Update ghosts' positions
        new_ghosts = []
        ghost_reward_given = False  # Track if reward has been given for catching Pac-Man

        for ghost, ghost_action in zip(ghosts, ghost_actions):
            new_ghost_pos = get_next_position(ghost, ghost_action, walls)

            if not ghost_reward_given and new_ghost_pos == position:  # Ghost catches Pac-Man
                ghost_reward = GHOST_CATCH_REWARD  # Assign the capture reward
                ghost_cumulative_reward += ghost_reward
                ghost_reward_given = True  # Ensure the reward is given only once
                #done = True  # End the episode
            else:
                ghost_reward = GHOST_MOVE_PENALTY  # Apply move penalty
                ghost_cumulative_reward += ghost_reward

            # Update Ghost's Q-table
            gx, gy = ghost
            ngx, ngy = new_ghost_pos
            ghosts_Q_table[gx, gy, ghost_action] += ALPHA * (
                ghost_reward + GAMMA * np.max(ghosts_Q_table[ngx, ngy]) - ghosts_Q_table[gx, gy, ghost_action]
            )

            new_ghosts.append(new_ghost_pos)

        ghosts = new_ghosts

        # Update Pac-Man's Q-table
        x, y = position
        nx, ny = next_position
        pacman_Q_table[x, y, pacman_action] += ALPHA * (
            pacman_reward + GAMMA * np.max(pacman_Q_table[nx, ny]) - pacman_Q_table[x, y, pacman_action]
        )

        # Update position
        position = next_position

        
        # Check if the game is done
        #if position == big_reward:
        # check whether medium_reward or small_rewards is empty -> pacman collected all. 
        #if position == big_reward or not medium_rewards or not small_rewards: 
        if position == big_reward:
            pacman_wins += 1
            print(f"Pac-Man reached the big goal! Episode {episode + 1}, Step {step}")
            pacman_win_conditions["big_reward"] += 1
            done = True
        elif not medium_rewards:
            pacman_wins += 1
            print(f"Pac-Man reached all medium goals! Episode {episode + 1}, Step {step}")
            pacman_win_conditions["all_medium_rewards"] += 1
            done = True
        elif not small_rewards:
            pacman_wins += 1
            print(f"Pac-Man reached all small goals! Episode {episode + 1}, Step {step}")
            pacman_win_conditions["all_small_rewards"] += 1
            done = True
        elif position in ghosts:
            # if not ghost_reward_given:  # Ensure only one reward is given
            ghost_wins += 1
            print(f"Pac-Man was caught by a ghost! Episode {episode + 1}, Step {step}")
            done = True
        # else :
        #     print(f"No winner? Episode {episode + 1}, Step {step}")

        # Print game state for each step
        #print(f"\nEpisode {episode + 1}, Step {step}:")
        # if step == 1:
        #if not medium_rewards:
        print(f"\n Episode {episode + 1}, Step {step}:")
        print(f"Pac-Man Wins: {pacman_wins}")
        print(f"Ghost Wins: {ghost_wins}")
        print(f"  Pac-Man Position: {position}")
        print(f"  Ghost Positions: {ghosts}")
        print(f"  Small Reward Positions: {small_rewards}")
        print(f"  Medium Reward Positions: {medium_rewards}")
        print(f"  Big Reward Position: {big_reward}")


        # Save the current game state
        current_state = {
            "step": step,
            "pacman_position": position,
            "ghost_positions": ghosts,
            "walls": walls,
            "big_reward": big_reward,
            "medium_rewards": deepcopy(medium_rewards),
            "small_rewards": deepcopy(small_rewards),
            "grid": env.tolist(),
            "pacman_cumulative_reward": pacman_cumulative_reward,
            "ghost_cumulative_reward": ghost_cumulative_reward,
        }
        game_states.append(current_state)


    # new 
    # print(f"Loop exited on Episode {episode + 1}, Step {step}. Done = {done}")
    #if not done:
        # print(f"Loop ended unexpectedly in Episode {episode + 1}, Step {step} with done=False.")
        # print("Game state at termination:")
    # print(f"  Pac-Man Position: {position}")
    # print(f"  Ghost Positions: {ghosts}")
    # print(f"  Small Reward Positions: {small_rewards}")
    # print(f"  Medium Reward Positions: {medium_rewards}")
    # print(f"  Big Reward Position: {big_reward}")
    # print(f"  Pac-Man Cumulative Reward: {pacman_cumulative_reward}")
    # print(f"  Ghost Cumulative Reward: {ghost_cumulative_reward}")
    # print(f"  Walls: {walls}")

    # Save game states to a file
    save_game_state(f"game_states/game_states_episode_{episode + 1}.json", game_states)

    # Save pos for Headmaps
    pacman_visit_counts[position[0], position[1]] += 1

    for ghost in ghosts:
        ghost_visit_counts[ghost[0], ghost[1]] += 1

    # Calculate rewards collected
    small_rewards_collected = initial_small_rewards - len(small_rewards)
    medium_rewards_collected = initial_medium_rewards - len(medium_rewards)
    total_small_rewards_collected.append(small_rewards_collected)
    total_medium_rewards_collected.append(medium_rewards_collected)

    print(f"Episode {episode + 1}:")
    print(f"  Small Rewards Collected: {small_rewards_collected}/{initial_small_rewards}")
    print(f"  Medium Rewards Collected: {medium_rewards_collected}/{initial_medium_rewards}")

    # Track rewards
    pacman_rewards_per_episode.append(pacman_cumulative_reward)
    ghost_rewards_per_episode.append(ghost_cumulative_reward)

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % 100 == 0:
        print(f"Episode {episode + 1} completed.")
# Calculate averages
average_small_rewards_collected = np.mean(total_small_rewards_collected)
average_medium_rewards_collected = np.mean(total_medium_rewards_collected)

# After training loop (end of the script)
# Calculate the average cumulative rewards for Pac-Man and Ghosts
average_pacman_reward = np.mean(pacman_rewards_per_episode)
average_ghost_reward = np.mean(ghost_rewards_per_episode)

# Print the averages to the terminal
print("\n--- Training Summary ---")
print(f"Total Episodes: {EPISODES_COUNT}")
print(f"Average Cumulative Reward for Pac-Man: {average_pacman_reward:.2f}")
print(f"Average Cumulative Reward for Ghosts: {average_ghost_reward:.2f}")
print(f"Average Small Rewards Collected: {average_small_rewards_collected:.2f}/{NUM_SMALL_REWARDS}")
print(f"Average Medium Rewards Collected: {average_medium_rewards_collected:.2f}/{NUM_MEDIUM_REWARDS}")
print(f"Pac-Man Wins: {pacman_wins}")
print(f"  - Wins by reaching big reward: {pacman_win_conditions['big_reward']}")
print(f"  - Wins by collecting all medium rewards: {pacman_win_conditions['all_medium_rewards']}")
print(f"  - Wins by collecting all small rewards: {pacman_win_conditions['all_small_rewards']}")
print(f"Ghost Wins: {ghost_wins}")


# Display training results
plot_training_performance(pacman_rewards_per_episode, ghost_rewards_per_episode)
plot_win_statistics(pacman_wins, ghost_wins)
plot_pacman_win_conditions(pacman_win_conditions)

# plot Headmaps
plot_annotated_heatmap(pacman_visit_counts, "Pac-Man State Visit Heatmap", original_big_reward, original_medium_rewards, original_small_rewards)
plot_annotated_heatmap(ghost_visit_counts, "Ghosts State Visit Heatmap", original_big_reward, original_medium_rewards, original_small_rewards, cmap="Reds",)