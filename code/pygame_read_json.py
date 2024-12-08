import pygame
import json
import os
import time

# Define Pygame parameters
CELL_SIZE = 50
GRID_SIZE = 8  # Set the grid size here
BACKGROUND_COLOR = (0, 0, 0)  # Black
SPEED = 0.1  # Adjust this value to control the speed (e.g., 0.2 seconds between frames)

# File paths for the sprites
SPRITE_PATHS = {
    "pacman": "sprites/pacman.png",
    "ghost": "sprites/ghost.png",
    "wall": "sprites/wall.png",
    "small_reward": "sprites/small_reward.png",
    "medium_reward": "sprites/medium_reward.png",
    "big_reward": "sprites/big_reward.png",
}

# Load game states from the specified directory
def load_game_states(directory):
    game_states = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                game_states.append(json.load(file))
    return game_states

# Initialize Pygame and set up the display
def initialize_pygame(grid_size):
    pygame.init()
    screen = pygame.display.set_mode((grid_size * CELL_SIZE, grid_size * CELL_SIZE))
    pygame.display.set_caption("Pac-Man Game Visualization")
    return screen

# Load sprites
def load_sprites(sprite_paths):
    sprites = {}
    for key, path in sprite_paths.items():
        sprites[key] = pygame.image.load(path).convert_alpha()
        sprites[key] = pygame.transform.scale(sprites[key], (CELL_SIZE, CELL_SIZE))
    return sprites

# Render a single game state
def render_state(screen, sprites, state, grid_size):
    screen.fill(BACKGROUND_COLOR)

    # Draw the grid (optional, can be removed for a cleaner look)
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)  # Grid lines

    # Draw walls
    for wall in state["walls"]:
        rect = pygame.Rect(wall[1] * CELL_SIZE, wall[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        screen.blit(sprites["wall"], rect)

    # Draw big reward
    big_reward = state["big_reward"]
    rect = pygame.Rect(big_reward[1] * CELL_SIZE, big_reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(sprites["big_reward"], rect)

    # Draw medium rewards
    for reward in state["medium_rewards"]:
        rect = pygame.Rect(reward[1] * CELL_SIZE, reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        screen.blit(sprites["medium_reward"], rect)

    # Draw small rewards
    for reward in state["small_rewards"]:
        rect = pygame.Rect(reward[1] * CELL_SIZE, reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        screen.blit(sprites["small_reward"], rect)

    # Draw ghosts
    for ghost in state["ghost_positions"]:
        rect = pygame.Rect(ghost[1] * CELL_SIZE, ghost[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        screen.blit(sprites["ghost"], rect)

    # Draw Pac-Man
    pacman = state["pacman_position"]
    rect = pygame.Rect(pacman[1] * CELL_SIZE, pacman[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(sprites["pacman"], rect)

    # Update the display
    pygame.display.flip()

# Display the winner with episode number
def display_winner(screen, winner, episode_num):
    screen.fill((0, 0, 0))  # Clear the screen

    # Set colors and text based on the winner
    if winner == "pacman":
        winner_message = "Pac-Man Wins!"
        color = (0, 255, 0)  # Green
    elif winner == "ghost":
        winner_message = "Ghost Wins!"
        color = (255, 0, 0)  # Red
    else:
        winner_message = "Game Over!"
        color = (255, 255, 255)  # White

    # Create fonts for episode and winner messages
    font_episode = pygame.font.Font(None, 40)  # Font for episode
    font_winner = pygame.font.Font(None, 50)  # Font for winner message

    # Render the episode text
    episode_text = font_episode.render(f"Episode {episode_num}", True, (255, 255, 255))  # White color for episode
    episode_text_rect = episode_text.get_rect(center=(GRID_SIZE * CELL_SIZE // 2, GRID_SIZE * CELL_SIZE // 2 - 30))
    screen.blit(episode_text, episode_text_rect)

    # Render the winner text
    winner_text = font_winner.render(winner_message, True, color)
    winner_text_rect = winner_text.get_rect(center=(GRID_SIZE * CELL_SIZE // 2, GRID_SIZE * CELL_SIZE // 2 + 20))
    screen.blit(winner_text, winner_text_rect)

    # Update the display and pause
    pygame.display.flip()
    time.sleep(1.5)  # Pause for 1.5 seconds

# Main function to visualize the game states
def visualize_game_states(game_states, grid_size, sprite_paths):
    screen = initialize_pygame(grid_size)
    sprites = load_sprites(sprite_paths)
    clock = pygame.time.Clock()

    total_episodes = len(game_states)
    for episode_num, episode_states in enumerate(game_states, start=1):
        winner = None  # Track the winner

        print(f"\nEpisode {episode_num}:")
        for i, state in enumerate(episode_states):
            # Extract positions for debugging
            pacman_pos = state["pacman_position"]
            ghost_positions = state["ghost_positions"]
            big_reward_pos = state["big_reward"]
            small_reward_positions = state["small_rewards"]
            medium_reward_positions = state["medium_rewards"]

            # Check if it's the last step of the last episode
            is_last_episode = episode_num == total_episodes
            is_last_step = i == len(episode_states) - 1

            # Print state information for each step
            print(f"  Step {i + 1}: (Episode {episode_num})")
            print(f"    Pac-Man Position: {pacman_pos}")
            print(f"    Ghost Positions: {ghost_positions}")
            print(f"    Big Reward Position: {big_reward_pos}")
            print(f"    Small Reward Positions: {small_reward_positions}")
            print(f"    Medium Reward Positions: {medium_reward_positions}")

            # Highlight the last step of the last episode
            if is_last_episode and is_last_step:
                print("    Last Episode and Last Step!")

            render_state(screen, sprites, state, grid_size)
            time.sleep(SPEED)  # Use the SPEED variable to control delay
            clock.tick(60)

            # Handle Pygame events (e.g., quitting)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Determine the winner after the last state of the episode
            if is_last_step:
                if pacman_pos == big_reward_pos:
                    winner = "pacman"
                elif len(medium_reward_positions) == 0:  # All medium rewards collected
                    winner = "pacman"
                elif len(small_reward_positions) == 0:  # All small rewards collected
                    winner = "pacman"
                elif pacman_pos in ghost_positions:
                    winner = "ghost"

        # Display the winner with the episode number at the end of the episode
        display_winner(screen, winner, episode_num)

    pygame.quit()

# Load and visualize the game states
if __name__ == "__main__":
    directory = "game_states"  # Directory where JSON files are stored
    game_states = load_game_states(directory)
    if game_states:
        visualize_game_states(game_states, GRID_SIZE, SPRITE_PATHS)
    else:
        print("No game state files found in the directory.")
