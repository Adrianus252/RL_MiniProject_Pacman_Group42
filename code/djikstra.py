import heapq

# Dijkstra's algorithm to compute shortest path
def dijkstra(grid, start, target, walls):
    rows, cols = len(grid), len(grid[0])
    distances = {start: 0}
    priority_queue = [(0, start)]  # (distance, position)
    visited = set()
    previous = {}  # To reconstruct the path
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while priority_queue:
        current_distance, current_position = heapq.heappop(priority_queue)

        if current_position in visited:
            continue
        visited.add(current_position)

        if current_position == target:
            # Reconstruct path
            path = []
            while current_position in previous:
                path.append(current_position)
                current_position = previous[current_position]
            return path[::-1]  # Return path from start to target

        for direction in directions:
            neighbor = (current_position[0] + direction[0], current_position[1] + direction[1])
            
            if (
                0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols  # Inside grid
                and neighbor not in walls  # Not a wall
                and neighbor not in visited  # Not already visited
            ):
                new_distance = current_distance + 1  # All edges have weight 1
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_position
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    return []  # Return empty list if no path exists

# Update the ghost positions in the game state
def move_ghosts_towards_pacman(state, grid_size):
    walls = set(tuple(wall) for wall in state["walls"])
    pacman_position = tuple(state["pacman_position"])
    updated_ghost_positions = []

    for ghost_position in state["ghost_positions"]:
        ghost_position = tuple(ghost_position)
        path = dijkstra([[0]*grid_size for _ in range(grid_size)], ghost_position, pacman_position, walls)
        
        if path and len(path) > 1:
            # Move ghost one step closer (next position in the path)
            updated_ghost_positions.append(path[1])
        else:
            # No path or already at the target
            updated_ghost_positions.append(ghost_position)
    
    state["ghost_positions"] = updated_ghost_positions
