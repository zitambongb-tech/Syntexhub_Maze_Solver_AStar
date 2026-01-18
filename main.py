#!/usr/bin/env python3
"""
A* Maze Solver - SyntecXHub Internship Project
Author: [Your Name]
Date: 2024
"""

import heapq
import math
import time
from typing import List, Tuple, Optional, Set
import sys

class Node:
    """Represents a node in the maze grid for A* search"""
    def __init__(self, x: int, y: int, g: float = 0, h: float = 0, parent: Optional['Node'] = None):
        self.x = x
        self.y = y
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic estimate to goal
        self.f = g + h  # Total cost (f = g + h)
        self.parent = parent
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __repr__(self) -> str:
        return f"Node({self.x}, {self.y}, g={self.g:.1f}, h={self.h:.1f}, f={self.f:.1f})"

class MazeSolver:
    """A* Search Algorithm Implementation for Maze Solving"""
    
    # Movement directions: up, down, left, right
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, maze: List[str]):
        """
        Initialize the maze solver
        
        Args:
            maze: List of strings representing the maze grid
                  'S' = Start position
                  'G' = Goal position
                  '#' = Wall/obstacle
                  '.' = Open path
                  ' ' = Also open path
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows > 0 else 0
        self.start_pos = None
        self.goal_pos = None
        
        # Validate and find start/goal
        self._validate_maze()
        self._find_start_goal()
    
    def _validate_maze(self) -> None:
        """Check if maze is valid"""
        if self.rows == 0:
            raise ValueError("Maze cannot be empty")
        
        first_row_len = len(self.maze[0])
        for i, row in enumerate(self.maze):
            if len(row) != first_row_len:
                raise ValueError(f"Row {i} has different length: expected {first_row_len}, got {len(row)}")
    
    def _find_start_goal(self) -> None:
        """Locate start (S) and goal (G) positions in the maze"""
        start_found = False
        goal_found = False
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.maze[i][j]
                if cell == 'S':
                    if start_found:
                        raise ValueError("Multiple start positions found")
                    self.start_pos = (i, j)
                    start_found = True
                elif cell == 'G':
                    if goal_found:
                        raise ValueError("Multiple goal positions found")
                    self.goal_pos = (i, j)
                    goal_found = True
        
        if not start_found:
            raise ValueError("Start position 'S' not found in maze")
        if not goal_found:
            raise ValueError("Goal position 'G' not found in maze")
    
    @staticmethod
    def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int], 
                 heuristic_type: str = "manhattan") -> float:
        """
        Calculate heuristic distance between two points
        
        Args:
            pos1: (x1, y1) coordinates
            pos2: (x2, y2) coordinates
            heuristic_type: Type of heuristic - "manhattan" or "euclidean"
        
        Returns:
            Heuristic distance value
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        if heuristic_type.lower() == "manhattan":
            # Manhattan distance: |x1-x2| + |y1-y2|
            return abs(x1 - x2) + abs(y1 - y2)
        elif heuristic_type.lower() == "euclidean":
            # Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:
            raise ValueError(f"Unknown heuristic type: {heuristic_type}. Use 'manhattan' or 'euclidean'")
    
    def _is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell is within bounds and not a wall"""
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.maze[x][y] != '#')
    
    def _get_neighbors(self, node: Node) -> List[Tuple[int, int]]:
        """Get valid neighboring cells from current position"""
        neighbors = []
        
        for dx, dy in self.DIRECTIONS:
            nx, ny = node.x + dx, node.y + dy
            
            if self._is_valid_cell(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def solve(self, heuristic_type: str = "manhattan", 
              verbose: bool = True) -> Optional[List[Tuple[int, int]]]:
        """
        Solve the maze using A* search algorithm
        
        Args:
            heuristic_type: "manhattan" or "euclidean"
            verbose: Whether to print progress information
        
        Returns:
            List of (x, y) coordinates representing the path from start to goal,
            or None if no path exists
        """
        if verbose:
            print("\n" + "="*60)
            print(f"A* MAZE SOLVER - {heuristic_type.upper()} HEURISTIC")
            print("="*60)
            print(f"Maze size: {self.rows} x {self.cols}")
            print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
            print("-"*60)
        
        # Initialize data structures
        open_set = []  # Priority queue (min-heap)
        closed_set = set()  # Set of visited nodes
        
        # Create start node
        start_node = Node(self.start_pos[0], self.start_pos[1])
        start_node.h = self.heuristic(self.start_pos, self.goal_pos, heuristic_type)
        start_node.f = start_node.g + start_node.h
        
        # Push start node to open set
        heapq.heappush(open_set, (start_node.f, start_node))
        
        # Dictionary to track nodes for quick access and update
        node_dict = {start_node: start_node}
        
        start_time = time.time()
        nodes_explored = 0
        
        # Main A* loop
        while open_set:
            # Get node with lowest f score
            current_f, current_node = heapq.heappop(open_set)
            nodes_explored += 1
            
            # Check if we reached the goal
            if (current_node.x, current_node.y) == self.goal_pos:
                elapsed_time = time.time() - start_time
                
                if verbose:
                    print(f"\n✅ GOAL REACHED!")
                    print(f"   Nodes explored: {nodes_explored}")
                    print(f"   Time taken: {elapsed_time:.6f} seconds")
                    print(f"   Path cost: {current_node.g} steps")
                
                # Reconstruct and return path
                return self._reconstruct_path(current_node)
            
            # Add current node to closed set
            closed_set.add((current_node.x, current_node.y))
            
            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current_node):
                nx, ny = neighbor_pos
                
                # Skip if already visited
                if (nx, ny) in closed_set:
                    continue
                
                # Calculate new g score (cost from start to neighbor)
                new_g = current_node.g + 1  # Each move costs 1
                
                # Create neighbor node
                neighbor = Node(nx, ny, new_g, 
                               self.heuristic((nx, ny), self.goal_pos, heuristic_type),
                               current_node)
                neighbor.f = neighbor.g + neighbor.h
                
                # Check if this node is already in open set
                existing_node = node_dict.get(neighbor)
                
                if existing_node is None:
                    # New node - add to open set
                    heapq.heappush(open_set, (neighbor.f, neighbor))
                    node_dict[neighbor] = neighbor
                elif new_g < existing_node.g:
                    # Found better path to existing node - update it
                    existing_node.g = new_g
                    existing_node.f = existing_node.g + existing_node.h
                    existing_node.parent = current_node
                    
                    # Re-heapify since we changed a node's priority
                    heapq.heapify(open_set)
        
        # No path found
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n❌ NO PATH FOUND")
            print(f"   Nodes explored: {nodes_explored}")
            print(f"   Time taken: {elapsed_time:.6f} seconds")
            print(f"   Start and goal are not connected")
        
        return None
    
    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal by backtracking parents"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get start→goal
    
    def print_solution(self, path: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Print the maze with solution path visualization
        
        Args:
            path: Solution path to visualize
        """
        print("\n" + "="*60)
        print("MAZE VISUALIZATION")
        print("="*60)
        
        # Create display grid
        display_grid = [list(row) for row in self.maze]
        
        if path:
            # Mark path on display grid (excluding start and goal)
            for x, y in path:
                if display_grid[x][y] not in ['S', 'G']:
                    display_grid[x][y] = '*'
        
        # Print column headers
        col_headers = "    " + " ".join(str(i % 10) for i in range(self.cols))
        print(col_headers)
        print("   " + "-" * (2 * self.cols + 1))
        
        # Print maze rows with row numbers
        for i, row in enumerate(display_grid):
            row_str = f"{i:2} | " + " ".join(row)
            print(row_str)
        
        # Print legend
        print("\nLEGEND:")
        print("  S = Start position")
        print("  G = Goal position")
        print("  # = Wall (blocked)")
        print("  . = Open path")
        print("  * = Solution path" if path else "")
    
    def print_path_details(self, path: List[Tuple[int, int]]) -> None:
        """Print detailed information about the found path"""
        if not path:
            print("No path to display")
            return
        
        print("\n" + "="*60)
        print("PATH DETAILS")
        print("="*60)
        print(f"Total steps: {len(path) - 1}")
        print(f"Path length: {len(path)} nodes")
        print("\nPath coordinates (row, column):")
        
        for i, (x, y) in enumerate(path):
            symbol = "S" if i == 0 else "G" if i == len(path) - 1 else f"{i}"
            print(f"  {symbol}: ({x}, {y})")


def run_example_mazes():
    """Run the solver on example mazes"""
    
    # Example 1: Simple solvable maze
    maze1 = [
        "S..#.",
        ".##..",
        ".#...",
        "...#G"
    ]
    
    # Example 2: Straight path
    maze2 = [
        "S....",
        ".....",
        ".....",
        "....G"
    ]
    
    # Example 3: Maze with obstacles
    maze3 = [
        "S###..",
        "...#..",
        ".###..",
        "......",
        "....#G"
    ]
    
    # Example 4: No path (blocked)
    maze4 = [
        "S.#",
        "###",
        "..#",
        ".#G"
    ]
    
    # Example 5: Complex maze
    maze5 = [
        "S.#.....",
        ".#.#.#.#",
        ".#.#.#.#",
        ".#.#.#.#",
        "........",
        "####.##.",
        "......#G"
    ]
    
    # Select which maze to run
    selected_maze = maze1
    maze_name = "Example 1: Simple Maze"
    
    print("\n" + "="*60)
    print(f"SYNTECXHUB INTERNSHIP - MAZE SOLVER USING A* SEARCH")
    print("="*60)
    print(f"\n{maze_name}")
    
    try:
        # Create solver instance
        solver = MazeSolver(selected_maze)
        
        # Solve using Manhattan heuristic
        print("\n" + "="*60)
        print("TEST 1: MANHATTAN HEURISTIC")
        print("="*60)
        path_manhattan = solver.solve(heuristic_type="manhattan", verbose=True)
        solver.print_solution(path_manhattan)
        
        if path_manhattan:
            solver.print_path_details(path_manhattan)
        
        # Solve using Euclidean heuristic
        print("\n" + "="*60)
        print("TEST 2: EUCLIDEAN HEURISTIC")
        print("="*60)
        
        # Create new solver instance to reset
        solver2 = MazeSolver(selected_maze)
        path_euclidean = solver2.solve(heuristic_type="euclidean", verbose=True)
        solver2.print_solution(path_euclidean)
        
        # Compare results if both found paths
        if path_manhattan and path_euclidean:
            print("\n" + "="*60)
            print("COMPARISON OF HEURISTICS")
            print("="*60)
            print(f"Manhattan path length: {len(path_manhattan) - 1} steps")
            print(f"Euclidean path length: {len(path_euclidean) - 1} steps")
            
            if len(path_manhattan) == len(path_euclidean):
                print("✓ Both heuristics found paths of equal length")
            else:
                print("⚠ Path lengths differ - both should find optimal paths")
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Run interactive mode where user can input custom mazes"""
    print("\n" + "="*60)
    print("INTERACTIVE MAZE SOLVER")
    print("="*60)
    
    print("\nEnter your maze row by row.")
    print("Use: S=Start, G=Goal, #=Wall, .=Open path")
    print("Press Enter twice to finish input.")
    
    maze = []
    row_num = 0
    
    while True:
        row = input(f"Row {row_num} (or press Enter to finish): ").strip()
        
        if not row:
            if row_num == 0:
                print("Using default maze...")
                return run_example_mazes()
            break
        
        maze.append(row)
        row_num += 1
    
    if not maze:
        print("No maze provided. Using example maze.")
        return run_example_mazes()
    
    try:
        solver = MazeSolver(maze)
        
        print("\nSelect heuristic:")
        print("1. Manhattan (recommended for grid movement)")
        print("2. Euclidean")
        choice = input("Enter 1 or 2 (default 1): ").strip()
        
        heuristic_type = "euclidean" if choice == "2" else "manhattan"
        
        path = solver.solve(heuristic_type=heuristic_type, verbose=True)
        solver.print_solution(path)
        
        if path:
            solver.print_path_details(path)
        else:
            print("\nTry modifying your maze to create a path from S to G!")
            
    except ValueError as e:
        print(f"\nError in maze: {e}")
        print("Please ensure:")
        print("1. Exactly one 'S' and one 'G'")
        print("2. All rows have same length")
        print("3. Use only S, G, #, . characters")


def main():
    """Main function with menu"""
    print("\n" + "="*60)
    print("A* MAZE SOLVER - SYNTECXHUB INTERNSHIP")
    print("="*60)
    
    while True:
        print("\nMENU:")
        print("1. Run example mazes")
        print("2. Interactive mode (custom maze)")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            run_example_mazes()
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            print("\nThank you for using A* Maze Solver!")
            print("Good luck with your SyntecXHub internship! 🚀")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()