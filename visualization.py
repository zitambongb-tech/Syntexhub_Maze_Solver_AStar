"""
🎨 A* Maze Solver - Advanced Visualization Module
SyntecXHub Internship Project - Professional Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from matplotlib import colors, cm
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import math

class MazeVisualizer:
    """
    Advanced maze visualization with realistic textures and vibrant colors
    """
    
    def __init__(self, maze: List[str], theme: str = "vibrant"):
        """
        Initialize visualizer with custom theme
        
        Args:
            maze: List of strings representing the maze
            theme: Color theme - "vibrant", "dark", "pastel", "cyber"
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows > 0 else 0
        
        # Define color themes
        self.themes = {
            "vibrant": {
                'background': '#1a1a2e',
                'grid': '#16213e',
                'wall': '#0f3460',  # Deep blue
                'wall_texture': '#1e3a5f',
                'path': '#00ff88',  # Neon green
                'path_glow': '#00ffaa',
                'start': '#ff2e63',  # Pink-red
                'start_glow': '#ff5e8a',
                'goal': '#00d4ff',  # Cyan
                'goal_glow': '#80eaff',
                'explored': '#ff9a00',  # Orange
                'frontier': '#9d4edd',  # Purple
                'current': '#ffbd00',  # Gold
                'text': '#ffffff',
                'text_secondary': '#b0b0b0'
            },
            "dark": {
                'background': '#121212',
                'grid': '#1e1e1e',
                'wall': '#2d2d2d',
                'wall_texture': '#3a3a3a',
                'path': '#4ecdc4',
                'path_glow': '#6ffff6',
                'start': '#ff6b6b',
                'start_glow': '#ff8e8e',
                'goal': '#45b7d1',
                'goal_glow': '#67d9f0',
                'explored': '#feca57',
                'frontier': '#b967ff',
                'current': '#ff9f43',
                'text': '#ffffff',
                'text_secondary': '#aaaaaa'
            },
            "pastel": {
                'background': '#f8f9fa',
                'grid': '#e9ecef',
                'wall': '#adb5bd',
                'wall_texture': '#ced4da',
                'path': '#38b000',
                'path_glow': '#70e000',
                'start': '#ff595e',
                'start_glow': '#ff8a8e',
                'goal': '#1982c4',
                'goal_glow': '#4da8ff',
                'explored': '#ffca3a',
                'frontier': '#6a4c93',
                'current': '#ff9100',
                'text': '#212529',
                'text_secondary': '#495057'
            },
            "cyber": {
                'background': '#0a0a0a',
                'grid': '#1a1a1a',
                'wall': '#003366',
                'wall_texture': '#004488',
                'path': '#00ffcc',  # Cyber teal
                'path_glow': '#66ffdd',
                'start': '#ff0066',  # Neon pink
                'start_glow': '#ff3399',
                'goal': '#0066ff',  # Neon blue
                'goal_glow': '#3399ff',
                'explored': '#ffcc00',
                'frontier': '#cc00ff',
                'current': '#ff6600',
                'text': '#00ffcc',
                'text_secondary': '#66ffdd'
            }
        }
        
        # Set theme
        self.theme = self.themes.get(theme, self.themes["vibrant"])
        
        # Create custom gradient for walls
        self.wall_gradient = LinearSegmentedColormap.from_list(
            'wall_gradient', 
            [self.theme['wall'], self.theme['wall_texture']]
        )
    
    def _create_wall_texture(self, ax, x: float, y: float, width: float = 1.0, height: float = 1.0):
        """Create realistic wall texture with shading"""
        # Main wall rectangle with gradient
        wall = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=self.theme['wall'],
            edgecolor=self.theme['wall_texture'],
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(wall)
        
        # Add brick pattern texture
        brick_width = width / 3
        brick_height = height / 3
        
        for i in range(3):
            for j in range(3):
                brick_x = x + i * brick_width + 0.05
                brick_y = y + j * brick_height + 0.05
                
                # Create subtle brick pattern
                brick = patches.Rectangle(
                    (brick_x, brick_y),
                    brick_width - 0.1,
                    brick_height - 0.1,
                    facecolor=self.theme['wall_texture'],
                    alpha=0.2,
                    linewidth=0
                )
                ax.add_patch(brick)
        
        # Add shadow effect
        shadow = patches.Rectangle(
            (x + 0.05, y - 0.05),
            width - 0.1,
            height - 0.1,
            facecolor='black',
            alpha=0.1,
            linewidth=0,
            zorder=1
        )
        ax.add_patch(shadow)
    
    def _create_start_marker(self, ax, x: float, y: float):
        """Create animated start position marker"""
        # Pulsing circle
        for i in range(3):
            circle = plt.Circle(
                (x + 0.5, y + 0.5),
                0.2 + i * 0.1,
                facecolor=self.theme['start_glow'],
                alpha=0.3 - i * 0.1,
                edgecolor='none',
                zorder=10 + i
            )
            ax.add_patch(circle)
        
        # Main start marker
        start = FancyBboxPatch(
            (x + 0.1, y + 0.1), 0.8, 0.8,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor=self.theme['start'],
            edgecolor='white',
            linewidth=2,
            zorder=15
        )
        ax.add_patch(start)
        
        # Start text
        ax.text(x + 0.5, y + 0.5, 'S', 
               ha='center', va='center',
               fontsize=20, fontweight='bold',
               color='white', zorder=20)
    
    def _create_goal_marker(self, ax, x: float, y: float):
        """Create animated goal position marker"""
        # Glowing effect
        for i in range(3):
            circle = plt.Circle(
                (x + 0.5, y + 0.5),
                0.25 + i * 0.15,
                facecolor=self.theme['goal_glow'],
                alpha=0.2 - i * 0.06,
                edgecolor='none',
                zorder=10 + i
            )
            ax.add_patch(circle)
        
        # Main goal marker (diamond shape instead of star)
        diamond = patches.RegularPolygon(
            (x + 0.5, y + 0.5),
            4,  # Diamond (square rotated 45 degrees)
            0.35,
            facecolor=self.theme['goal'],
            edgecolor='white',
            linewidth=2,
            zorder=15
        )
        ax.add_patch(diamond)
        
        # Goal text
        ax.text(x + 0.5, y + 0.5, 'G', 
               ha='center', va='center',
               fontsize=20, fontweight='bold',
               color='white', zorder=20)
    
    def _create_path_segment(self, ax, start: Tuple[float, float], end: Tuple[float, float]):
        """Create glowing path segment with gradient"""
        # Main path line with glow
        line = ConnectionPatch(
            start, end,
            coordsA="data", coordsB="data",
            arrowstyle="-",
            color=self.theme['path'],
            linewidth=4,
            alpha=0.9,
            zorder=8
        )
        ax.add_patch(line)
        
        # Glow effect around the line
        for i in range(2):
            glow = ConnectionPatch(
                start, end,
                coordsA="data", coordsB="data",
                arrowstyle="-",
                color=self.theme['path_glow'],
                linewidth=8 + i * 2,
                alpha=0.2 - i * 0.1,
                zorder=7 - i
            )
            ax.add_patch(glow)
        
        # Start point of segment
        circle_start = plt.Circle(
            start, 0.15,
            facecolor=self.theme['path'],
            edgecolor='white',
            linewidth=2,
            zorder=9
        )
        ax.add_patch(circle_start)
        
        # End point of segment
        circle_end = plt.Circle(
            end, 0.15,
            facecolor=self.theme['path'],
            edgecolor='white',
            linewidth=2,
            zorder=9
        )
        ax.add_patch(circle_end)
    
    def create_3d_effect_maze(self, path: Optional[List[Tuple[int, int]]] = None,
                             explored: Optional[List[Tuple[int, int]]] = None,
                             frontier: Optional[List[Tuple[int, int]]] = None,
                             title: str = "A* Maze Solver - 3D Visualization"):
        """
        Create 3D-like visualization with depth and shadows
        
        Args:
            path: Solution path coordinates
            explored: Explored nodes
            frontier: Frontier nodes
            title: Plot title
        
        Returns:
            matplotlib Figure and Axes
        """
        fig, ax = plt.subplots(figsize=(max(self.cols * 1.2, 10), max(self.rows * 1.2, 8)))
        
        # Set background
        fig.patch.set_facecolor(self.theme['background'])
        ax.set_facecolor(self.theme['background'])
        
        # Draw grid with 3D effect
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.maze[i][j]
                x, y = j, self.rows - i - 1
                
                # Draw cell base with shadow
                base = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor=self.theme['grid'],
                    edgecolor='none',
                    alpha=0.7,
                    zorder=1
                )
                ax.add_patch(base)
                
                # Shadow effect (right-bottom)
                shadow = patches.Rectangle(
                    (x + 0.05, y - 0.05), 0.95, 0.95,
                    facecolor='black',
                    alpha=0.1,
                    zorder=2
                )
                ax.add_patch(shadow)
                
                # Draw cell content based on type
                if cell == '#':  # Wall
                    self._create_wall_texture(ax, x, y)
                elif cell == 'S':  # Start
                    self._create_start_marker(ax, x, y)
                elif cell == 'G':  # Goal
                    self._create_goal_marker(ax, x, y)
                else:  # Open cell
                    # Add subtle grid pattern
                    grid_cell = patches.Rectangle(
                        (x + 0.1, y + 0.1), 0.8, 0.8,
                        facecolor='white',
                        alpha=0.05,
                        edgecolor='none',
                        zorder=3
                    )
                    ax.add_patch(grid_cell)
                
                # Add coordinate labels
                ax.text(x + 0.1, y + 0.85, f'{i},{j}',
                       fontsize=7, color=self.theme['text_secondary'],
                       alpha=0.6, zorder=4)
        
        # Draw explored nodes
        if explored:
            for r, c in explored:
                if self.maze[r][c] not in ['S', 'G']:
                    y_pos, x_pos = self.rows - r - 1, c
                    explored_circle = plt.Circle(
                        (x_pos + 0.5, y_pos + 0.5), 0.25,
                        facecolor=self.theme['explored'],
                        edgecolor='white',
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=5
                    )
                    ax.add_patch(explored_circle)
                    
                    # Add exploration count
                    idx = explored.index((r, c))
                    ax.text(x_pos + 0.5, y_pos + 0.5, str(idx + 1),
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white', zorder=6)
        
        # Draw frontier nodes
        if frontier:
            for r, c in frontier:
                if self.maze[r][c] not in ['S', 'G']:
                    y_pos, x_pos = self.rows - r - 1, c
                    frontier_circle = plt.Circle(
                        (x_pos + 0.5, y_pos + 0.5), 0.3,
                        facecolor=self.theme['frontier'],
                        edgecolor='white',
                        linewidth=2,
                        alpha=0.9,
                        zorder=7
                    )
                    ax.add_patch(frontier_circle)
                    
                    ax.text(x_pos + 0.5, y_pos + 0.5, 'F',
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='white', zorder=8)
        
        # Draw path
        if path and len(path) > 1:
            for idx in range(len(path) - 1):
                r1, c1 = path[idx]
                r2, c2 = path[idx + 1]
                
                y1, x1 = self.rows - r1 - 1, c1
                y2, x2 = self.rows - r2 - 1, c2
                
                start_point = (x1 + 0.5, y1 + 0.5)
                end_point = (x2 + 0.5, y2 + 0.5)
                
                self._create_path_segment(ax, start_point, end_point)
        
        # Set plot limits and style
        ax.set_xlim(-0.5, self.cols + 0.5)
        ax.set_ylim(-0.5, self.rows + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with styling
        title_text = ax.text(
            self.cols / 2, self.rows + 0.8,
            title,
            ha='center', va='center',
            fontsize=24, fontweight='bold',
            color=self.theme['text'],
            fontfamily='monospace'
        )
        title_text.set_path_effects([
            plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black', alpha=0.5)
        ])
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor=self.theme['start'], label='Start (S)', alpha=0.9),
            patches.Patch(facecolor=self.theme['goal'], label='Goal (G)', alpha=0.9),
            patches.Patch(facecolor=self.theme['wall'], label='Wall (#)', alpha=0.9),
            patches.Patch(facecolor=self.theme['explored'], label='Explored', alpha=0.7),
            patches.Patch(facecolor=self.theme['frontier'], label='Frontier (F)', alpha=0.7),
            patches.Patch(facecolor=self.theme['path'], label='Solution Path', alpha=0.9),
        ]
        
        legend = ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            framealpha=0.9,
            facecolor=self.theme['background'],
            edgecolor=self.theme['text'],
            fontsize=10,
            title='Legend',
            title_fontsize=12
        )
        legend.get_title().set_color(self.theme['text'])
        
        # Add stats box
        stats_text = f"""
        Maze Stats:
        • Size: {self.rows} × {self.cols}
        • Total Cells: {self.rows * self.cols}
        • Walls: {sum(row.count('#') for row in self.maze)}
        """
        
        if path:
            stats_text += f"\n• Path Length: {len(path) - 1} steps"
            stats_text += f"\n• Path Nodes: {len(path)}"
        
        if explored:
            stats_text += f"\n• Nodes Explored: {len(explored)}"
        
        stats_box = ax.text(
            1.02, 0.3, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            color=self.theme['text_secondary'],
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=self.theme['grid'],
                edgecolor=self.theme['text_secondary'],
                alpha=0.8
            )
        )
        
        plt.tight_layout()
        return fig, ax
    
    def create_minimal_visualization(self, path: Optional[List[Tuple[int, int]]] = None,
                                    title: str = "A* Maze Solution"):
        """
        Create clean, minimal visualization
        
        Args:
            path: Solution path
            title: Plot title
        
        Returns:
            matplotlib Figure and Axes
        """
        fig, ax = plt.subplots(figsize=(max(self.cols, 8), max(self.rows, 6)))
        
        # Clean background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create color grid
        grid = np.zeros((self.rows, self.cols, 3))
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.maze[i][j]
                if cell == '#':
                    grid[i, j] = [0.2, 0.2, 0.2]  # Dark gray
                elif cell == 'S':
                    grid[i, j] = [0.2, 0.8, 0.2]  # Green
                elif cell == 'G':
                    grid[i, j] = [0.8, 0.2, 0.2]  # Red
                else:
                    grid[i, j] = [0.95, 0.95, 0.95]  # Light gray
        
        # Display grid
        ax.imshow(grid, interpolation='none', aspect='auto')
        
        # Add path
        if path:
            rows, cols = zip(*path)
            ax.plot(cols, rows, 'b-', linewidth=3, alpha=0.8, marker='o',
                   markersize=8, markerfacecolor='blue', markeredgecolor='white')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Add coordinate labels
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j, i, f'{i},{j}', ha='center', va='center',
                       fontsize=8, color='gray', alpha=0.7)
        
        plt.tight_layout()
        return fig, ax
    
    def save_visualization(self, filename: str = 'maze_3d_visualization.png',
                          style: str = '3d', **kwargs):
        """
        Save visualization to high-quality image
        
        Args:
            filename: Output filename
            style: Visualization style ('3d' or 'minimal')
            **kwargs: Arguments for visualization function
        """
        if style == '3d':
            fig, ax = self.create_3d_effect_maze(**kwargs)
        else:
            fig, ax = self.create_minimal_visualization(**kwargs)
        
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close(fig)
        print(f"✅ Visualization saved as '{filename}'")
    
    def show_visualization(self, style: str = '3d', **kwargs):
        """
        Display visualization
        
        Args:
            style: Visualization style
            **kwargs: Arguments for visualization function
        """
        if style == '3d':
            fig, ax = self.create_3d_effect_maze(**kwargs)
        else:
            fig, ax = self.create_minimal_visualization(**kwargs)
        
        plt.show()
        plt.close(fig)


def demo_all_visualizations():
    """
    Demonstrate all visualization styles
    """
    print("="*70)
    print("🎨 A* MAZE SOLVER - ADVANCED VISUALIZATION DEMO")
    print("="*70)
    
    # Example maze
    maze = [
        "S..#.",
        ".##..",
        ".#...",
        "...#G"
    ]
    
    # Example data
    path = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]
    explored = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]
    frontier = [(0, 2), (1, 0), (2, 0), (2, 2)]
    
    # Test different themes
    themes = ["vibrant", "dark", "pastel", "cyber"]
    
    for theme in themes:
        print(f"\n🎨 Testing '{theme}' theme...")
        
        # Create visualizer with theme
        visualizer = MazeVisualizer(maze, theme=theme)
        
        # Create and save 3D visualization
        filename = f"maze_{theme}_theme.png"
        try:
            visualizer.save_visualization(
                filename=filename,
                style='3d',
                path=path,
                explored=explored,
                frontier=frontier,
                title=f"A* Maze Solver - {theme.capitalize()} Theme"
            )
            print(f"   ✓ Created {filename}")
        except Exception as e:
            print(f"   ✗ Error creating {filename}: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE! Check the generated PNG files.")
    print("="*70)


def simple_demo():
    """Simple working demo without complex features"""
    print("="*70)
    print("🎨 A* MAZE SOLVER - SIMPLE VISUALIZATION DEMO")
    print("="*70)
    
    # Example maze
    maze = [
        "S..#.",
        ".##..",
        ".#...",
        "...#G"
    ]
    
    # Example path
    path = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]
    
    print("\nCreating simple visualization...")
    
    # Create simple visualizer
    visualizer = MazeVisualizer(maze, theme="vibrant")
    
    # Show minimal visualization (more reliable)
    visualizer.show_visualization(
        style='minimal',
        path=path,
        title="A* Maze Solver - Simple View"
    )
    
    # Save it
    visualizer.save_visualization(
        'simple_maze_solution.png',
        style='minimal',
        path=path,
        title="A* Maze Solution"
    )
    
    print("✅ Simple demo completed! Check 'simple_maze_solution.png'")


def quick_test():
    """Quick test function - simplified"""
    print("Running quick visualization test...")
    
    maze = [
        "S..#.",
        ".##..",
        ".#...",
        "...#G"
    ]
    
    path = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]
    
    visualizer = MazeVisualizer(maze, theme="vibrant")
    
    print("1. Creating minimal visualization (most reliable)...")
    visualizer.show_visualization(style='minimal', path=path)
    
    print("2. Saving to file...")
    visualizer.save_visualization('test_minimal.png', style='minimal', path=path)
    
    print("3. Trying 3D visualization...")
    try:
        visualizer.show_visualization(style='3d', path=path)
        visualizer.save_visualization('test_3d.png', style='3d', path=path)
        print("   ✓ 3D visualization created successfully!")
    except Exception as e:
        print(f"   ⚠ 3D visualization failed: {e}")
        print("   Minimal visualization still works perfectly!")
    
    print("✅ Quick test completed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 A* MAZE SOLVER - VISUALIZATION MODULE")
    print("="*70)
    print("\nChoose an option:")
    print("1. Quick test (recommended)")
    print("2. Simple demo")
    print("3. Full demo (all themes)")
    print("4. Exit")
    
    try:
        choice = input("\nSelect (1-4): ").strip()
        
        if choice == '1':
            quick_test()
        elif choice == '2':
            simple_demo()
        elif choice == '3':
            demo_all_visualizations()
        elif choice == '4':
            print("Goodbye! 👋")
        else:
            print("Invalid choice. Running quick test...")
            quick_test()
            
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying simple demo instead...")
        simple_demo()