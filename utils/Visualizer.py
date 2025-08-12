import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from geopy.distance import geodesic
from tqdm import tqdm
import plotly.graph_objects as go
import json

class SimulationAnalyzer:
    def __init__(self, simulation_folder):
        """Initialize analyzer with simulation data"""
        self.simulation_dir = None
        self.simulation_id = None
        self.output_dir = None
        self.metadata = None
        self.particles = None
        self.robot = None
        self.performance = None
        self.constraints = None
        self.colors = None
        self.figsize = None
        self.particle_metrics = None
        self.localization_error = None
        self.convergence_data = None

        # Convert to absolute path and normalize
        simulation_folder = os.path.abspath(os.path.normpath(simulation_folder))

        # Check if path exists as given
        if os.path.exists(simulation_folder):
            self.simulation_dir = simulation_folder
        else:
            # Get potential base directories to search from
            search_roots = [
                os.getcwd(),  # Current working directory
                os.path.dirname(os.path.abspath(__file__)),  # Where this script lives
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Parent directory
            ]

            # Try different possible locations
            possible_locations = []
            for root in search_roots:
                possible_locations.extend([
                    os.path.join(root, simulation_folder),
                    os.path.join(root, "simulation_data", os.path.basename(simulation_folder)),
                    os.path.join(root, "data", os.path.basename(simulation_folder))
                ])

            # Remove duplicates while preserving order
            possible_locations = list(dict.fromkeys(possible_locations))

            # Try all possible locations
            for location in possible_locations:
                if os.path.exists(location):
                    self.simulation_dir = os.path.abspath(location)
                    break
            else:

                searched_paths = "\n  - ".join([simulation_folder] + possible_locations)
                raise FileNotFoundError(
                    f"Could not find simulation directory. Searched in:\n  - {searched_paths}"
                )

        # Verify the directory contains required files
        required_files = {'metadata.json', 'particles.csv', 'robot.csv'}
        existing_files = set(os.listdir(self.simulation_dir))
        missing_files = required_files - existing_files

        if missing_files:
            raise FileNotFoundError(
                f"Simulation directory is missing required files: {missing_files}\n"
                f"Directory contains: {existing_files}"
            )

        self.simulation_id = os.path.basename(self.simulation_dir)
        self.output_dir = self.simulation_dir  # set to different path if needed

        try:
            self.load_data()
            self.setup_visualization()
            self.calculate_metrics()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize analyzer: {str(e)}")

    def setup_visualization(self):
        """Initialize visualization settings with fallbacks"""
        style_preferences = [
            'seaborn-v0_8-darkgrid',  # First try exact match
            'seaborn-v0_8',  # Fallback to general seaborn
            'ggplot',  # Alternative good style
            'default'  # Guaranteed to work
        ]

        for style in style_preferences:
            if style in plt.style.available:
                plt.style.use(style)
                break

        self.colors = {
            'robot': '#FF2E63',
            'particles': '#21B6A8',
            'ground_truth': '#252A34',
            'constraints': '#FFD56F'
        }
        self.figsize = (12, 8)

    def load_data(self):
        """Load and preprocess all simulation data"""
        try:
            # Load metadata from JSON
            with open(os.path.join(self.simulation_dir, "metadata.json"), encoding='utf-8') as f:
                self.metadata = json.load(f)

            # Load CSV files with explicit encoding
            self.particles = pd.read_csv(os.path.join(self.simulation_dir, "particles.csv"), encoding='utf-8')
            self.robot = pd.read_csv(os.path.join(self.simulation_dir, "robot.csv"), encoding='utf-8')
            self.performance = pd.read_csv(os.path.join(self.simulation_dir, "performance.csv"), encoding='utf-8')

            # Process constraints with encoding fallback
            constraints_path = os.path.join(self.simulation_dir, "constraints.csv")
            if os.path.exists(constraints_path):
                try:
                    self.constraints = pd.read_csv(constraints_path, encoding='utf-8')
                except UnicodeDecodeError:
                    # Fallback to latin-1 if utf-8 fails
                    self.constraints = pd.read_csv(constraints_path, encoding='latin-1')

            # Convert timestamps to relative time
            for df in [self.particles, self.robot, self.constraints, self.performance]:
                if 'timestamp' in df.columns:
                    df['time_elapsed'] = df['timestamp'] - df['timestamp'].min()

            # Calculate path lengths
            self.robot['distance'] = self.calculate_path_length(self.robot)

        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def calculate_path_length(self, df):
        """Vectorized implementation"""
        lats = df['lat'].values
        lons = df['lon'].values

        # Calculate pairwise distances
        dlat = np.radians(np.diff(lats))
        dlon = np.radians(np.diff(lons))
        a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(lats[:-1])) *
             np.cos(np.radians(lats[1:])) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371 * c  # Earth radius in km

        return np.concatenate(([0], np.cumsum(distances)))

    def calculate_metrics(self):
        """Pre-calculate key metrics"""
        # Existing particle metrics
        self.particle_metrics = self.particles.groupby('step').agg({
            'weight': ['mean', 'std', 'min', 'max'],
            'stuck_counter': 'sum'
        }).reset_index()
        self.particle_metrics.columns = ['step', 'weight_mean', 'weight_std',
                                         'weight_min', 'weight_max', 'stuck_count']

        # Calculate localization error
        self.localization_error = self.calculate_localization_error()

        # Add accuracy metrics (1 - normalized error)
        if not self.localization_error.empty:
            max_error = self.localization_error['mean_error'].max()
            self.localization_error['accuracy'] = 1 - (self.localization_error['mean_error'] /
                                                       (max_error if max_error > 0 else 1))
            self.localization_error['accuracy_pct'] = self.localization_error['accuracy'] * 100

        # Existing convergence metrics
        self.convergence_data = self.analyze_convergence()

    def calculate_localization_error(self):
        """Calculate error between particles and robot position"""
        error_data = []
        for step in tqdm(self.robot['step'].unique(), desc="Calculating errors"):
            robot_pos = self.robot[self.robot['step'] == step][['lat', 'lon']].values[0]
            particles = self.particles[self.particles['step'] == step]

            if len(particles) == 0:
                continue

            # Calculate distances from robot
            distances = particles.apply(
                lambda row: geodesic((row['lat'], row['lon']), robot_pos).kilometers,
                axis=1
            )

            error_data.append({
                'step': step,
                'mean_error': distances.mean(),
                'median_error': distances.median(),
                'max_error': distances.max(),
                'min_error': distances.min(),
                'std_error': distances.std()
            })

        return pd.DataFrame(error_data)

    def analyze_convergence(self):
        """Analyze particle convergence over time"""
        convergence = []
        window_size = max(5, len(self.robot) // 5)  # 5% of steps as window

        for i in range(window_size, len(self.robot)):
            steps = self.robot['step'].iloc[i - window_size:i]
            errors = self.localization_error[
                self.localization_error['step'].isin(steps)
            ]['mean_error']

            convergence.append({
                'step': self.robot['step'].iloc[i],
                'mean_error': errors.mean(),
                'trend': self.calculate_trend(errors)
            })

        return pd.DataFrame(convergence)

    @staticmethod
    def calculate_trend(series):
        """Calculate linear trend of a series"""
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return 'improving' if slope < -0.001 else 'stable' if abs(slope) <= 0.001 else 'diverging'

    def analyze_all(self):
        """Run complete analysis"""
        print(f"\nAnalyzing simulation: {os.path.basename(self.simulation_dir)}")
        print(f"Total steps: {len(self.robot['step'].unique())}")
        print(f"Total particles: {len(self.particles['particle_id'].unique())}")

        # Generate all visualizations
        #self.plot_particle_distribution()
        self.plot_error_evolution()
        #self.plot_weight_distribution()
        #self.plot_robot_path()
        #self.plot_constraint_impact()
        self.plot_localization_accuracy()
        #self.create_interactive_plot()
        self.generate_report()

    def plot_particle_distribution(self):
        """Visualize particle spread at key steps"""
        steps = np.linspace(0, self.robot['step'].max(), 5, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, step in enumerate(steps):
            ax = axes[i]
            step_data = self.particles[self.particles['step'] == step]

            if len(step_data) == 0:
                continue

            # Create hexbin plot
            hb = ax.hexbin(step_data['lon'], step_data['lat'],
                           gridsize=25, cmap='viridis',
                           mincnt=1, bins='log')

            # Plot robot position
            robot_pos = self.robot[self.robot['step'] == step]
            ax.scatter(robot_pos['lon'], robot_pos['lat'],
                       color=self.colors['robot'], s=100, label='Robot')

            ax.set_title(f'Step {step}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            # Add colorbar
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Particle density (log)')

        # Add overall trajectory on last subplot
        self.plot_trajectory(axes[-1])
        fig.tight_layout()
        plt.savefig(os.path.join(self.simulation_dir, 'particle_distribution.png'))
        plt.close()

    def plot_trajectory(self, ax):
        """Helper to plot trajectory on given axis"""
        # Plot all particle paths (sampled)
        particles_sample = self.particles[self.particles['particle_id'].isin(
            np.random.choice(self.particles['particle_id'].unique(),
                             min(100, len(self.particles['particle_id'].unique())),
                             replace=False)
        )]

        for pid, group in particles_sample.groupby('particle_id'):
            ax.plot(group['lon'], group['lat'],
                    color=self.colors['particles'], alpha=0.1, linewidth=0.5)

        # Plot robot path
        ax.plot(self.robot['lon'], self.robot['lat'],
                color=self.colors['robot'], linewidth=2, label='Robot Path')

        ax.set_title('Complete Trajectories')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()

    def plot_error_evolution(self):
        """Plot only mean error evolution (single plot)"""
        fig, ax = plt.subplots(figsize=(12, 6))  # Single subplot

        # Mean error plot only
        sns.lineplot(data=self.localization_error, x='step', y='mean_error',
                     ax=ax, color=self.colors['particles'])
        ax.set_title('Mean Localization Error')

        plt.tight_layout()
        plt.savefig(os.path.join(self.simulation_dir, 'mean_error_evolution.png'))

    def plot_weight_distribution(self):
        """Visualize particle weight dynamics"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Weight distribution over time
        sns.violinplot(data=self.particles, x='step', y='weight',
                       ax=axes[0], color=self.colors['particles'])
        axes[0].set_title('Particle Weight Distribution')
        axes[0].set_xlabel('Simulation Step')
        axes[0].set_ylabel('Weight')

        # Weight vs error
        merged = pd.merge(self.particles, self.localization_error, on='step')
        sns.scatterplot(data=merged, x='weight', y='mean_error',
                        ax=axes[1], hue='stuck_counter', palette='viridis')
        axes[1].set_title('Weight vs Localization Error')
        axes[1].set_xlabel('Particle Weight')
        axes[1].set_ylabel('Mean Error (km)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.simulation_dir, 'weight_distribution.png'))
        plt.close()

    def create_interactive_plot(self):
        """Enhanced with animation and more controls"""
        fig = go.Figure()

        # Create animation frames
        frames = []
        steps = sorted(self.robot['step'].unique())

        for step in steps:
            frame_data = [
                # Robot position
                go.Scatter(
                    x=[self.robot[self.robot['step'] == step]['lon'].values[0]],
                    y=[self.robot[self.robot['step'] == step]['lat'].values[0]],
                    mode='markers',
                    marker=dict(size=15, color=self.colors['robot'])
                ),
                # Particles
                go.Scatter(
                    x=self.particles[self.particles['step'] == step]['lon'],
                    y=self.particles[self.particles['step'] == step]['lat'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.particles[self.particles['step'] == step]['weight'],
                        colorscale='Viridis',
                        opacity=0.7,
                        showscale=True
                    )
                )
            ]
            frames.append(go.Frame(data=frame_data, name=str(step)))

        # Add slider and play button
        sliders = [{
            'steps': [{
                'args': [[f.name], {'frame': {'duration': 100, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 50}}],
                'label': f'Step {f.name}',
                'method': 'animate'
            } for f in frames],
            'active': 0,
            'transition': {'duration': 300},
        }]

        fig.update_layout(
            sliders=sliders,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                        'fromcurrent': True,
                                        'transition': {'duration': 50}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                          'mode': 'immediate',
                                          'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ]
            }]
        )

        fig.write_html(os.path.join(self.simulation_dir, 'animated_visualization.html'))

    def generate_report(self):
        """Generate analysis report with all metrics"""
        report_path = os.path.join(self.simulation_dir, "analysis_report.txt")

        with open(report_path, 'w') as f:
            # Header and basic info
            f.write(f"Simulation Analysis Report\n{'=' * 50}\n\n")
            f.write(f"Simulation ID: {self.metadata.get('simulation_id', 'N/A')}\n")
            f.write(f"Start Time: {self.metadata.get('start_time', 'N/A')}\n")
            f.write(f"Duration: {float(self.metadata.get('duration_sec', 0)):.2f} seconds\n")
            f.write(f"Total Steps: {len(self.robot['step'].unique())}\n")
            f.write(f"Total Particles: {len(self.particles['particle_id'].unique())}\n\n")

            # Localization Accuracy Section
            f.write("Localization Accuracy Metrics\n{'=' * 50}\n")
            if hasattr(self, 'localization_error') and not self.localization_error.empty:
                acc = self.localization_error['accuracy_pct']
                f.write(f"Final Accuracy: {acc.iloc[-1]:.1f}%\n")
                f.write(f"Peak Accuracy: {acc.max():.1f}% (step {acc.idxmax()})\n")
                f.write(f"Average Accuracy: {acc.mean():.1f}%\n")
                f.write(f"Minimum Accuracy: {acc.min():.1f}% (step {acc.idxmin()})\n")
                f.write(f"Standard Deviation: {acc.std():.1f}%\n\n")

                # Accuracy trend analysis
                improving = len(self.convergence_data[self.convergence_data['trend'] == 'improving'])
                stable = len(self.convergence_data[self.convergence_data['trend'] == 'stable'])
                diverging = len(self.convergence_data[self.convergence_data['trend'] == 'diverging'])
                f.write(f"Accuracy Trends:\n")
                f.write(f"  - Improving phases: {improving} steps\n")
                f.write(f"  - Stable phases: {stable} steps\n")
                f.write(f"  - Diverging phases: {diverging} steps\n\n")
            else:
                f.write("No accuracy data available\n\n")

            # Particle Statistics Section
            f.write("Particle Statistics\n{'=' * 50}\n")
            if hasattr(self, 'particle_metrics') and not self.particle_metrics.empty:
                f.write(self.particle_metrics.describe().to_string())
                f.write("\n\n")
            else:
                f.write("No particle metrics available\n\n")

            # Constraint Impact Section
            f.write("Constraint Impact\n{'=' * 50}\n")
            if hasattr(self, 'constraints') and not self.constraints.empty:
                f.write(f"Total Constraints Applied: {len(self.constraints)}\n")
                f.write(f"Constraint Cities: {', '.join(self.constraints['city'].unique())}\n")

                # Show constraint effects on accuracy
                if hasattr(self, 'localization_error'):
                    for _, row in self.constraints.iterrows():
                        step = row['step']
                        if step in self.localization_error['step'].values:
                            before = self.localization_error[
                                self.localization_error['step'] < step]['accuracy_pct'].mean()
                            after = self.localization_error[
                                self.localization_error['step'] >= step]['accuracy_pct'].mean()
                            f.write(f"\nConstraint at step {step} ({row['city']} {row['distance_km']}km):\n")
                            f.write(f"  - Avg accuracy before: {before:.1f}%\n")
                            f.write(f"  - Avg accuracy after: {after:.1f}%\n")
                            f.write(f"  - Change: {after - before:+.1f}%\n")
                f.write("\n")
            else:
                f.write("No constraints were applied\n\n")

            # Performance Summary Section
            f.write("Performance Summary\n{'=' * 50}\n")
            if hasattr(self, 'performance') and not self.performance.empty:
                f.write(f"Final Effective Particles: {self.performance['effective_particles'].iloc[-1]:.1f}\n")
                f.write(f"Average Effective Particles: {self.performance['effective_particles'].mean():.1f}\n")
                f.write(f"Resampling Events: {self.metadata.get('total_resampling_events', 0)}\n")
                f.write(f"Stuck Particles (final step): {self.particle_metrics['stuck_count'].iloc[-1]}\n")
                f.write(f"Final Particle Weight Mean: {self.particle_metrics['weight_mean'].iloc[-1]:.3f}\n")
            else:
                f.write("No performance data available\n")

            # Visualizations Summary
            f.write("\nGenerated Visualizations\n{'=' * 50}\n")
            f.write("- Particle distribution at key steps\n")
            f.write("- Error evolution over time\n")
            f.write("- Weight distribution analysis\n")
            f.write("- Robot path with milestones\n")
            f.write("- Constraint impact analysis\n")
            f.write("- Localization accuracy trends\n")
            f.write("- Interactive visualization (HTML)\n")

        print(f"Analysis report saved to: {report_path}")

    def plot_robot_path(self):
        """Visualize the robot's complete path with milestones"""
        plt.figure(figsize=self.figsize)

        # Plot robot path
        plt.plot(self.robot['lon'], self.robot['lat'],
                 color=self.colors['robot'], linewidth=2, label='Robot Path')

        # Plot milestones if available
        if hasattr(self, 'milestone_memory') and self.milestone_memory:
            milestones = pd.DataFrame(self.milestone_memory)
            plt.scatter(milestones['position'].apply(lambda x: x[1]),  # lon
                        milestones['position'].apply(lambda x: x[0]),  # lat
                        color=self.colors['constraints'], s=100,
                        marker='*', label='Milestones')

        plt.title('Robot Path with Milestones')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)

        # Save and close
        plt.savefig(os.path.join(self.simulation_dir, 'robot_path.png'))
        plt.close()

    def plot_constraint_impact(self):
        """Visualize how constraints affected particle distribution"""
        if self.constraints.empty:
            return

        # Get steps where constraints were added
        constraint_steps = self.constraints['step'].unique()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Before/after constraint comparison
        for i, step in enumerate(constraint_steps[:2]):  # Limit to first 2 constraints
            ax = axes[i]

            # Get particles before and after constraint
            particles_before = self.particles[
                (self.particles['step'] >= step - 5) &
                (self.particles['step'] < step)
                ]
            particles_after = self.particles[
                (self.particles['step'] >= step) &
                (self.particles['step'] <= step + 5)
                ]

            # Plot particle distributions
            ax.scatter(particles_before['lon'], particles_before['lat'],
                       color=self.colors['particles'], alpha=0.3, label='Before')
            ax.scatter(particles_after['lon'], particles_after['lat'],
                       color=self.colors['constraints'], alpha=0.3, label='After')

            # Plot robot position
            robot_pos = self.robot[self.robot['step'] == step]
            ax.scatter(robot_pos['lon'], robot_pos['lat'],
                       color=self.colors['robot'], s=100, label='Robot')

            ax.set_title(f'Constraint Impact (Step {step})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.simulation_dir, 'constraint_impact.png'))
        plt.close()

    def init_data_files(self):
        """Initialize all data files with UTF-8 encoding"""
        # Metadata
        with open(os.path.join(self.output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "simulation_id": self.simulation_id,
                "start_time": datetime.now().isoformat(),
                "system": "Particle Filter Localization",
                "version": "2.1"
            }, f, indent=2, ensure_ascii=False)  # ensure_ascii=False allows non-ASCII characters

        # Main data files
        files = {
            "particles.csv": [...],
            "robot.csv": [...],
            "performance.csv": [...],
            "constraints.csv": [...]
        }

        for filename, headers in files.items():
            with open(os.path.join(self.output_dir, filename), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)



    def plot_localization_accuracy(self):
        """Visualize localization accuracy over time"""
        if not hasattr(self, 'localization_error') or self.localization_error.empty:
            print("No localization error data available")
            return

        plt.figure(figsize=self.figsize)

        # Plot accuracy percentage
        plt.plot(self.localization_error['step'],
                 self.localization_error['accuracy_pct'],
                 color=self.colors['particles'],
                 linewidth=2,
                 label='Localization Accuracy')

        # Highlight constraint points if available
        if not self.constraints.empty:
            constraint_steps = self.constraints['step'].unique()
            for step in constraint_steps:
                if step in self.localization_error['step'].values:
                    acc = self.localization_error.loc[
                        self.localization_error['step'] == step, 'accuracy_pct'].values[0]
                    plt.scatter(step, acc, color=self.colors['constraints'],
                                s=100, zorder=5)

        plt.title('Particle Localization Accuracy Over Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()

        # Save and close
        plt.savefig(os.path.join(self.simulation_dir, 'localization_accuracy.png'))
        plt.close()

def __init__(self, simulation_folder, config=None):
    """Allow customization via config"""
    """self.config = {
        'sample_size': 100,  # Particles to sample for visualization
        'plot_steps': 5,     # Number of steps to show in distribution plots
        'error_window': 0.05 # Percentage of steps for convergence window
    }"""
    if config:
        self.config.update(config)


if __name__ == "__main__":
    # For a folder named "particle_filter_20250511_021014" in simulation_data/
    analyzer = SimulationAnalyzer("particle_filter_20250812_210852")

    # Or for a folder at a specific path
    # analyzer = SimulationAnalyzer("/full/path/to/simulation_folder")

    analyzer.analyze_all()

