import csv
import os
import time
from datetime import datetime
import json
import numpy as np
from geopy.distance import geodesic


class DataLogger:
    def __init__(self, simulation_name="particle_filter"):
        self.main_dir = "simulation_data"
        os.makedirs(self.main_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_id = f"{simulation_name}_{timestamp}"
        self.output_dir = os.path.join(self.main_dir, self.simulation_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Add directory for simulation pictures
        self.pictures_dir = os.path.join(self.output_dir, "simulation_pictures")
        os.makedirs(self.pictures_dir, exist_ok=True)

        # Initialize internal state
        self.start_time = time.time()
        self.last_robot_position = None
        self.last_robot_time = self.start_time
        self.resampling_count = 0
        self.constraint_updates = 0

        # Initialize data files
        self.init_data_files()

    def save_simulation_snapshot(self, fig, step):
        """Save the current simulation visualization as an image"""
        filename = f"step_{step:04d}.png"
        filepath = os.path.join(self.pictures_dir, filename)
        try:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved simulation snapshot to {filepath}")
        except Exception as e:
            print(f"Error saving simulation snapshot: {str(e)}")

    def init_data_files(self):
        """Initialize all data files"""
        # Metadata
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
            json.dump({
                "simulation_id": self.simulation_id,
                "start_time": datetime.now().isoformat(),
                "system": "Particle Filter Localization",
                "version": "2.1"
            }, f, indent=2)

        # Main data files
        files = {
            "particles.csv": [
                'timestamp', 'step', 'particle_id', 'lat', 'lon',
                'weight', 'color', 'opacity', 'stuck_counter',
                'consecutive_stuck', 'current_road'
            ],
            "robot.csv": [
                'timestamp', 'step', 'lat', 'lon', 'speed_kmh',
                'segment_index', 'segment_progress',
                'active_constraints', 'milestones_passed'
            ],
            "performance.csv": [
                'timestamp', 'step', 'elapsed_sec',
                'effective_particles', 'weight_mean', 'weight_std',
                'stuck_particles', 'resampling_events', 'constraint_updates'
            ],
            "constraints.csv": [
                'timestamp', 'step', 'city', 'distance_km',
                'constraint_type', 'is_active'
            ]
        }

        for filename, headers in files.items():
            with open(os.path.join(self.output_dir, filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_particles(self, particles, step):
        """Particle logging with statistics"""
        timestamp = time.time() - self.start_time
        weights = []

        with open(os.path.join(self.output_dir, "particles.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            for i, p in enumerate(particles):
                try:
                    lat, lon = p.get_position()
                    writer.writerow([
                        timestamp, step, i, lat, lon,
                        p.weight, p.color, p.opacity, p.stuck_counter,
                        p.consecutive_stuck, getattr(p, '_current_road_number', '')
                    ])
                    weights.append(p.weight)
                except Exception as e:
                    print(f"Error logging particle {i}: {str(e)}")
                    continue

        # Calculate and log performance metrics
        if weights:
            self.log_performance(step, timestamp, weights)

    def log_performance(self, step, timestamp, weights):
        """Calculate and log performance metrics"""
        weights = np.array(weights)
        total_weight = weights.sum()
        effective_particles = 1.0 / sum((weights / total_weight) ** 2) if total_weight > 0 else 0

        with open(os.path.join(self.output_dir, "performance.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, step, timestamp,
                effective_particles, weights.mean(), weights.std(),
                sum(w < 0.01 for w in weights),  # Count of stuck particles
                self.resampling_count,
                self.constraint_updates
            ])

    def log_robot_state(self, robot, step):
        """Robot state logging"""
        timestamp = time.time() - self.start_time

        # Calculate speed
        speed = 0.0
        if self.last_robot_position and self.last_robot_time:
            try:
                dist_km = geodesic(self.last_robot_position, robot.position).kilometers
                time_hr = (timestamp - self.last_robot_time) / 3600
                speed = dist_km / time_hr if time_hr > 0 else 0.0
            except Exception as e:
                print(f"Error calculating speed: {str(e)}")

        self.last_robot_position = robot.position
        self.last_robot_time = timestamp

        with open(os.path.join(self.output_dir, "robot.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            try:
                writer.writerow([
                    timestamp, step,
                    robot.position[0], robot.position[1], speed,
                    robot.current_segment_index,
                    robot.distance_into_segment,
                    json.dumps(robot.display_constraints) if hasattr(robot, 'display_constraints') else '{}',
                    len(robot.milestone_memory) if hasattr(robot, 'milestone_memory') else 0
                ])
            except Exception as e:
                print(f"Error logging robot state: {str(e)}")

    def log_constraints(self, constraints, step, is_active=True):
        """Constraint logging with JSON"""
        if not constraints:
            return

        timestamp = time.time() - self.start_time
        self.constraint_updates += 1

        with open(os.path.join(self.output_dir, "constraints.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            try:
                for city, distance in constraints.items():
                    writer.writerow([
                        timestamp, step, city, distance,
                        "distance_constraint", is_active
                    ])
            except Exception as e:
                print(f"Error logging constraints: {str(e)}")

    def log_resampling_event(self):
        """Track resampling occurrences"""
        self.resampling_count += 1

    def save_to_file(self):
        """Finalize logging with summary statistics"""
        metadata = {
            "end_time": datetime.now().isoformat(),
            "duration_sec": time.time() - self.start_time,
            "total_resampling_events": self.resampling_count,
            "total_constraint_updates": self.constraint_updates
        }

        # Update metadata file
        with open(os.path.join(self.output_dir, "metadata.json"), 'r+') as f:
            try:
                data = json.load(f)
                data.update(metadata)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            except Exception as e:
                print(f"Error saving metadata: {str(e)}")

        print(f"Simulation data saved to: {self.output_dir}")

