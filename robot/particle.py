import osmnx as ox
import numpy as np
import random
from geopy.distance import geodesic

from config import USE_ROAD_NUMBERS


class Particle:
    def __init__(self, road_graph, movement_step_size,distance_cache):
        self.visited_nodes = None
        self.road_graph = road_graph
        self.movement_step_size = float(movement_step_size * random.uniform(0.9, 1.2))
        self.stuck_counter = 0
        self.max_stuck_count = 1
        self.weight = 1
        self.color = 'green'
        self.opacity = 0.5
        self.consecutive_stuck = 0
        self.max_stuck_before_kill = 1
        self.needs_weight_update = False
        self.is_in_constraints = False
        self.current_road_number = None

        # Choose from all nodes, including intermediates
        self.current_node = random.choice(list(road_graph.nodes()))
        self.current_edge = None
        self.position = (road_graph.nodes[self.current_node]['y'],
                         road_graph.nodes[self.current_node]['x'])

        self.current_path = []
        self.current_segment_index = 0
        self.distance_into_segment = 0.0
        self.prev_node = None
        self.next_node = None
        self.current_edge_data = None
        self.find_new_path()
        # self.position = self.get_position()
        self.old_position = None
        self.distance_cache = distance_cache

    def clone(self):
        new_particle = Particle(self.road_graph, movement_step_size=self.movement_step_size, distance_cache=self.distance_cache)
        new_particle.__dict__.update(self.__dict__)  # shallow copy
        return new_particle

    def get_cached_distance(self, city):
        """Get cached distance to city in meters"""
        try:
            # Try exact node match first
            if str(self.current_node) in self.distance_cache:
                return self.distance_cache[str(self.current_node)].get(city, float('inf'))

            # Fallback to nearest node
            nearest_node = ox.distance.nearest_nodes(
                self.road_graph,
                self.position[1],  # lon
                self.position[0]  # lat
            )
            return self.distance_cache[str(nearest_node)].get(city, float('inf'))
        except:
            return float('inf')  # Safe fallback

    def update_position_from_edge(self):
        """Update position when on an edge"""
        u, v = self.current_edge
        edge_data = self.road_graph.edges[u, v, 0]

        if 'geometry' in edge_data:
            line = edge_data['geometry']
            point = line.interpolate(self.progress_along_edge, normalized=True)
            self.position = (point.y, point.x)
        else:
            u_pos = (self.road_graph.nodes[u]['y'], self.road_graph.nodes[u]['x'])
            v_pos = (self.road_graph.nodes[v]['y'], self.road_graph.nodes[v]['x'])
            self.position = (
                u_pos[0] + self.progress_along_edge * (v_pos[0] - u_pos[0]),
                u_pos[1] + self.progress_along_edge * (v_pos[1] - u_pos[1]))

    def transition_to_node(self, node):
        """Transition from edge movement to node"""
        self.current_node = node
        self.current_edge = None
        self.progress_along_edge = 0.0
        self.visited_nodes = node
        self.find_new_path()

    def update_road_number(self):
        """Update current road number from edge data"""
        if hasattr(self, 'prev_node') and hasattr(self, 'current_node'):
            edges = self.road_graph.get_edge_data(self.prev_node, self.current_node)
            if edges and len(edges) > 0:
                edge_data = next(iter(edges.values()))
                self.current_road_number = str(edge_data.get('ref', '')) or None
            else:
                self.current_road_number = None

    def find_new_path(self):
        """Find a new path with connectivity checks"""
        neighbors = list(self.road_graph.neighbors(self.current_node))
        if neighbors:
            next_node = random.choice(neighbors)
            self.current_path = self.create_path(self.current_node, next_node)
            self.prev_node = self.current_node
            self.current_node = next_node
            self.current_segment_index = 0
            self.distance_into_segment = 0.0
            self.update_road_number()
            return True
        return False

    def create_path(self, u, v):
        """Create path between nodes with proper geometry"""
        edge_data = self.road_graph.get_edge_data(u, v)[0]
        if 'geometry' in edge_data:
            line = edge_data['geometry']
            return [(coord[1], coord[0]) for coord in line.coords]
        return [
            (self.road_graph.nodes[u]['y'], self.road_graph.nodes[u]['x']),
            (self.road_graph.nodes[v]['y'], self.road_graph.nodes[v]['x'])
        ]

    def move(self):

        noisy_step = self.movement_step_size * random.uniform(0.9, 1.3)
        self.old_position = self.position
        distance_remaining = noisy_step
        moved = False
        max_attempts = 3  # Max attempts to find a valid move
        attempts = 0

        while distance_remaining > 0 and attempts < max_attempts:
            attempts += 1

            # If on an edge, continue moving along it
            if self.current_edge:
                u, v = self.current_edge
                edge_data = self.road_graph.edges[u, v, 0]

                if 'geometry' in edge_data:
                    line = edge_data['geometry']
                    edge_length = line.length
                else:
                    u_pos = (self.road_graph.nodes[u]['y'], self.road_graph.nodes[u]['x'])
                    v_pos = (self.road_graph.nodes[v]['y'], self.road_graph.nodes[v]['x'])
                    edge_length = geodesic(u_pos, v_pos).meters

                movable_distance = edge_length * (1 - self.progress_along_edge)

                if distance_remaining < movable_distance:
                    self.progress_along_edge += distance_remaining / edge_length
                    distance_remaining = 0
                    moved = True
                else:
                    distance_remaining -= movable_distance
                    # Transition to the end node
                    self.transition_to_node(v)
                    moved = True

            # If on a node, choose a new path
            elif not self.current_path or self.current_segment_index >= len(self.current_path) - 1:
                if not self.find_new_path():
                    self.consecutive_stuck += 1
                    break  # Couldn't find a new path

            # Move along current path segment
            else:
                p1 = self.current_path[self.current_segment_index]
                p2 = self.current_path[self.current_segment_index + 1]
                segment_length = geodesic(p1, p2).meters

                if segment_length == 0:  # Handle zero-length segments
                    self.current_segment_index += 1
                    continue

                if self.distance_into_segment + distance_remaining < segment_length:
                    self.distance_into_segment += distance_remaining
                    distance_remaining = 0
                    moved = True
                else:
                    distance_remaining -= (segment_length - self.distance_into_segment)
                    self.current_segment_index += 1
                    self.distance_into_segment = 0
                    moved = True

        self.update_position()

        if not moved:
            self.consecutive_stuck += 1
            self.stuck_counter += 1
            # Try to unstuck by finding a new path
            if self.consecutive_stuck > 2:
                self.find_new_path()
                self.consecutive_stuck = 0
        else:
            self.consecutive_stuck = 0

        self.needs_weight_update = False or self.consecutive_stuck > 0

        return moved

    def update_position(self):
        """Update position based on current state"""
        if self.current_edge:
            self.update_position_from_edge()
        elif self.current_path and self.current_segment_index < len(self.current_path) - 1:
            p1 = self.current_path[self.current_segment_index]
            p2 = self.current_path[self.current_segment_index + 1]
            if p1 == p2:
                frac = 1.0
            else:
                distance = geodesic(p1, p2).meters
                frac = min(1.0, self.distance_into_segment / distance)
            self.position = (
                p1[0] + frac * (p2[0] - p1[0]),
                p1[1] + frac * (p2[1] - p1[1]))
        elif self.current_node:
            self.position = (self.road_graph.nodes[self.current_node]['y'],
                             self.road_graph.nodes[self.current_node]['x'])

    def get_position(self):
        if not self.current_path or self.current_segment_index >= len(self.current_path) - 1:
            return (self.road_graph.nodes[self.current_node]['y'],
                    self.road_graph.nodes[self.current_node]['x'])

        p1 = self.current_path[self.current_segment_index]
        p2 = self.current_path[self.current_segment_index + 1]

        distance = geodesic(p1, p2).meters
        if distance == 0:
            frac = 0.0
        else:
            frac = min(1.0, self.distance_into_segment / distance)

        return (
            p1[0] + frac * (p2[0] - p1[0]),
            p1[1] + frac * (p2[1] - p1[1])
        )

    def update_weight(self, constraints, road_number,robot):
        """weight update logic"""
        # Initialize with default weight
        total_weight = 1.0


        # Check stuck condition first (cheapest check)
        if self.stuck_counter > self.max_stuck_before_kill:
            self.weight = 1e-10
            self.is_in_constraints = False
            return self.weight

        print("CONSTRAINTS, ",constraints)

        print("Number, ", road_number)

        # Process constraints if they exist
        if constraints:
            constraint_weights = []


            for city, target_dist in constraints.items():
                try:
                    # Calculate distance once and reuse
                    cached_dist_m = self.get_cached_distance(city) / 1000  # Convert to km

                    error = abs(cached_dist_m - target_dist)



                    # Immediate kill check
                    if error >= 6.0:  # >=6km error
                        self.weight = 1e-10
                        self.is_in_constraints = False
                        return self.weight


                    # Calculate weight using piecewise linear function
                    if error < 2.0:
                        weight = 1.0
                    elif error < 4.0:
                        weight = 0.95
                    elif error < 6.0:
                        weight = 0.75
                    else:
                        weight = 0.25

                    constraint_weights.append(weight)

                except Exception as e:
                    print(f"Distance calculation failed: {e}")

                    break


            # Geometric mean of weights
            if constraint_weights:
                total_weight *= np.prod(constraint_weights) ** (1.0 / len(constraint_weights))


        #If road numbers are used
        if USE_ROAD_NUMBERS and road_number:

            current_road = None
            if hasattr(self, 'prev_node') and hasattr(self, 'current_node'):
                edges = self.road_graph.get_edge_data(self.prev_node, self.current_node)
                if edges and len(edges) > 0:
                    edge_data = next(iter(edges.values()))

                    current_road = str(edge_data.get('ref', '')) or None

            if current_road != road_number:
                if not constraints:
                    total_weight = 1e-10  # KILL if wrong road (strict mode)
                else:
                    total_weight *= 0.2  # STRONG penalty (10% weight)

            # Update final weight with numerical stability
        #self.weight = max(1e-10, min(1.0, total_weight))
        self.is_in_constraints = True

        return self.weight
