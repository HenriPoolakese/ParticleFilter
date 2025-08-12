from config import USE_ROAD_NUMBERS
from main import *
from utils.path import extract_interpolated_path


class Robot:
    def __init__(self, start_city, end_city, road_graph, data_logger, movement_step_size):
        self.current_step = 0
        self.road_graph = road_graph
        self.movement_step_size = movement_step_size
        # self.start_node = ox.distance.nearest_nodes(road_graph, CITIES[start_city][1], CITIES[start_city][0])
        # 59.396113, 24.820716
        self.start_node = ox.distance.nearest_nodes(road_graph, CITIES[start_city][1], CITIES[start_city][0])
        self.end_node = ox.distance.nearest_nodes(road_graph, CITIES[end_city][1], CITIES[end_city][0])
        self.full_path_nodes = nx.shortest_path(road_graph, self.start_node, self.end_node, weight='length')
        self.full_path = extract_interpolated_path(road_graph, self.full_path_nodes, resolution=10)
        self.current_segment_index = 0
        self.distance_into_segment = 0.0
        self.position = self.full_path[0]
        # Memory systems
        self.milestone_memory = []
        self.active_constraints = {}
        self.old_constraints = []
        self.display_constraints = {}
        # Road number tracking
        self.current_road_number = None
        self.known_road_numbers = set()
        # Visualization
        self.current_milestone = None
        self.data_logger = data_logger

    def move(self):
        self.current_step += 1
        """ Mmovement using distance-based advancement through dense segmented path"""
        distance_remaining = self.movement_step_size

        while distance_remaining > 0 and self.current_segment_index < len(self.full_path) - 1:
            p1 = self.full_path[self.current_segment_index]
            p2 = self.full_path[self.current_segment_index + 1]
            segment_length = geodesic(p1, p2).meters
            remaining_in_segment = segment_length - self.distance_into_segment

            if distance_remaining < remaining_in_segment:
                self.distance_into_segment += distance_remaining
                distance_remaining = 0
            else:
                distance_remaining -= remaining_in_segment
                self.current_segment_index += 1
                self.distance_into_segment = 0.0

        # Update position
        if self.current_segment_index < len(self.full_path) - 1:
            p1 = self.full_path[self.current_segment_index]
            p2 = self.full_path[self.current_segment_index + 1]
            total_dist = geodesic(p1, p2).meters
            frac = min(1.0, self.distance_into_segment / total_dist)
            self.position = (
                p1[0] + frac * (p2[0] - p1[0]),
                p1[1] + frac * (p2[1] - p1[1])
            )
        else:
            self.position = self.full_path[-1]

        self.check_for_destinations()

    def check_for_destinations(self):
        """Check for nearby destination signs"""
        detection_radius = 100  # meters
        found_new = False

        if "milestones" not in self.road_graph.graph:
            return found_new

        robot_lat, robot_lon = self.position

        for milestone in self.road_graph.graph["milestones"]:
            # Handle both list and tuple formats for coordinates
            milestone_coordinates = milestone["x,y"]
            if isinstance(milestone_coordinates, list):
                mlat, mlon = milestone_coordinates[0], milestone_coordinates[1]
            else:  # Assume tuple
                mlat, mlon = milestone_coordinates

            distance = geodesic((robot_lat, robot_lon), (mlat, mlon)).meters

            if distance <= detection_radius and "destination" in milestone:
                print(f"Found milestone at {distance:.1f}m: {milestone['destination']}")
                if self.process_destination(milestone):
                    found_new = True

        return found_new

    def process_destination(self, milestone):
        """Process destination sign and update constraints"""
        destination = milestone.get('destination', [])
        xy = milestone.get('x,y', [])

        if isinstance(destination, str):
            destinations = [destination.strip()]
        elif isinstance(destination, list):
            destinations = [d.strip() for d in destination if d.strip()]
        else:
            return False

        # Process road number if enabled
        new_road_number = None
        if USE_ROAD_NUMBERS and 'ref' in milestone:
            road_num = str(milestone['ref'])
            if road_num:
                new_road_number = road_num
                self.known_road_numbers.add(road_num)

        self.data_logger.log_constraints(self.active_constraints, self.current_step)
        self.data_logger.log_robot_state(self, self.current_step)

        # Process distance constraints
        new_constraints = {}
        for dest_str in destinations:
            if '_' in dest_str:  # Format: "City_Distance"
                try:
                    city, dist = dest_str.split('_', 1)
                    city = self.normalize_city_name(city.strip())
                    dist = float(''.join(c for c in dist if c.isdigit() or c == '.'))
                    if city in CITIES:
                        new_constraints[city] = dist
                except:
                    continue

        if len(self.old_constraints) != 0 and self.old_constraints[-1] == xy:
            return False

        self.old_constraints.append(xy)
        # Check if we have new information
        has_new_info = False

        # Check for new road number
        if USE_ROAD_NUMBERS and new_road_number and new_road_number != self.current_road_number:
            self.current_road_number = new_road_number
            print("New Road number, ",self.current_road_number )
            has_new_info = True

        # Check for new constraints
        if new_constraints != self.active_constraints:
            self.active_constraints.update(new_constraints)
            self.display_constraints.update(new_constraints)
            has_new_info = True

        if has_new_info:
            # Add to permanent memory
            self.milestone_memory.append({
                'constraints': copy.deepcopy(new_constraints),
                'road_number': new_road_number if USE_ROAD_NUMBERS else None,
                'position': copy.deepcopy(self.position),
                'step': self.current_step
            })
            return True

        return False

    def normalize_city_name(self, city_name):
        """Normalize city names to match our known cities"""
        if not city_name:
            return None

        # Create a mapping of common alternatives
        special_char_map = {
            'Ãµ': 'õ',
            'Ãµ': 'õ',
            'Ã¤': 'ä',
            'Ã¶': 'ö',
            'Ã¼': 'ü',
            'Ã': 'õ',
            'VÃµru': 'Võru',
            'JÃµgeva': 'Jõgeva'
        }

        # Replace common encoding errors
        for wrong, correct in special_char_map.items():
            city_name = city_name.replace(wrong, correct)

        city_lower = city_name.lower()
        for known_city in CITIES:
            if known_city.lower() == city_lower:
                return known_city
            if city_lower in known_city.lower() or known_city.lower() in city_lower:
                return known_city
        return None
