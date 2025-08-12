import copy
import itertools
import json
import os
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from shapely.geometry.linestring import LineString

from utils.RoadNetworkManager import CITIES
from main import robot_step_distance_global
from robot.particle import Particle


def calculate_accurate_distance(road_graph, point1, point2, max_search_distance=1000):
    #Calculate distance through road network
    try:
        # Get nearest nodes with a reasonable search distance
        node1 = ox.distance.nearest_nodes(road_graph, point1[1], point1[0])
        node2 = ox.distance.nearest_nodes(road_graph, point2[1], point2[0])
        print(point1,point2)
        # Verify nodes exist in graph
        if node1 not in road_graph or node2 not in road_graph:
            raise ValueError("One or both nodes not found in graph")

        # Verify nodes are reasonably close to input points
        node1_point = (road_graph.nodes[node1]['y'], road_graph.nodes[node1]['x'])
        node2_point = (road_graph.nodes[node2]['y'], road_graph.nodes[node2]['x'])

        direct_dist1 = geodesic(point1, node1_point).meters
        direct_dist2 = geodesic(point2, node2_point).meters

        if direct_dist1 > max_search_distance or direct_dist2 > max_search_distance:
            raise ValueError("Nearest node too far from input point")

        # Try to calculate shortest path
        try:
            path_length = nx.shortest_path_length(road_graph, node1, node2, weight='length')
        except nx.NetworkXNoPath:
            raise ValueError("No path exists between nodes")

        # Verify the result isn't unreasonably long
        direct_distance = geodesic(point1, point2).meters
        if path_length > direct_distance * 3:  # Allow 3x longer than direct distance
            raise ValueError("Calculated path too long")


        return path_length / 1000, 1.0  # Convert to kilometers

    except Exception as e:
        print(f"Network distance calculation failed: {e}. Using geodesic distance.")
        # Fallback to geodesic distance if path finding fails
        # Not good
        return geodesic(point1, point2).kilometers, 0.1


def enhance_road_graph(road_graph, max_nodes=20000, min_segment_length=2000):
    """Enhanced version that removes dead ends first, then adds nodes"""
    from shapely.geometry import LineString, Point

    print(f"Original graph has {len(road_graph.nodes())} nodes and {len(road_graph.edges())} edges")

    # Create a working copy
    enhanced_graph = road_graph.copy()

    # Remove dead ends first, roads that lead to nowhere (more efficient)
    print("Removing dead ends...")
    changed = True
    iterations = 0
    max_iterations = 5

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        dead_ends = []

        # Find all dead ends (nodes with 1 edge)
        for node in enhanced_graph.nodes():
            if enhanced_graph.degree(node) == 1:
                dead_ends.append(node)

        # For each dead end, trace back the entire branch
        nodes_to_remove = set()
        for node in dead_ends:
            current = node
            branch = [current]

            while True:
                neighbors = list(enhanced_graph.neighbors(current))
                if len(neighbors) != 1:
                    break

                if neighbors[0] in branch:
                    break  # Prevent cycles

                current = neighbors[0]
                branch.append(current)

            # If we ended at another dead end, remove whole branch
            if enhanced_graph.degree(current) <= 1:
                nodes_to_remove.update(branch)

        if nodes_to_remove:
            enhanced_graph.remove_nodes_from(nodes_to_remove)
            changed = True
            print(f"Iteration {iterations}: Removed {len(nodes_to_remove)} dead end nodes")

    # Now add intermediate nodes to remaining edges
    print("Adding intermediate nodes...")
    new_node_id = itertools.count(start=max(enhanced_graph.nodes()) + 1)
    added_nodes = 0
    edges_to_process = list(enhanced_graph.edges(data=True))

    for u, v, data in edges_to_process:
        if added_nodes >= max_nodes:
            break

        # Get edge geometry
        if 'geometry' in data:
            line = data['geometry']
            coords = list(line.coords)
        else:
            coords = [
                (enhanced_graph.nodes[u]['x'], enhanced_graph.nodes[u]['y']),
                (enhanced_graph.nodes[v]['x'], enhanced_graph.nodes[v]['y'])
            ]
            line = LineString(coords)

        edge_length = line.length
        if edge_length < min_segment_length * 1.5:  # Don't split short edges
            continue

        # Calculate number of points to add
        num_points = min(int(edge_length / min_segment_length), 10)

        # Remove original edge
        enhanced_graph.remove_edge(u, v)

        # Add intermediate points
        prev_node = u
        for i in np.linspace(0, 1, num_points + 2)[1:-1]:  # Skip endpoints
            if added_nodes >= max_nodes:
                break

            point = line.interpolate(i, normalized=True)
            new_node = next(new_node_id)
            enhanced_graph.add_node(new_node, x=point.x, y=point.y)

            # Create new edge segment
            new_data = data.copy()
            segment_length = edge_length * (i - line.project(
                Point(enhanced_graph.nodes[prev_node]['x'], enhanced_graph.nodes[prev_node]['y'])) / line.length)
            new_data['length'] = segment_length
            enhanced_graph.add_edge(prev_node, new_node, **new_data)

            prev_node = new_node
            added_nodes += 1

            # Add final segment
            final_data = data.copy()
            final_length = edge_length - line.project(Point(enhanced_graph.nodes[prev_node]['x'],
                                                            enhanced_graph.nodes[prev_node]['y']))
            final_data['length'] = final_length
            enhanced_graph.add_edge(prev_node, v, **final_data)

            print(f"Added {added_nodes} intermediate nodes")

            # Final check for connectivity
            undirected = enhanced_graph.to_undirected()
            components = list(nx.connected_components(undirected))
            if len(components) > 1:
                largest = max(components, key=len)
            for component in components:
                if component != largest:
                    enhanced_graph.remove_nodes_from(component)
            print(f"Removed {len(components) - 1} disconnected components")

            print(f"Final graph has {len(enhanced_graph.nodes())} nodes and {len(enhanced_graph.edges())} edges")
    return enhanced_graph

def normalize_weights(particles):
    total_weight = sum(p.weight for p in particles)
    if total_weight > 0:
        for p in particles:
            p.weight /= total_weight

def scatter_particles_randomly(particles):
    """Scatter particles randomly across all possible nodes when weights get too low"""
    if not particles:
        return

    road_graph = particles[0].road_graph
    all_nodes = list(road_graph.nodes())

    for particle in particles:
        # Reset particle to random position
        random_node = random.choice(all_nodes)
        particle.current_node = random_node
        particle.position = (road_graph.nodes[random_node]['y'],
                             road_graph.nodes[random_node]['x'])
        particle.current_edge = None
        particle.current_path = []
        particle.find_new_path()

        # Reset weight and other properties
        particle.weight = 1.0 / len(particles)
        particle.stuck_counter = 0
        particle.consecutive_stuck = 0
        particle.needs_weight_update = False
        particle.is_in_constraints = False
        particle.color = 'green'
        particle.opacity = 0.5

def initialize_particles(road_graph, count, distance_cache):
    """Initialize particles randomly across the entire map"""
    nodes = list(road_graph.nodes())

    # Create particles
    particles = []
    for i in range(count):
        # Select a random node from the entire graph
        random_node = random.choice(nodes)
        pos = (road_graph.nodes[random_node]['y'], road_graph.nodes[random_node]['x'])

        p = Particle(road_graph, robot_step_distance_global, distance_cache)
        p.weight = 1 / count
        p.position = pos

        p.current_node = random_node
        p.find_new_path()
        particles.append(p)

    return particles



def precompute_distances(road_graph, cities, cache_file="distance_cache.json"):

    print("precompute----------------------------")
    # Check for existing cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    print("Precomputing distances...")
    distance_cache = defaultdict(dict)
    undirected_graph = road_graph.to_undirected()

    for city, (lat, lon) in cities.items():
        # Find nearest node for each city
        city_node = ox.distance.nearest_nodes(undirected_graph, lon, lat)

        # Compute shortest paths to all nodes
        lengths = nx.shortest_path_length(undirected_graph, source=city_node, weight='length')

        # Store in cache
        for node, dist in lengths.items():
            distance_cache[str(node)][city] = dist

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(distance_cache, f)

    return distance_cache



def resample_particles(particles):

    print(f"Starting resampling of {len(particles)} particles")
    normalize_weights(particles)
    start_time = time.time()

    weights = np.array([p.weight for p in particles])


    N = len(particles)
    new_particles = []
    step = 1.0 / N
    u = random.uniform(0, step)
    c = weights[0]
    i = 0

    for _ in range(N):
        while u > c:
            i += 1
            c += weights[i]

        # Clone with road-network-aware noise
        new_p = particles[i].clone()

        # Add constrained noise (stay on roads)
        if hasattr(new_p, 'current_path') and len(new_p.current_path) > 1:
            max_offset = min(2, len(new_p.current_path) // 3)
            offset = random.randint(-max_offset, max_offset)
            new_p.current_segment_index = max(0, min(
                len(new_p.current_path) - 2,
                new_p.current_segment_index + offset
            ))
            new_p.update_position()
            new_p.weight = new_p.weight / 1.5

        new_particles.append(new_p)
        u += step



    print(f"Resampling completed in {time.time() - start_time:.3f} seconds")

    return new_particles


