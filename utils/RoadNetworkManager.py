import json
import os
import osmnx as ox

import networkx as nx
from shapely.geometry.linestring import LineString

cache_file = 'road_data_cache.json'
CITIES = { # can add more, recommend to import tertiary road info from overpass-turbo and then use convert.py
    "Tallinn": (59.4372, 24.7453),  # Tallinn Old Town Square
    "Tartu": (58.3833, 26.7167),  # Tartu Town Hall Square
    "Narva": (59.3758, 28.1961),  # Narva Castle
    "Pärnu": (58.385959, 24.499921),  # Pärnu Rüütli Square
    "Kohtla-Järve": (59.3978, 27.2572),  # City administration
    "Viljandi": (58.3667, 25.6000),  # Viljandi Castle ruins
    "Maardu": (59.4781, 25.0161),  # Maardu city center
    "Rakvere": (59.3500, 26.3500),  # Rakvere Castle
    "Sillamäe": (59.3931, 27.7742),  # Sillamäe city center
    "Valga": (57.7833, 26.0333),  # Valga city center
    "Võru": (57.8486, 26.9928),  # Võru city center
    "Keila": (59.30854, 24.42246),  # Keila city center
    "Jõhvi": (59.3575, 27.4269),  # Jõhvi city center
    "Haapsalu": (58.9394, 23.5408),  # Haapsalu city center
    "Paide": (58.8833, 25.5572),  # Paide city center

}




def serialize_graph(road_graph):
    graph_data = nx.node_link_data(road_graph, edges="links")
    for edge in graph_data["links"]:
        if "geometry" in edge and isinstance(edge["geometry"], LineString):
            edge["geometry"] = list(edge["geometry"].coords)
    return graph_data


def save_graph_to_file(road_graph, cache_file):
    graph_data = serialize_graph(road_graph)
    with open(cache_file, 'w',encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)


def load_or_cache_road_data():
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                road_data = json.load(f)

            # Validate and clean milestones
            if "milestones" in road_data:
                road_data["milestones"] = [m for m in road_data["milestones"]
                                           if m.get("x,y") and len(m["x,y"]) == 2
                                           and m.get("destination")]

            road_graph = deserialize_graph(road_data)
            if "milestones" in road_data:
                road_graph.graph["milestones"] = road_data["milestones"]
            return road_graph
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            print("Regenerating road data...")

    # Fallback to downloading fresh data if cache is corrupt
    """custom_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link"]'
    road_graph = ox.graph_from_place("Estonia", network_type="drive", custom_filter=custom_filter)
    save_graph_to_file(road_graph)"""
    return road_graph



def deserialize_graph(graph_data):
    for edge in graph_data["links"]:
        if "geometry" in edge and isinstance(edge["geometry"], list):
            edge["geometry"] = LineString(edge["geometry"])
    return nx.node_link_graph(graph_data, edges="links")
