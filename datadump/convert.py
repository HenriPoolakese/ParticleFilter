import os
import json
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from geopy.distance import geodesic


class OSMConverter:
    def __init__(self):
        self.cache_file = 'x'

    def convert_and_merge(self, osm_input_file, output_file=None):
        """
        Converts OSM JSON to compatible format with UTF-8 encoding
        Args:
            osm_input_file: Path to OSM JSON file (UTF-8 encoded)
            output_file: Optional output path (default: updates cache file)
        Returns:
            NetworkX graph object

            This can be used to query data from overpass-turbo:

            [out:json][timeout:25];

            // Define the bounding box
            (
              node["highway"~"tertiary|tertiary_link"]({{bbox}});
              way["highway"~"tertiary|tertiary_link"]({{bbox}});
            );

            out body;
            >;
            out skel qt;

        """
        # Load existing graph if available
        existing_graph = self.load_or_cache_road_data() if os.path.exists(self.cache_file) else None

        # Convert new OSM data with UTF-8
        with open(osm_input_file, 'r', encoding='utf-8') as f:
            osm_data = json.load(f)
        new_graph = self.convert_osm_data(osm_data)

        # Merge graphs
        final_graph = self.merge_graphs(existing_graph, new_graph) if existing_graph else new_graph

        # Save with UTF-8
        output_path = output_file if output_file else self.cache_file
        self.save_graph_to_file(final_graph, output_path)
        return final_graph

    def convert_osm_data(self, osm_data):
        # Convert OSM data to NetworkX graph with UTF-8 compatible fields
        G = nx.MultiDiGraph()

        # Process nodes
        for element in osm_data['elements']:
            if element['type'] == 'node':
                node_attrs = {
                    'y': element['lat'],
                    'x': element['lon'],
                    'street_count': self.calculate_street_count(element['id'], osm_data)
                }
                # Add tags with UTF-8 handling
                if 'tags' in element:
                    node_attrs.update({
                        k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v
                        for k, v in element['tags'].items()
                    })
                G.add_node(element['id'], **node_attrs)

        # Process ways (edges)
        for element in osm_data['elements']:
            if element['type'] == 'way':
                tags = element.get('tags', {})
                geometry = element.get('geometry', [])

                # Prepare edge attributes with UTF-8
                edge_attrs = {
                    'key': 0,
                    'attributes': {
                        'highway': tags.get('highway', ''),
                        'length': 0.0,
                        **{k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v
                           for k, v in tags.items()
                           if k not in ['highway', 'ref']}
                    }
                }

                # Handle road references
                if 'ref' in tags:
                    edge_attrs['ref'] = tags['ref'].encode('utf-8').decode('utf-8')

                # Create edges
                for i in range(len(element['nodes']) - 1):
                    u, v = element['nodes'][i], element['nodes'][i + 1]

                    # Calculate geometry
                    if geometry and len(geometry) > i + 1:
                        line_coords = [
                            (geometry[i]['lon'], geometry[i]['lat']),
                            (geometry[i + 1]['lon'], geometry[i + 1]['lat'])
                        ]
                        edge_attrs['geometry'] = LineString(line_coords)
                        edge_attrs['attributes']['length'] = geodesic(
                            (line_coords[0][1], line_coords[0][0]),
                            (line_coords[1][1], line_coords[1][0])
                        ).meters

                    G.add_edge(u, v, **edge_attrs)
        return G

    def calculate_street_count(self, node_id, osm_data):
        #Count how many ways reference this node
        count = 0
        for element in osm_data['elements']:
            if element['type'] == 'way' and node_id in element.get('nodes', []):
                count += 1
        return count

    def merge_graphs(self, existing_graph, new_graph):
        #Merge graphs while preserving UTF-8 encoding
        merged = nx.compose(existing_graph, new_graph)
        if 'milestones' in existing_graph.graph:
            merged.graph['milestones'] = [
                {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v
                 for k, v in milestone.items()}
                for milestone in existing_graph.graph['milestones']
            ]
        return merged

    def save_graph_to_file(self, road_graph, filename=None):
        #Save graph with explicit UTF-8 encoding
        filename = filename or self.cache_file
        graph_data = self.serialize_graph(road_graph)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)

    def load_or_cache_road_data(self):
        #Load graph with UTF-8 encoding
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                road_data = json.load(f)
            return self.deserialize_graph(road_data)

        custom_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
        road_graph = ox.graph_from_place("Estonia", network_type="drive", custom_filter=custom_filter)
        self.save_graph_to_file(road_graph)
        return road_graph

    def serialize_graph(self, road_graph):
        graph_data = nx.node_link_data(road_graph, edges="links")
        for edge in graph_data["links"]:
            if "geometry" in edge and isinstance(edge["geometry"], LineString):
                edge["geometry"] = list(edge["geometry"].coords)
        return graph_data

    def deserialize_graph(self, graph_data):
        for edge in graph_data["links"]:
            if "geometry" in edge and isinstance(edge["geometry"], list):
                edge["geometry"] = LineString(edge["geometry"])
        return nx.node_link_graph(graph_data, edges="links")

converter = OSMConverter()

# For completely new data:
converter.convert_and_merge('export.json', 'updated_network.json')
print("Data converted!")

