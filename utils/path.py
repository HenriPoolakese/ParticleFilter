import numpy as np
from shapely.geometry.linestring import LineString


def extract_interpolated_path(road_graph, path_nodes, resolution=10):
    """Create detailed path using segment-by-segment interpolation"""
    detailed_path = []

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if road_graph.has_edge(u, v):
            data = road_graph.get_edge_data(u, v)
            for key in data:
                edge = data[key]
                if 'geometry' in edge:
                    line = edge['geometry']
                    coordinates = list(line.coords)
                else:
                    coordinates = [
                        (road_graph.nodes[u]['x'], road_graph.nodes[u]['y']),
                        (road_graph.nodes[v]['x'], road_graph.nodes[v]['y'])
                    ]

                # Split edge into segments and interpolate each
                for i in range(len(coordinates) - 1):
                    segment = LineString([coordinates[i], coordinates[i + 1]])
                    for j in np.linspace(0, 1, resolution):
                        point = segment.interpolate(j, normalized=True)
                        detailed_path.append((point.y, point.x))

    # Remove duplicates while preserving order
    seen = set()
    return [x for x in detailed_path if not (x in seen or seen.add(x))]
