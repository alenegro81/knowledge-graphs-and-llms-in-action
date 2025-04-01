import networkx as nx
from collections import defaultdict


def compute_geodesic_metrics(G, max_hops=3):
    path_metrics = {}

    fraudster_nodes = [n for n, attr in G.nodes(data=True)
                       if attr.get('is_fraudster', False)]

    for node in G.nodes():
        if G.nodes[node].get('is_fraudster', False):
            geodesic_path = 0
            hop_counts = defaultdict(int)
        else:
            paths_to_fraudsters = []
            hop_counts = defaultdict(int)

            for fraudster in fraudster_nodes:
                try:
                    path = nx.shortest_path(G, node, fraudster)
                    path_length = len(path) - 1  # Convert to number of hops
                    paths_to_fraudsters.append(path_length)

                    if path_length <= max_hops:
                        hop_counts[path_length] += 1
                except nx.NetworkXNoPath:
                    continue

            geodesic_path = min(paths_to_fraudsters) if paths_to_fraudsters else float('inf')

        path_metrics[node] = {
            'geodesic_path': geodesic_path,
            '#1-hop_paths': hop_counts[1],
            '#2-hop_paths': hop_counts[2],
            '#3-hop_paths': hop_counts[3]
        }

    return path_metrics


def get_node_paths(G, node):
    metrics = compute_geodesic_metrics(G)
    return metrics.get(node, {
        'geodesic_path': float('inf'),
        '#1-hop_paths': 0,
        '#2-hop_paths': 0,
        '#3-hop_paths': 0
    })