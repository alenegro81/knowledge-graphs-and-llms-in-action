import networkx as nx
from collections import defaultdict


def compute_closeness_metrics(G):
    closeness_metrics = {}

    for node in G.nodes():
        total_distance = 0
        reachable_nodes = 0

        shortest_paths = nx.single_source_shortest_path_length(G, node)

        for other_node, distance in shortest_paths.items():
            if other_node != node:
                total_distance += distance
                reachable_nodes += 1

        n = len(G.nodes()) - 1
        if reachable_nodes > 0 and n > 0:
            # Normalize by reachable nodes to handle disconnected graphs
            closeness = (reachable_nodes / n) * (reachable_nodes / total_distance)
        else:
            closeness = 0.0

        closeness_metrics[node] = round(closeness, 2)

    return closeness_metrics


def get_node_closeness(G, node):
    metrics = compute_closeness_metrics(G)
    return metrics.get(node, 0.0)


def analyze_closeness_distribution(G):
    metrics = compute_closeness_metrics(G)
    values = list(metrics.values())

    stats = {
        'max_closeness': max(values),
        'min_closeness': min(values),
        'avg_closeness': sum(values) / len(values),
        'most_central_node': max(metrics.items(), key=lambda x: x[1])[0],
        'least_central_node': min(metrics.items(), key=lambda x: x[1])[0]
    }

    return stats