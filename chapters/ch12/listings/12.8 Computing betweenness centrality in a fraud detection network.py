import networkx as nx
from collections import defaultdict


def compute_betweenness_metrics(G, normalized=True):
    betweenness_metrics = {}

    betweenness = nx.betweenness_centrality(
        G,
        normalized=normalized,
        endpoints=False
    )

    for node in G.nodes():
        betweenness_metrics[node] = round(betweenness[node], 3)

    return betweenness_metrics


def analyze_betweenness_distribution(G):
    metrics = compute_betweenness_metrics(G)
    values = list(metrics.values())

    return {
        'max_betweenness': max(values),
        'min_betweenness': min(values),
        'avg_betweenness': sum(values) / len(values),
        'key_bridges': [node for node, score in metrics.items()
                        if score > sum(values) / len(values)]
    }


def get_node_betweenness(G, node):
    metrics = compute_betweenness_metrics(G)
    return metrics.get(node, 0.0)


def identify_potential_bottlenecks(G, threshold=0.5):
    metrics = compute_betweenness_metrics(G)

    bottlenecks = {node: score for node, score in metrics.items()
                   if score > threshold}
    return bottlenecks