import networkx as nx


def compute_density_metrics(G):
    density_metrics = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        egonet_nodes = neighbors + [node]

        N = len(egonet_nodes)

        if N < 2:  # Handle special case where egonet is too small
            density_metrics[node] = 0.0
            continue

        M = 0
        for i in range(len(egonet_nodes)):
            for j in range(i + 1, len(egonet_nodes)):
                if G.has_edge(egonet_nodes[i], egonet_nodes[j]):
                    M += 1

        max_possible_edges = (N * (N - 1)) / 2
        density = M / max_possible_edges

        density_metrics[node] = round(density, 2)

    return density_metrics


def get_node_density(G, node):
    metrics = compute_density_metrics(G)
    return metrics.get(node, 0.0)