from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

def create_fraud_network():
    G = nx.Graph()

    fraudsters = ['D', 'E', 'F', 'I']

    # Add all nodes first
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

    G.add_nodes_from(nodes)

    for node in G.nodes():
        G.nodes[node]['is_fraudster'] = node in fraudsters

    edges = [
        ('A', 'B'), ('A', 'G'), ('A', 'H'), ('A', 'I'), ('A', 'O'),
        ('A', 'T'), ('B', 'D'), ('B', 'C'), ('D', 'E'), ('D', 'F'),
        ('D', 'G'), ('E', 'F'), ('F', 'G'), ('G', 'I'), ('H', 'K'),
        ('I', 'K'), ('I', 'N'), ('K', 'J'), ('L', 'M'), ('L', 'N'),
        ('N', 'M'), ('O', 'P'), ('O', 'Q'), ('Q', 'R'), ('Q', 'S')
    ]

    G.add_edges_from(edges)

    return G


def compute_degree_metrics(G):
    degree_metrics = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        total_degree = len(neighbors)

        fraud_degree = sum(1 for neighbor in neighbors
                           if G.nodes[neighbor].get('is_fraudster', False))

        legit_degree = total_degree - fraud_degree

        degree_metrics[node] = {
            'total_degree': total_degree,
            'fraud_degree': fraud_degree,
            'legit_degree': legit_degree
        }

    return degree_metrics


def get_node_degrees(G, node):
    metrics = compute_degree_metrics(G)
    return metrics.get(node, {
        'total_degree': 0,
        'fraud_degree': 0,
        'legit_degree': 0
    })

def compute_triangle_metrics(G):
    triangle_metrics = {}

    for node in G.nodes():
        triangles = []
        neighbors = list(G.neighbors(node))

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if G.has_edge(neighbors[i], neighbors[j]):
                    triangles.append((neighbors[i], neighbors[j]))

        total_triangles = len(triangles)
        fraud_triangles = 0
        legit_triangles = 0
        semi_fraud_triangles = 0

        for n1, n2 in triangles:
            n1_fraud = G.nodes[n1].get('is_fraudster', False)
            n2_fraud = G.nodes[n2].get('is_fraudster', False)

            if n1_fraud and n2_fraud:
                fraud_triangles += 1
            elif not n1_fraud and not n2_fraud:
                legit_triangles += 1
            else:
                semi_fraud_triangles += 1

        triangle_metrics[node] = {
            'total_triangles': total_triangles,
            'fraud_triangles': fraud_triangles,
            'legit_triangles': legit_triangles,
            'semi_fraud_triangles': semi_fraud_triangles
        }

    return triangle_metrics


def get_node_triangles(G, node):
    metrics = compute_triangle_metrics(G)
    return metrics.get(node, {
        'total_triangles': 0,
        'fraud_triangles': 0,
        'legit_triangles': 0,
        'semi_fraud_triangles': 0
    })

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

def compute_pagerank_metrics(G, fraud_weight=2.0, damping_factor=0.85):
    pagerank_metrics = {}

    base_pagerank = nx.pagerank(
        G,
        alpha=damping_factor,
        personalization=None,
        weight=None
    )

    fraud_personalization = {}
    for node in G.nodes():
        if G.nodes[node].get('is_fraudster', False):
            fraud_personalization[node] = fraud_weight
        else:
            fraud_personalization[node] = 1.0

    fraud_pagerank = nx.pagerank(
        G,
        alpha=damping_factor,
        personalization=fraud_personalization,
        weight=None
    )

    for node in G.nodes():
        pagerank_metrics[node] = {
            'pagerank_base': round(base_pagerank[node], 3),
            'pagerank_fraud': round(fraud_pagerank[node], 3)
        }

    return pagerank_metrics


def get_node_pagerank(G, node):
    metrics = compute_pagerank_metrics(G)
    return metrics.get(node, {
        'pagerank_base': 0.0,
        'pagerank_fraud': 0.0
    })

def create_node_features_dataset(G):
    degree_metrics = compute_degree_metrics(G)
    triangle_metrics = compute_triangle_metrics(G)
    density_metrics = compute_density_metrics(G)
    path_metrics = compute_geodesic_metrics(G)
    closeness_metrics = compute_closeness_metrics(G)
    betweenness_metrics = compute_betweenness_metrics(G)
    pagerank_metrics = compute_pagerank_metrics(G)

    features_dict = {}
    for node in G.nodes():
        features_dict[node] = {
            # Degree features
            'total_degree': degree_metrics[node]['total_degree'],
            'fraud_degree': degree_metrics[node]['fraud_degree'],
            'legit_degree': degree_metrics[node]['legit_degree'],

            # Triangle features
            'total_triangles': triangle_metrics[node]['total_triangles'],
            'fraud_triangles': triangle_metrics[node]['fraud_triangles'],
            'legit_triangles': triangle_metrics[node]['legit_triangles'],
            'semi_fraud_triangles': triangle_metrics[node]['semi_fraud_triangles'],

            # Other metrics
            'density': density_metrics[node],
            'geodesic_path': path_metrics[node]['geodesic_path'],
            'paths_1hop': path_metrics[node]['#1-hop_paths'],
            'paths_2hop': path_metrics[node]['#2-hop_paths'],
            'paths_3hop': path_metrics[node]['#3-hop_paths'],
            'closeness': closeness_metrics[node],
            'betweenness': betweenness_metrics[node],
            'pagerank_base': pagerank_metrics[node]['pagerank_base'],
            'pagerank_fraud': pagerank_metrics[node]['pagerank_fraud'],

            # Label
            'is_fraudster': G.nodes[node]['is_fraudster']
        }

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(features_dict, orient='index')

    return df

def train_fraud_classifier(G):
    df = create_node_features_dataset(G)

    X = df.drop('is_fraudster', axis=1)
    y = df['is_fraudster']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(clf.coef_[0])
    }).sort_values('importance', ascending=False)

    results = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': feature_importance,
        'model': clf,
        'scaler': scaler
    }

    return results

def predict_fraud_probability(G, node, trained_results):
    df = create_node_features_dataset(G)
    node_features = df.loc[node].drop('is_fraudster')

    scaled_features = trained_results['scaler'].transform(
        node_features.values.reshape(1, -1)
    )

    fraud_prob = trained_results['model'].predict_proba(scaled_features)[0][1]

    return fraud_prob


if __name__ == '__main__':
    G = create_fraud_network()
    results = train_fraud_classifier(G)

    print("Classification Report:")
    print(results['classification_report'])

    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    print("\nTop 5 Most Important Features:")
    print(results['feature_importance'].head())

    node_of_interest = 'A'
    prob = predict_fraud_probability(G, node_of_interest, results)
    print(f"\nFraud probability for node {node_of_interest}: {prob:.3f}")