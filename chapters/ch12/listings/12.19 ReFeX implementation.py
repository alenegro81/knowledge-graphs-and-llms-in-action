import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ReFeX:
    def __init__(self, max_iterations=2, correlation_threshold=0.95):
        self.max_iterations = max_iterations
        self.correlation_threshold = correlation_threshold

    def extract_features(self, G):
        features, features_names = self._extract_local_features(G)

        egonet_features, egonet_features_names = self._extract_egonet_features(G)

        features = np.column_stack((features, egonet_features))
        features_names = np.hstack((features_names, egonet_features_names))

        for iteration in range(self.max_iterations):
            new_features, new_features_names = self._generate_recursive_features(G, features, features_names)

            features = np.column_stack((features, new_features))
            features_names = np.hstack((features_names, new_features_names))

            features, features_names = self._prune_features(features, features_names)
        return features, features_names

    def _extract_local_features(self, G):
        """Extract local (node-level) features"""
        n_nodes = G.number_of_nodes()
        features = np.zeros((n_nodes, 3))

        for idx, node in enumerate(G.nodes()):
            # Degree features
            features[idx, 0] = G.degree(node)

            # In-degree and out-degree for directed graphs
            if G.is_directed():
                features[idx, 1] = G.in_degree(node)
                features[idx, 2] = G.out_degree(node)
            else:
                features[idx, 1] = features[idx, 2] = G.degree(node)

        return features, np.array(["degree", "in_degree", "out_degree"])

    def _extract_egonet_features(self, G):
        n_nodes = G.number_of_nodes()
        features = np.zeros((n_nodes, 3))
        for idx, node in enumerate(G.nodes()):
            ego = nx.ego_graph(G, node, radius=1)
            features[idx, 0] = ego.number_of_nodes()  # Number of nodes in egonet
            features[idx, 1] = ego.number_of_edges()  # Number of edges in egonet
            features[idx, 2] = nx.density(ego)  # Density of egonet

        return features, np.array(["ego_nodes", "ego_edges", "ego_density"])

    def _generate_recursive_features(self, G, current_features, current_features_names):
        n_nodes = G.number_of_nodes()
        n_features = current_features.shape[1]
        new_features = np.zeros((n_nodes, n_features * 2))

        for idx, node in enumerate(G.nodes()):
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            neighbor_feats = current_features[[list(G.nodes()).index(n) for n in neighbors]]
            new_features[idx, :n_features] = np.sum(neighbor_feats, axis=0)
            new_features[idx, n_features:] = np.mean(neighbor_feats, axis=0)

        new_features_names = np.hstack([np.char.add(current_features_names, ".sum"),
                                        np.char.add(current_features_names, ".mean")])
        return new_features, new_features_names

    def _prune_features(self, features, features_names):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        corr_matrix = np.corrcoef(scaled_features.T)

        to_remove = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    to_remove.add(j)

        keep_features = list(set(range(features.shape[1])) - to_remove)
        return features[:, keep_features], features_names[keep_features]


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


def main():
    G = create_fraud_network()
    refex = ReFeX()
    data, columns = refex.extract_features(G)
    # create a DataFrame using nodes and features names as rows and coumns labels
    features = pd.DataFrame(data, columns=columns, index=G.nodes)
    pd.set_option('display.max_columns', None)
    print(features)


if __name__ == '__main__':
    main()
