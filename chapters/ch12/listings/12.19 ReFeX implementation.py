import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


class ReFeX:
    def __init__(self, max_iterations=2, correlation_threshold=0.95):
        self.max_iterations = max_iterations
        self.correlation_threshold = correlation_threshold

    def extract_features(self, G):
        features = self._extract_local_features(G)

        egonet_features = self._extract_egonet_features(G)
        features = np.column_stack((features, egonet_features))

        for iteration in range(self.max_iterations):
            new_features = self._generate_recursive_features(G, features)
            features = np.column_stack((features, new_features))
            features = self._prune_features(features)
        return features

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

        return features

    def _extract_egonet_features(self, G):
        n_nodes = G.number_of_nodes()
        features = np.zeros((n_nodes, 3))
        for idx, node in enumerate(G.nodes()):
            ego = nx.ego_graph(G, node, radius=1)
            features[idx, 0] = ego.number_of_nodes()  # Number of nodes in egonet
            features[idx, 1] = ego.number_of_edges()  # Number of edges in egonet
            features[idx, 2] = nx.density(ego)  # Density of egonet

        return features

    def _generate_recursive_features(self, G, current_features):
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
        return new_features

    def _prune_features(self, features):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        corr_matrix = np.corrcoef(scaled_features.T)

        to_remove = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    to_remove.add(j)

        keep_features = list(set(range(features.shape[1])) - to_remove)
        return features[:, keep_features]
