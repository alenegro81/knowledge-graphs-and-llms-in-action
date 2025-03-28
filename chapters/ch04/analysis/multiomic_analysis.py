from pathlib import Path

from util.graphdb_base import GraphDBBase

import networkx as nx
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from util.networkx_utility import graph_undirected_from_cypher


class MultiOmicAnalysis(GraphDBBase):
    def __init__(self, argv, database):
        super().__init__(command=__file__, argv=argv)
        self.__database = database

    def get_data(self, query, param={}):
        with self._driver.session(database=self.__database) as session:
            results = session.run(query, param)
            data = pd.DataFrame(results.values(), columns=results.keys())
        return data

    def get_raw_data(self, query, param):
        with self._driver.session(database=self.__database) as session:
            results = session.run(query, param)
            return results.graph()

    def load_full_graph(self):
        query = """
            MATCH (m0:Protein)
            OPTIONAL MATCH (m0)-[r:INTERACTS_WITH]->(m1)
            return distinct m0, r, m1 
        """
        return self.load_graph_and_get_nx_graph(query)

    def load_assoc_gene_vector(self, disease):
        query = """
            MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
            return id(p) as node_id, p.id as protein_id 
        """
        param = {"id": disease}
        return self.get_data(query, param)

    def load_hd(self, disease):
        query = """
            MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
            WITH collect(p) as proteins
            UNWIND proteins as m0
            UNWIND proteins as m1
            OPTIONAL MATCH (m0)-[r:INTERACTS_WITH]->(m1)
            return distinct m0, r, m1
        """
        param = {"id": disease}
        return self.load_graph_and_get_nx_graph(query, param)

    def load_graph_and_get_nx_graph(self, query, param={}):
        data = self.get_raw_data(query, param)
        G = graph_undirected_from_cypher(data)
        return G

    def compute_largest_components(self, networkx_graph):
        largest_cc = max(nx.connected_components(networkx_graph), key=len)
        return largest_cc

    def get_list_of_diseases(self):
        query = """
            MATCH (d:Disease)
            return d.id as id, d.name as name
        """
        return self.get_data(query, {})

    def compute_bd(self, disease):
        query = """
            MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
            WITH collect(p) as proteins
            MATCH (m0)-[r:INTERACTS_WITH]-(m1)
            WHERE m0 in proteins and not m1 in proteins
            RETURN count(DISTINCT r) as bd
        """
        param = {'id': disease}

        return self.get_data(query, param)["bd"][0]


def sub_graph(G, largest_wcc):
    # Create a subgraph SG based on a (possibly multigraph) G
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
                          for n, nbrs in G.adj.items() if n in largest_wcc
                          for nbr, keydict in nbrs.items() if nbr in largest_wcc
                          for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
                          for n, nbrs in G.adj.items() if n in largest_wcc
                          for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)
    return SG


def getCommunityScores(nx_graph, assoc_gene_vector):
    adjacency_matrix = nx.adjacency_matrix(nx_graph)
    disease_properties = {}
    disease_indices = np.nonzero(assoc_gene_vector)[0]
    num_total_nodes = adjacency_matrix.shape[0]
    num_total_edges = (np.sum(adjacency_matrix) - sum(adjacency_matrix.diagonal())) / 2 + sum(
        adjacency_matrix.diagonal())

    num_nodes = len(disease_indices)
    disease_properties["num_nodes"] = num_nodes

    subgraph = nx_graph.subgraph([list(ppi_graph.nodes())[i] for i in disease_indices])
    sliced_adj_matrix = adjacency_matrix[disease_indices, :]
    num_disease_node_edges = np.sum(
        sliced_adj_matrix)  # here no need to divide by 2 since the matrix is not simmetric anymore after the slicing, the self loop are not an issue since they are counted once
    sub_adj_matrix = sliced_adj_matrix[:, disease_indices]
    num_internal_edges = (np.sum(sub_adj_matrix) - np.sum(sub_adj_matrix.diagonal())) / 2 + np.sum(
        sub_adj_matrix.diagonal())
    external_edges = num_disease_node_edges - num_internal_edges

    # Internal connectivity
    disease_properties["density"] = nx.density(subgraph)
    disease_properties["average_degree"] = 2 * num_internal_edges / num_nodes
    disease_properties["average_internal_clustering"] = nx.average_clustering(subgraph)

    conn_comps = nx.connected_components(subgraph)
    sorted_cc = [len(c) for c in sorted(conn_comps, key=len, reverse=True)]
    disease_properties["size_largest_connected_component"] = sorted_cc[0]
    disease_properties["percent_in_largest_connected_component"] = float(sorted_cc[0]) / num_nodes
    disease_properties["number_connected_components"] = len(sorted_cc)

    # External connectivity
    disease_properties["expansion"] = external_edges / num_nodes
    disease_properties["cut_ratio"] = external_edges / (num_nodes * (num_total_nodes - num_nodes))

    # External and internal connectivity
    disease_properties["conductance"] = external_edges / (2 * num_internal_edges + external_edges)
    disease_properties["normalized_cut"] = disease_properties["conductance"] + external_edges / (
            2 * (num_total_edges - num_internal_edges) + external_edges)

    return disease_properties


def create_plot(dataframe, filename):
    cm = 1 / 2.54
    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(35 * cm, 10 * cm))
    counts, bins = np.histogram(dataframe['percent_in_largest_connected_component'].values, bins=20)
    axs[0].hist(bins[:-1], bins, weights=counts, facecolor='g', align='mid', edgecolor="black", linewidth=0.4)
    # axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Occurrences', fontsize=22)
    axs[0].set_title('a) Largest CC', fontsize=22)
    axs[0].tick_params(labelsize='x-large', width=2)
    axs[0].margins(x=0.01)
    counts, bins = np.histogram(dataframe['density'].values, bins=20)
    axs[1].hist(bins[:-1], bins, weights=counts, facecolor='g', align='mid', edgecolor="black", linewidth=0.4)
    # axs[1].set_ylabel('Occurrences')
    axs[1].set_title('b) Density', fontsize=22)
    axs[1].margins(x=0.01)
    axs[1].tick_params(labelsize='x-large', width=2)
    counts, bins = np.histogram(dataframe['conductance'].values, bins=20)
    axs[2].hist(bins[:-1], bins, weights=counts, facecolor='g', align='mid', edgecolor="black", linewidth=0.4)
    # axs[2].set_ylabel('Occurrences')
    axs[2].set_title('c) Conductance', fontsize=22)
    axs[2].margins(x=0.01)
    axs[2].tick_params(labelsize='x-large', width=2)
    fig.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':
    base = Path(__file__).parent
    start = time.time()
    analysis = MultiOmicAnalysis(argv=sys.argv[1:], database="ppi")  # database should be also among the arguments
    diseases = analysis.get_list_of_diseases()
    # networkx_graph = analysis.load_hd("C0036095") #Salivary Gland Neoplasms
    # networkx_graph = analysis.load_hd("C0019693")  # HIV Infections
    results = []
    results_second_approach = []
    ppi_graph = analysis.load_full_graph()
    for index, disease in diseases.iterrows():
        disease_id = disease['id']
        assoc_gene_vector = analysis.load_assoc_gene_vector(disease_id)
        assoc_gene_vector_index = [1 if x[1]['id'] in list(assoc_gene_vector['protein_id']) else 0 for x in
                                   ppi_graph.nodes(data=True)]
        disease_property = getCommunityScores(ppi_graph, assoc_gene_vector_index)
        disease_property['id'] = disease['id']
        disease_property['name'] = disease['name']
        results.append(disease_property)

        networkx_graph = analysis.load_hd(disease_id)
        bd = analysis.compute_bd(disease_id)
        nodes_count = networkx_graph.nodes.__len__()
        edges_count = networkx_graph.edges.__len__()
        largest_cc = analysis.compute_largest_components(networkx_graph)
        largest_cc_size = largest_cc.__len__()
        relative_size_of_largest_cc = float(largest_cc_size) / nodes_count
        density_pathway = 2.0 * float(edges_count) / (nodes_count * (nodes_count - 1))
        conductance = float(bd) / (bd + 2 * edges_count)
        results_second_approach.append({
            'id': disease['id'],
            'name': disease['name'],
            'relative_size_of_largest_cc': relative_size_of_largest_cc,
            'nodes_count': nodes_count,
            'largest_cc_size': largest_cc_size,
            'density_pathway': density_pathway,
            'conductance': conductance}
        )

    df = pd.DataFrame(results)
    largest_cc_frequency = df['percent_in_largest_connected_component'].value_counts(bins=20, sort=False)
    density_frequency = df['density'].value_counts(bins=20, sort=False)
    conductance_frequency = df['conductance'].value_counts(bins=20, sort=False)

    df2 = pd.DataFrame(results_second_approach)
    largest_cc_frequency2 = df2['relative_size_of_largest_cc'].value_counts(bins=20, sort=False)
    density_frequency2 = df2['density_pathway'].value_counts(bins=20, sort=False)
    conductance_frequency2 = df2['conductance'].value_counts(bins=20, sort=False)
    create_plot(df, base/'pharma_analysis_plot.png')
    end = time.time() - start
    analysis.close()
    print("Time to complete:", end)
    print("done")
