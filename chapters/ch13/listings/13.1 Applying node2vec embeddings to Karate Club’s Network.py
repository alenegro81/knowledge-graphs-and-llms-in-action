import networkx as nx
import numpy as np
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

G = nx.karate_club_graph()

faction_labels = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
    9: 1, 10: 1, 11: 0, 12: 0, 13: 0, 14: 1, 15: 1, 16: 0,
    17: 0, 18: 1, 19: 0, 20: 1, 21: 0, 22: 1, 23: 1, 24: 1,
    25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1,
    33: 1
}
nx.set_node_attributes(G, faction_labels, 'faction')

node2vec = Node2Vec(
    G,
    dimensions=16,
    walk_length=10,
    num_walks=20,
    p=1,
    q=1,
    workers=4
)

model = node2vec.fit(window=5)


def decode_similarity(model, node1, node2):
    return model.wv.similarity(str(node1), str(node2))


def visualize_graph_and_embeddings(model, G):
    embeddings = np.zeros((len(G.nodes()), model.vector_size))
    for i, node in enumerate(G.nodes()):
        embeddings[i] = model.wv[str(node)]

    tsne = TSNE(n_components=2, random_state=42)
    node_pos_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left plot: original graph
    pos = nx.spring_layout(G, seed=42)
    colors = ['red' if G.nodes[node]['faction'] == 0 else 'blue'
              for node in G.nodes()]
    nx.draw(
        G,
        pos,
        ax=axes[0],
        with_labels=True,
        node_color=colors,
        node_size=500,
        font_size=10
    )
    axes[0].set_title("Original Karate Club Graph")

    # Right plot: node embeddings
    axes[1].scatter(node_pos_2d[:, 0], node_pos_2d[:, 1], c=colors, s=100)
    for i, node in enumerate(G.nodes()):
        axes[1].annotate(str(node), (node_pos_2d[i, 0], node_pos_2d[i, 1]), fontsize=9)
    axes[1].set_title("Node2Vec Embeddings (t-SNE)")

    plt.tight_layout()
    plt.show()


print("Similarity between instructor (0) and their close ally (1):",
      decode_similarity(model, 0, 1))
print("Similarity between instructor (0) and president (33):",
      decode_similarity(model, 0, 33))

visualize_graph_and_embeddings(model, G)


def analyze_community_structure(model, G):
    instructor_allies = []
    president_allies = []

    for node in G.nodes():
        if node not in [0, 33]:
            sim_to_instructor = decode_similarity(model, node, 0)
            sim_to_president = decode_similarity(model, node, 33)

            if sim_to_instructor > sim_to_president:
                instructor_allies.append(node)
            else:
                president_allies.append(node)

    return instructor_allies, president_allies


instructor_group, president_group = analyze_community_structure(model, G)
print("\nPredicted instructor's faction:", sorted(instructor_group))
print("Predicted president's faction:", sorted(president_group))
